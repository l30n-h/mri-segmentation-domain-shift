import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import os
import glob
import helpers as hlp

import scipy
import skimage
import sklearn
import sklearn.decomposition
import sklearn.cluster
import sklearn.mixture

def make_receptive2d(interim, kernel=[3,3], stride=[2,2], makeflat=True):
    ## interim [batch, X, Y, channels]
    features = np.pad(
        interim, (
            (0, 0),
            (np.floor((kernel[0]-1) / 2.0).astype(int), np.ceil((kernel[0]-1) / 2.0).astype(int)),
            (np.floor((kernel[1]-1) / 2.0).astype(int), np.ceil((kernel[1]-1) / 2.0).astype(int)),
            (0,0)
        ), 'constant')
    windows = skimage.util.view_as_windows(
        features,
        window_shape=[1, *kernel, interim.shape[-1]],
        step=[1, *stride, interim.shape[-1]]
    ).squeeze((3,4))
    windows = windows.reshape([
        *windows.shape[0:3],
        np.product(windows.shape[3:5]),
        windows.shape[-1]
    ])
    # windows.shape [num_batch, num_windows_x, num_windows_y, num_activations_per_window, num_features]
    if makeflat:
        return windows.reshape([
            windows.shape[0],
            np.product(windows.shape[1:3]),
            *windows.shape[-2:]
        ])
        # windows.shape [num_batch, num_windows_total, num_activations_per_window, num_features]
    return windows

def align_and_subsample(src, src_gt, dst, dst_gt, MAX_SAMPLES=60000, kernel_size=[3,3], stride=[2,2]):
    ## ALIGN DATA USING RECEPTIVE FIELD
    pt_src = make_receptive2d(src, kernel=kernel_size, stride=stride, makeflat=True)
    pt_src = pt_src.reshape([*pt_src.shape[0:2],-1])
    pt_src_gt = make_receptive2d(src_gt, kernel=kernel_size, stride=stride, makeflat=True).max(axis=(2))
    
    pt_dst = dst.reshape([dst.shape[0], np.product(dst.shape[1:3]), -1])
    pt_dst_gt = dst_gt.reshape([dst_gt.shape[0], np.product(dst_gt.shape[1:3]), -1])

    ## SUB-SAMPLE DATA
    num_per_example = MAX_SAMPLES // pt_src.shape[0]
    sample_idx = []
    for b in range(pt_src.shape[0]):
        uniq = np.unique(pt_src_gt[b])
        # print(uniq)
        num_per_class = num_per_example // len(uniq)
        for lab in uniq:
            labmask = np.argwhere(pt_src_gt[b]==lab)[:,0]
            sampsidx = np.random.choice(labmask, size=min(num_per_class,labmask.shape[0]))
            sampsidx = np.stack( [np.repeat(b,len(sampsidx)), sampsidx], axis=0).T
            sample_idx.append(sampsidx)

    sample_idx = np.concatenate(sample_idx,axis=0)
    pt_src = pt_src[sample_idx[:,0],sample_idx[:,1]]
    pt_src_gt = pt_src_gt[sample_idx[:,0],sample_idx[:,1]]
    pt_dst = pt_dst[sample_idx[:,0],sample_idx[:,1]]
    pt_dst_gt = pt_dst_gt[sample_idx[:,0],sample_idx[:,1]]

    return pt_src, pt_src_gt, pt_dst, pt_dst_gt

def learn_manifold(pt, PCA_components, kmeans_components):
    pca = sklearn.decomposition.PCA(PCA_components)
    pt_transformed = pca.fit_transform(pt)
    kmeans = sklearn.cluster.KMeans(n_clusters=min(kmeans_components,pt_transformed.shape[0]//30))
    clustidx = kmeans.fit_predict(pt_transformed)
    cluster_masks = [(clustidx==k).squeeze() for k in np.unique(clustidx)]
    mixtures = [sklearn.mixture.GaussianMixture(n_components=min(3,mask.sum())).fit(pt_transformed[mask]) for mask in cluster_masks]
    return pca, kmeans, mixtures

def project_manifold(pt, pca, kmeans, mixtures):
    pt_transformed = pca.transform(pt)
    clustidx = kmeans.predict(pt_transformed)
    gmm_confidence = np.stack([m.score_samples(pt_transformed) for m in mixtures],axis=1)
    return pt_transformed, clustidx, gmm_confidence

def compute_confidence(mixtures_src, pt_src_transformed, pt_dst_transformed):
    scores = np.stack(
        [ m.score_samples(pt_src_transformed) for m in mixtures_src ],
        axis=1
    )
    scores = scipy.special.softmax(scores, axis=1).max(axis=1)
    score_weight = np.linalg.norm(pt_dst_transformed, axis=1)
    return scores, score_weight

def train_roughness_and_confidence(data_src, data_src_gt, data_dst, data_dst_gt, kernel_size, stride, max_samples, pca_components=0.8, kmeans_components=5):
    data_src = data_src.movedim(1, -1).numpy()
    data_src_gt = data_src_gt.movedim(1, -1).numpy()
    data_dst = data_dst.movedim(1, -1).numpy()
    data_dst_gt = data_dst_gt.movedim(1, -1).numpy()

    pt_src, pt_src_gt, pt_dst_pred, pt_dst_gt = align_and_subsample(
        data_src,
        data_src_gt,
        data_dst,
        data_dst_gt,
        MAX_SAMPLES=max_samples,
        kernel_size=kernel_size,
        stride=stride
    )
    pca_src, kmeans_src, mixtures_src = learn_manifold(
        pt_src,
        min(pca_components, pt_src.shape[1]),
        kmeans_components
    )
    pca_dst, kmeans_dst, mixtures_dst = learn_manifold(
        pt_dst_pred,
        min(pca_components, pt_dst_pred.shape[1]),
        kmeans_components
    )
    return (
        (pca_src, kmeans_src, mixtures_src),
        (pca_dst, kmeans_dst, mixtures_dst)
    )

def get_roughness_and_confidence(data_src, data_src_gt, data_dst, data_dst_gt, kernel_size, stride, max_samples, pca_src, kmeans_src, mixtures_src, pca_dst, kmeans_dst, mixtures_dst):
    data_src = data_src.movedim(1, -1).numpy()
    data_src_gt = data_src_gt.movedim(1, -1).numpy()
    data_dst = data_dst.movedim(1, -1).numpy()
    data_dst_gt = data_dst_gt.movedim(1, -1).numpy()

    test_src, test_src_gt, test_dst_pred, test_dst_gt = align_and_subsample(
        data_src,
        data_src_gt,
        data_dst,
        data_dst_gt,
        MAX_SAMPLES=max_samples,
        kernel_size=kernel_size,
        stride=stride
    )
    pt_src_transformed, _, _ = project_manifold(test_src, pca_src, kmeans_src, mixtures_src)
    pt_dst_transformed, dst_clustidx, _ = project_manifold(test_dst_pred, pca_dst, kmeans_dst, mixtures_dst)

    ## COMPUTE CLUSTERING USING PT_PRED --> This is the roughness metric
    try:
        roughness = sklearn.metrics.davies_bouldin_score(
            pt_src_transformed,
            dst_clustidx
        )
    except ValueError as e:
        # happens if dst_clustidx has only one unique value
        print(e)
        roughness = 0

    ## COMPUTE CONFIDENCE -->
    scores, score_weight = compute_confidence(mixtures_src, pt_src_transformed, pt_dst_transformed)
    scores_weighted = scores * score_weight
    confidence = scores.mean()
    confidence_weighted = scores_weighted.mean()

    return roughness, confidence, confidence_weighted




def get_pca_scores(pca):
    variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
    return {
        'noise_variance': pca.noise_variance_,
        'variance_ratio_cumsum_index_gt_90': np.searchsorted(variance_ratio_cumsum, 0.90),
        'variance_ratio_cumsum_index_gt_95': np.searchsorted(variance_ratio_cumsum, 0.95),
        'variance_ratio_cumsum_index_gt_98': np.searchsorted(variance_ratio_cumsum, 0.98),
        'variance_ratio_cumsum_index_gt_99': np.searchsorted(variance_ratio_cumsum, 0.99),
        'variance_ratio_cumsum_index_max': len(variance_ratio_cumsum),
        'variance_ratio_cumsum_min': variance_ratio_cumsum[0],
        'variance_ratio_cumsum_max': variance_ratio_cumsum[-1],
        'variance_ratio_cumsum_mean': variance_ratio_cumsum.mean(),
        'variance_ratio_cumsum_std': variance_ratio_cumsum.std(),
        'variance_ratio_cumsum_skewness': hlp.standardized_moment(torch.from_numpy(variance_ratio_cumsum), 3, dim=0).item(),
        'variance_ratio_cumsum_kurtosis': hlp.standardized_moment(torch.from_numpy(variance_ratio_cumsum), 4, dim=0).item()
    }

import gc
def load_data_live(task, trainer, fold_train, epoch, layers, dataset_keys, batches_per_scan=64):
    layers_set = set(layers)
    def is_module_tracked(name, module):
        return name in layers_set

    def extract_activations_full(name, activations, slices_and_tiles_count, activations_dict):
        activations_dict.setdefault(name, []).append(activations[0].clone())
        return activations_dict

    def merge_activations_dict(activations_dict):
        return dict(map(
            lambda item: (
                item[0],
                torch.stack(item[1]).cpu()
            ),
            activations_dict.items()
        ))

    tmodel = hlp.load_model(trainer, fold_train, epoch)
    hlp.apply_prediction_data_filter_monkey_patch(tmodel, batches_per_scan=batches_per_scan)

    out = hlp.get_activations_dicts_merged([ activations_dict for id, prediction, activations_dict in hlp.generate_predictions_ram(
        trainer=tmodel,
        dataset_keys=dataset_keys,
        activations_extractor_kwargs=dict(
            extract_activations=extract_activations_full,
            is_module_tracked=is_module_tracked,
            merge_activations_dict=merge_activations_dict
        )
    )])

    #TODO gc seems to be too slow sometimes
    del tmodel
    gc.collect()
    
    return out



def generate_roughness_and_confidence_stats_per_model(task, trainer, fold_train, epoch, scores):
    # paper/code batchsize 45 (val 5, test 50) volsize [64,64,16] max_samples 80000 => per vol 80000/50 = 1600 patches => 1600/(64*64*16) = 0.024
    # here batchsize 6 scans with 48 batches volsize [256,192] => 280000 ( > 0.02 * (256*192) * (12*24) ) needed?
    SCANS_PER_FOLD=6
    BATCHES_PER_SCAN=48
    MAX_SAMPLES=40000
    PCA_COMPONENTS=10 #0.8 #
    KMEANS_COMPONENTS=5
    LAYERS = list(filter(
        lambda x: '.lrelu' in x or 'tu.' in x or 'seg_outputs.3' in x,
        hlp.get_layers_ordered()
    ))
    scores = scores[(scores['fold_test'] == fold_train) | (~scores['is_validation'])]

    def load_data(keys):
        return load_data_live(task, trainer, fold_train, epoch, LAYERS, keys, batches_per_scan=BATCHES_PER_SCAN)

    def generate_manifolds_per_layer():
        scores_sub = scores[
            (scores['fold_test'] == fold_train) & (~scores['is_validation'])
        ].sort_values('id').head(SCANS_PER_FOLD)
        activations_dict = load_data(scores_sub['id_long'])
        data_gt = activations_dict['gt']

        print('train', fold_train)
        for layer_cur in LAYERS:
            print(layer_cur)
            data_src = hlp.get_activations_input(layer_cur.replace('.lrelu', '.conv'), activations_dict)
            data_dst = activations_dict[layer_cur]

            #data_src_gt = torch.zeros_like(data_src[:,:1,:,:])
            #data_dst_gt = torch.zeros_like(data_dst[:,:1,:,:])
            data_src_gt = torch.nn.functional.interpolate(data_gt, data_src.shape[2:], mode='bilinear').ceil()
            data_dst_gt = torch.nn.functional.interpolate(data_gt, data_dst.shape[2:], mode='bilinear').ceil()

            layer_config = hlp.get_layer_config(layer_cur)
            
            yield layer_cur, train_roughness_and_confidence(
                data_src=data_src.float(),
                data_src_gt=data_src_gt.float(),
                data_dst=data_dst.float(),
                data_dst_gt=data_dst_gt.float(),
                kernel_size=layer_config['kernel_size'],
                stride=layer_config['stride'],
                max_samples=MAX_SAMPLES,
                pca_components=PCA_COMPONENTS,
                kmeans_components=KMEANS_COMPONENTS
            )


    manifolds_per_layer = dict(generate_manifolds_per_layer())

    for (fold_test, is_validation) in scores[['fold_test', 'is_validation']].drop_duplicates().values.tolist():
        scores_sub = scores[
            (scores['fold_test'] == fold_test) & (scores['is_validation'] == is_validation)
        ].sort_values('id').head(SCANS_PER_FOLD)
        activations_dict = load_data(scores_sub['id_long'])

        print('evaluate', fold_test, is_validation)
        for layer_cur, (manifold_src, manifold_dst) in manifolds_per_layer.items():
            print(layer_cur)
            pca_src, kmeans_src, mixtures_src = manifold_src
            pca_dst, kmeans_dst, mixtures_dst = manifold_dst
            
            data_gt = activations_dict['gt']
            data_src = hlp.get_activations_input(layer_cur.replace('.lrelu', '.conv'), activations_dict)
            data_dst = activations_dict[layer_cur]
            data_gt = activations_dict['gt']

            #TODO zeros should by used here instead of gt in my opinion
            #data_src_gt = torch.zeros_like(data_src[:,:1,:,:])
            #data_dst_gt = torch.zeros_like(data_dst[:,:1,:,:])
            data_src_gt = torch.nn.functional.interpolate(data_gt, data_src.shape[2:], mode='bilinear').ceil()
            data_dst_gt = torch.nn.functional.interpolate(data_gt, data_dst.shape[2:], mode='bilinear').ceil()
            
            layer_config = hlp.get_layer_config(layer_cur)

            roughness, confidence, confidence_weighted = get_roughness_and_confidence(
                data_src=data_src.float(),
                data_src_gt=data_src_gt.float(),
                data_dst=data_dst.float(),
                data_dst_gt=data_dst_gt.float(),
                kernel_size=layer_config['kernel_size'],
                stride=layer_config['stride'],
                max_samples=MAX_SAMPLES * 2,
                pca_src=pca_src,
                kmeans_src=kmeans_src,
                mixtures_src=mixtures_src,
                pca_dst=pca_dst,
                kmeans_dst=kmeans_dst,
                mixtures_dst=mixtures_dst
            )
            


            train_metrics = {
                **{ 
                    'pca_src_{}'.format(key): value for key, value in get_pca_scores(pca_src).items()
                },
                **{ 
                    'pca_dst_{}'.format(key): value for key, value in get_pca_scores(pca_dst).items()
                },
                'kmeans_src_inertia': kmeans_src.inertia_,
                'kmeans_src_n_iter': kmeans_src.n_iter_,
                'kmeans_dst_inertia': kmeans_dst.inertia_,
                'kmeans_dst_n_iter': kmeans_dst.n_iter_
            }
            yield {
                'fold_test': fold_test,
                'is_validation': is_validation,
                'name': layer_cur,
                'roughness': roughness,
                'confidence': confidence,
                'confidence_weighted': confidence_weighted,
                'iou_score_mean': scores_sub['iou_score'].mean(),
                'iou_score_std': scores_sub['iou_score'].std(),
                'dice_score_mean': scores_sub['dice_score'].mean(),
                'dice_score_std': scores_sub['dice_score'].std(),
                'sdice_score_mean': scores_sub['sdice_score'].mean(),
                'sdice_score_std': scores_sub['sdice_score'].std(),
                **train_metrics
            }

def calc_roughness_and_confidence_grouped(task, trainers, folds, epochs):
    output_dir = 'data/csv/activations-roughness-and-confidence'
    os.makedirs(output_dir, exist_ok=True)

    for trainer, fold_train, epoch in itertools.product(trainers, folds, epochs):
        print(task, trainer, fold_train, epoch)
        
        scores = hlp.get_scores(task, trainer, fold_train, epoch)
        
        stats = pd.DataFrame(
            generate_roughness_and_confidence_stats_per_model(
                task,
                trainer,
                fold_train,
                epoch,
                scores
            )
        )
        stats['trainer'] = trainer
        stats['fold_train'] = fold_train
        stats['epoch'] = epoch

        stats.to_csv(
            os.path.join(
                output_dir,
                'activations-roughness-and-confidence-grouped-relu-and-residual{}-{}-{}-{}.csv'.format(
                    '-pca10c-gt-s6-b48-testaug',
                    trainer,
                    fold_train,
                    epoch
                )
            )
        )


def plot_roughness_and_confidence_grouped():
    stats = pd.concat([
        #pd.read_csv(path) for path in glob.iglob('data/csv/activations-roughness-and-confidence/activations-roughness-and-confidence-grouped-relu-and-residual-pca10c-gt-s12-b24*.csv', recursive=False)
        pd.read_csv(path) for path in glob.iglob('data/csv/activations-roughness-and-confidence/activations-roughness-and-confidence-grouped-relu-and-residual-pca10c-gt-s6-b48-testaug*.csv', recursive=False)
    ]).reset_index(drop=True)

    scores = pd.concat([
        hlp.get_scores(
            'Task601_cc359_all_training',
            *ids
        ) for ids in stats[['trainer', 'fold_train', 'epoch']].drop_duplicates().values.tolist()
    ])[['trainer', 'optimizer', 'wd', 'DA', 'bn']].drop_duplicates().reset_index(drop=True)
    stats = stats.join(scores.set_index('trainer'), on='trainer')

    stats.rename(columns={'name':'layer'}, inplace = True)
    layers_position_map = hlp.get_layers_position_map()
    stats['layer_pos'] = stats['layer'].apply(lambda d: layers_position_map.get(d))
    stats['trainer_short'] = stats['trainer'].apply(hlp.get_trainer_short)
    stats['same_domain'] = stats['fold_train'] == stats['fold_test']
    stats['trained_on'] = stats['same_domain'] & (~stats['is_validation'])
    stats['domain_val'] = stats['same_domain'].apply(lambda x: 'same' if x else 'other') + ' ' + stats['is_validation'].apply(lambda x: 'validation' if x else '')
    stats['wd_bn'] = stats['wd'].apply(lambda x: 'wd=' + str(x)) + ' ' + stats['bn'].apply(lambda x: 'bn=' + str(x))
    stats['optimizer_wd_bn'] = stats['optimizer'] + ' ' + stats['wd_bn']

    stats_weights = pd.concat([
        pd.read_csv(path) for path in glob.iglob('data/csv/weights/weights-*.csv', recursive=False)
    ]).reset_index(drop=True)
    stats_weights['name'] = stats_weights['name'].fillna('all')
    # take .conv weights if instancenorm was last layer
    stats_weights = stats_weights[~stats_weights['name'].str.endswith('.lrelu')]
    stats_weights['name'] = stats_weights['name'].str.replace('.conv', '.lrelu', regex=True)
    stats_weights = stats_weights[['trainer', 'fold_train', 'epoch', 'name', 'norm_frob_weight', 'norm_spectral']]
    stats = stats.join(
        stats_weights.set_index(['trainer', 'fold_train', 'epoch', 'name']),
        on=['trainer', 'fold_train', 'epoch', 'layer']
    )
    for column in ['roughness', 'confidence', 'confidence_weighted']:
        stats[column + '_frob'] = stats[column] / stats['norm_frob_weight']
        stats[column + '_spectral'] = stats[column] / stats['norm_spectral']



    stats = stats[~stats['wd']]
    #subdir = 'multi/train-test-all'
    subdir = 'testaug'

    # TODO
    stats = stats[~stats['layer'].str.startswith('conv_blocks_context.0.blocks.0')]
    stats = stats[~stats['layer'].str.startswith('seg_outputs')]
    stats = stats[~stats['layer'].str.startswith('tu')]
    stats = stats[stats['domain_val'] != 'other validation']
    # 

    

    print(stats)

    output_dir = os.path.join(
        'data/fig/activations-roughness-and-confidence/',
        subdir
    )
    os.makedirs(output_dir, exist_ok=True)

    add_async_task, join_async_tasks = hlp.get_async_queue(num_threads=12)

    columns_measurements = [
        'roughness',
        #'confidence',
        'confidence_weighted',
        'roughness_frob',
        #'confidence_frob',
        'confidence_weighted_frob',
        #'roughness_spectral',
        #'confidence_spectral',
        #'confidence_weighted_spectral'
    ]
    columns_measurements_train = [
        'pca_src_noise_variance',
        #'pca_src_variance_ratio_cumsum_index_gt_90',
        #'pca_src_variance_ratio_cumsum_index_gt_95',
        #'pca_src_variance_ratio_cumsum_index_gt_98',
        #'pca_src_variance_ratio_cumsum_index_gt_99',
        #'pca_src_variance_ratio_cumsum_index_max',
        #'pca_src_variance_ratio_cumsum_min',
        'pca_src_variance_ratio_cumsum_max',
        'pca_src_variance_ratio_cumsum_mean',
        #'pca_src_variance_ratio_cumsum_std',
        #'pca_src_variance_ratio_cumsum_skewness',
        #'pca_src_variance_ratio_cumsum_kurtosis',

        'pca_dst_noise_variance',
        #'pca_dst_variance_ratio_cumsum_index_gt_90',
        #'pca_dst_variance_ratio_cumsum_index_gt_95',
        #'pca_dst_variance_ratio_cumsum_index_gt_98',
        #'pca_dst_variance_ratio_cumsum_index_gt_99',
        #'pca_dst_variance_ratio_cumsum_index_max',
        #'pca_dst_variance_ratio_cumsum_min',
        'pca_dst_variance_ratio_cumsum_max',
        'pca_dst_variance_ratio_cumsum_mean',
        #'pca_dst_variance_ratio_cumsum_std',
        #'pca_dst_variance_ratio_cumsum_skewness',
        #'pca_dst_variance_ratio_cumsum_kurtosis',

        'norm_frob_weight',
        'norm_spectral',
    ]
    stats_meaned_over_layer = stats.groupby(['trainer_short', 'fold_train', 'epoch', 'fold_test', 'domain_val']).agg(
        trained_on=('trained_on', 'first'),
        iou_score_mean=('iou_score_mean', 'first'),
        dice_score_mean=('dice_score_mean', 'first'),
        sdice_score_mean=('sdice_score_mean', 'first'),
        optimizer=('optimizer', 'first'),
        optimizer_wd_bn=('optimizer_wd_bn', 'first'),
        **{
            '{}_mean'.format(column): (column, 'mean') for column in columns_measurements
        },
        **{
            '{}_mean'.format(column): (column, 'mean') for column in columns_measurements_train
        }
    ).reset_index()

    for measurement, score in itertools.product(columns_measurements, ['iou_score_mean', 'sdice_score_mean']):
        print(measurement)
        hlp.plot_scattered_and_layered(add_async_task, stats, stats_meaned_over_layer, measurement, output_dir, row=None, score=score, palette='rainbow')
        hlp.plot_scattered_and_layered(add_async_task, stats, stats_meaned_over_layer, measurement, output_dir, row='trainer_short', score=score, palette='rainbow')
        hlp.plot_scattered_and_layered(add_async_task, stats, stats_meaned_over_layer, measurement, output_dir, row='optimizer', score=score, palette='rainbow')
        hlp.plot_scattered_and_layered(add_async_task, stats, stats_meaned_over_layer, measurement, output_dir, row='optimizer_wd_bn', score=score, palette='rainbow')

    
    
    stats_train = stats.groupby(['trainer_short', 'fold_train', 'epoch', 'layer_pos']).agg(
        iou_score_mean=('iou_score_mean', 'mean'),
        dice_score_mean=('dice_score_mean', 'mean'),
        sdice_score_mean=('sdice_score_mean', 'mean'),
        optimizer=('optimizer', 'first'),
        optimizer_wd_bn=('optimizer_wd_bn', 'first'),
        **{
            column: (column, 'mean') for column in columns_measurements_train
        }
    ).reset_index()
    stats_train_meaned_over_layer = stats.groupby(['trainer_short', 'fold_train', 'epoch']).agg(
        iou_score_mean=('iou_score_mean', 'mean'),
        dice_score_mean=('dice_score_mean', 'mean'),
        sdice_score_mean=('sdice_score_mean', 'mean'),
        optimizer=('optimizer', 'first'),
        optimizer_wd_bn=('optimizer_wd_bn', 'first'),
        **{
            '{}_mean'.format(column): (column, 'mean') for column in columns_measurements_train
        }
    ).reset_index()
    print(stats_train)
    print(stats_train_meaned_over_layer)

    # for measurement, score in itertools.product(columns_measurements_train, ['iou_score_mean', 'sdice_score_mean']):
    #     print(measurement)
    #     hlp.plot_scattered_and_layered(add_async_task, stats_train, stats_train_meaned_over_layer, measurement, output_dir, row=None, score=score)
    #     hlp.plot_scattered_and_layered(add_async_task, stats_train, stats_train_meaned_over_layer, measurement, output_dir, row='trainer_short', score=score)
    #     hlp.plot_scattered_and_layered(add_async_task, stats_train, stats_train_meaned_over_layer, measurement, output_dir, row='optimizer', score=score)
    #     hlp.plot_scattered_and_layered(add_async_task, stats_train, stats_train_meaned_over_layer, measurement, output_dir, row='optimizer_wd_bn', score=score)

    join_async_tasks()



task = 'Task601_cc359_all_training'
trainers = [
    # 'nnUNetTrainerV2_MA_noscheduler_depth5_SGD_ep120__nnUNetPlansv2.1',
    # 'nnUNetTrainerV2_MA_noscheduler_depth5_SGD_ep120_noDA__nnUNetPlansv2.1',
    # 'nnUNetTrainerV2_MA_noscheduler_depth5_ep120__nnUNetPlansv2.1',
    # 'nnUNetTrainerV2_MA_noscheduler_depth5_ep120_noDA__nnUNetPlansv2.1',

    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120_noDA__nnUNetPlansv2.1',
    # 'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120_nogamma__nnUNetPlansv2.1',
    # 'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120_nomirror__nnUNetPlansv2.1',
    # 'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120_norotation__nnUNetPlansv2.1',
    # 'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120_noscaling__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120_noDA__nnUNetPlansv2.1',
    # 'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120_nogamma__nnUNetPlansv2.1',
    # 'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120_nomirror__nnUNetPlansv2.1',
    # 'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120_norotation__nnUNetPlansv2.1',
    # 'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120_noscaling__nnUNetPlansv2.1',

    # 'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_bn_SGD_ep120__nnUNetPlansv2.1',
    # 'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_bn_SGD_ep120_noDA__nnUNetPlansv2.1',
    # 'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_bn_ep120__nnUNetPlansv2.1',
    # 'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_bn_ep120_noDA__nnUNetPlansv2.1',
]
folds = [
    'siemens15',
    'siemens3',
    'ge15',
    'ge3',
    # 'philips15',
    # 'philips3'
]
epochs = [10,20,30,40,80,120]


#calc_roughness_and_confidence_grouped(task, trainers, folds, epochs)
plot_roughness_and_confidence_grouped()


#TODO paper uses 4 differnt losses and store after each of 32 epochs => 128 models
#     here 2 different optimizer 2 epochs 4 train folds => 16 models each with 2 * 6 test folds => 192 data points
#TODO paper uses 198 "label-free" testing exams
#     here 12 (scans per fold) * 4 (slices) => 48 testing exams
#  but how can they use 198 testing exams???
#  they use a dataset of 973 3D mri scans
#  split 80% training 20% test
#  => max number of test exams = 0.2 * 973 = 194.6!!!
#  patches???
#TODO paper uses mask in test data to sample well distributed samples. Still label free??
#TODO which layers to consider?
#     model 600 referenced in paper seems to define a layer as conv-norm-relu...
#     here only .conv .tu .seg_outputs
#TODO number of pca components needed for 80% variance seems to correlate with epochs and optimizer
#     at least its lower for adam than for sgd
#     its lower for 120 epochs for SGD and higher for 120 epochs for Adam
#     its slightly lower for both if DA is used (execept sgd ge3)
#TODO check for seg_output shift