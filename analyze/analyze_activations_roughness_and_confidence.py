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


def plot_roughness_and_confidence_grouped(stats_glob, output_dir):
    stats = pd.concat([
        pd.read_csv(path) for path in glob.iglob(stats_glob, recursive=False)
    ]).reset_index(drop=True)
    stats.rename(
        columns={
            'name': 'layer',
            'iou_score_mean': 'iou_score_mean_sampled',
            'iou_score_std': 'iou_score_std_sampled',
            'dice_score_mean': 'dice_score_mean_sampled',
            'dice_score_std': 'dice_score_std_sampled',
            'sdice_score_mean': 'sdice_score_mean_sampled',
            'sdice_score_std': 'sdice_score_std_sampled',
        },
        inplace=True
    )
    layers_position_map = hlp.get_layers_position_map()
    layers_position_map['input'] = -2
    layers_position_map['gt'] = -1
    stats['layer_pos'] = stats['layer'].apply(lambda d: layers_position_map.get(d))

    index = ['trainer', 'fold_train', 'epoch']
    scores = pd.concat([
        hlp.get_scores(
            'Task601_cc359_all_training',
            *ids
        ) for ids in stats[index].drop_duplicates().values.tolist()
    ]).reset_index(drop=True)
    index = [*index, 'fold_test', 'is_validation']
    scores = scores.groupby(index).agg(
        fold_test_base=('fold_test_base', 'first'),
        optimizer=('optimizer', 'first'),
        wd=('wd', 'first'),
        DA=('DA', 'first'),
        bn=('bn', 'first'),
        test_augmentation=('test_augmentation', 'first'),
        dice_score_mean=('dice_score', 'mean'),
        dice_score_std=('dice_score', 'std'),
        iou_score_mean=('iou_score', 'mean'),
        iou_score_std=('iou_score', 'std'),
        sdice_score_mean=('sdice_score', 'mean'),
        sdice_score_std=('sdice_score', 'std')
    ).reset_index()
    scores['same_domain'] = scores['fold_train'] == scores['fold_test']
    scores['same_domain_base'] = scores['fold_train'] == scores['fold_test_base']
    scores['domain_val'] = scores['same_domain'].apply(lambda x: 'same' if x else 'other') + scores['is_validation'].apply(lambda x: ' validation' if x else '')
    scores['domain_val_base'] = scores['same_domain_base'].apply(lambda x: 'same' if x else 'other') + scores['is_validation'].apply(lambda x: ' validation' if x else '')
    scores['domain_val_testaug'] = scores['domain_val'] + scores['test_augmentation'].apply(lambda x: '' if x == 'None' else ' testaug')
    scores['trainer_short'] = scores['trainer'].apply(hlp.get_trainer_short)

    stats = stats.join(scores.set_index(index), on=index)
    stats['is_validation'] = stats['is_validation'].astype(bool)

    stats_weights = pd.concat([
        pd.read_csv(path) for path in glob.iglob('data/csv/weights/weights-*.csv', recursive=False)
    ]).reset_index(drop=True)
    stats_weights['name'] = stats_weights['name'].fillna('all')
    # take .conv weights if instancenorm was last layer
    stats_weights = stats_weights[~stats_weights['name'].str.endswith('.lrelu')]
    stats_weights['name'] = stats_weights['name'].str.replace('.conv', '.lrelu')
    stats_weights = stats_weights[['trainer', 'fold_train', 'epoch', 'name', 'norm_frob_weight', 'norm_spectral']]
    stats = stats.join(
        stats_weights.set_index(['trainer', 'fold_train', 'epoch', 'name']),
        on=['trainer', 'fold_train', 'epoch', 'layer']
    )
    for column in ['roughness', 'confidence', 'confidence_weighted']:
        stats[column + '_frob'] = stats[column] / stats['norm_frob_weight']
        stats[column + '_spectral'] = stats[column] / stats['norm_spectral']


    stats = stats[~stats['wd']]
    stats = stats[stats['DA'].isin(['none', 'full'])]
    stats = stats[stats['same_domain'] | (~stats['is_validation'])]


    stats['Data Aug.'] = stats['DA'].str.replace(r'^no([^n].+)$', r'no-\1', n=1, regex=True)
    stats['Domain'] = stats['domain_val_testaug'].str.replace('same validation', 'Validation').str.replace('same', 'Training').str.replace('other testaug', 'Other w/ test aug.').str.replace('other', 'Other w/o test aug.')
    stats['Base Domain'] = stats['domain_val_base'].str.replace('same validation', 'Validation').str.replace('same', 'Training').str.replace('other', 'Other') + stats['test_augmentation'].apply(lambda x: '' if x == 'None' else ' w/ test aug.')
    stats['Optimizer'] = stats['optimizer']
    stats['DSC'] = stats['dice_score_mean']
    stats['Surface DSC'] = stats['sdice_score_mean']
    stats['IoU'] = stats['iou_score_mean']
    stats['Epoch'] = stats['epoch']
    stats['Layer'] = stats['layer']
    stats['# Layer'] = stats['layer_pos']
    stats['Normalization'] = stats['bn'].apply(lambda x: 'Batch' if x else 'Instance')
    stats['Roughness'] = stats['roughness_frob']
    stats['Roughness no-frob'] = stats['roughness']
    stats['Confidence'] = stats['confidence_frob']
    stats['Confidence no-frob'] = stats['confidence']
    stats['Weighted-Confidence'] = stats['confidence_weighted_frob']
    stats['Weighted-Confidence no-frob'] = stats['confidence_weighted']

    # TODO
    stats = stats[~stats['Layer'].str.startswith('conv_blocks_context.0.blocks.0')]# roughness more problems than confidence
    stats = stats[~stats['Layer'].str.startswith('tu')] # roughness more problems than confidence
    stats = stats[~stats['Layer'].str.startswith('seg_outputs')]  # confidence more problems than roughness
    #

    layers_block_map = dict(map(reversed, enumerate(sorted(stats['layer_pos'].drop_duplicates().values))))
    stats['# Block'] = stats['layer_pos'].apply(lambda x: layers_block_map[x])
    print(stats[['# Block', 'Layer']].drop_duplicates().sort_values('# Block').to_string())

    stats = stats[stats['Surface DSC'] > 0.03]
    output_dir = output_dir + '-gt_0_03'

    columns_measurements = [
        'Roughness',
        'Roughness no-frob',
        'Confidence',
        'Confidence no-frob',
        'Weighted-Confidence',
        'Weighted-Confidence no-frob',
        #'roughness_spectral',
        #'confidence_spectral',
        #'confidence_weighted_spectral'
    ]
    stats_meaned_over_layer = stats.groupby(['trainer_short', 'fold_train', 'Epoch', 'fold_test', 'is_validation']).agg(**{
        'IoU': ('IoU', 'first'),
        'DSC': ('DSC', 'first'),
        'Surface DSC': ('Surface DSC', 'first'),
        'Optimizer': ('Optimizer', 'first'),
        'Normalization': ('Normalization', 'first'),
        'Data Aug.': ('Data Aug.', 'first'),
        'Domain': ('Domain', 'first'),
        'Base Domain': ('Base Domain', 'first'),
        **{
            column: (column, 'mean') for column in columns_measurements
        }
    }).reset_index()

    stats_meaned_over_layer['ro'] = stats_meaned_over_layer['Roughness no-frob']
    stats_meaned_over_layer['ro_f'] = stats_meaned_over_layer['Roughness']
    stats_meaned_over_layer['cw'] = stats_meaned_over_layer['Weighted-Confidence no-frob']
    stats_meaned_over_layer['cw_f'] = stats_meaned_over_layer['Weighted-Confidence']
    stats_meaned_over_layer['c'] = stats_meaned_over_layer['Confidence no-frob']
    stats_meaned_over_layer['c_f'] = stats_meaned_over_layer['Confidence']
    stats_meaned_over_layer['sd'] = stats_meaned_over_layer['Surface DSC']
    stats_meaned_over_layer['e'] = stats_meaned_over_layer['Epoch']
    columns_measurements_short = ['ro', 'ro_f', 'cw', 'cw_f', 'c', 'c_f']
    for groupby in [
        [],
        ['Optimizer',],
        ['Normalization'],
        ['Data Aug.'],
        ['Optimizer', 'Normalization'],
        ['Optimizer', 'Normalization', 'Data Aug.'],

        ['fold_train'],
        ['Optimizer', 'fold_train'],
        ['Normalization', 'fold_train'],
        ['Optimizer', 'Normalization', 'fold_train'],

        ['Domain'],
        ['Domain', 'Optimizer', 'Normalization'],
    ]:
        print(groupby)
        print(
            hlp.get_corr_stats(
                stats=stats_meaned_over_layer,
                groupby=groupby,
                columns=[*columns_measurements_short, 'sd', 'e'],
                column_combinations=[
                    *itertools.product(columns_measurements_short, [ 'sd' ]),
                    #*itertools.product(columns_measurements_short, [ 'e' ]),
                    ('cw', 'ro'),
                    ('cw_f', 'ro_f'),
                    ('sd', 'e')
                ]
            ).to_string()
        )


    print(stats)
    os.makedirs(output_dir, exist_ok=True)
    add_async_task, join_async_tasks = hlp.get_async_queue(num_threads=12)

    for measurement, score in itertools.product(columns_measurements, ['IoU', 'Surface DSC']):
        hlp.plot_scattered_and_layered(add_async_task, stats, stats_meaned_over_layer, measurement, output_dir, col='Normalization', row='Optimizer', hue='Base Domain', style='Data Aug.', score=score, yscale='log', x_line='# Block')
        hlp.plot_scattered_and_layered(add_async_task, stats, stats_meaned_over_layer, measurement, output_dir, col='Normalization', row='Optimizer', hue='fold_train', style='Data Aug.', score=score, yscale='log', x_line='# Block')
        hlp.plot_scattered_and_layered(add_async_task, stats, stats_meaned_over_layer, measurement, output_dir, col='Normalization', row='Optimizer', hue='Data Aug.', style=None, score=score, yscale='log', x_line='# Block')

        col='Normalization'
        row='Optimizer'
        hue='fold_train'
        style='Data Aug.'
        score=score

        stats_corr = hlp.get_corr_stats(
            stats,
            groupby=['# Block', col, row, hue, style],
            columns=[measurement, score],
            column_combinations=None
        ).reset_index()
        measurement_pval = '{}-{}_pval'.format(measurement, score)
        measurement_corr = '{}-{}_corr'.format(measurement, score)
        stats_corr.loc[stats_corr[measurement_pval] >= 0.01, measurement] = np.nan
        stats_corr[measurement] = stats_corr[measurement_corr]
        suffix = '{}-{}-{}-{}'.format(
            'single' if col is None else col,
            'single' if row is None else row,
            'single' if hue is None else hue,
            'single' if style is None else style,
        )
        add_async_task(
            hlp.relplot_and_save,
            outpath=os.path.join(
                output_dir,
                '{}-{}-corr-layered-{}.png'.format(measurement, score, suffix)
            ),
            data=stats_corr,
            kind='line',
            x='# Block',
            y=measurement,
            row=row,
            row_order=None if row is None else stats_corr[row].sort_values().unique(),
            col=col,
            col_order=None if col is None else stats_corr[col].sort_values().unique(),
            style=style if style in stats_corr else None,
            hue=hue,
            palette='cool',
            aspect=2,
            height=6,
        )
    
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

        'kmeans_src_inertia',
        #'kmeans_src_n_iter',
        'kmeans_dst_inertia',
        #'kmeans_dst_n_iter',

        'norm_frob_weight',
        'norm_spectral',
    ]
    #stats = stats[stats['Domain'] == 'Training']
    stats_train = stats.groupby(['trainer_short', 'fold_train', 'Epoch', '# Block', 'Domain']).agg(**{
        'IoU': ('IoU', 'mean'),
        'DSC': ('DSC', 'mean'),
        'Surface DSC': ('Surface DSC', 'mean'),
        'Optimizer': ('Optimizer', 'first'),
        'Normalization': ('Normalization', 'first'),
        'Data Aug.': ('Data Aug.', 'first'),
        **{
            column: (column, 'mean') for column in columns_measurements_train
        }
    }).reset_index()
    stats_train_meaned_over_layer = stats_train.groupby(['trainer_short', 'fold_train', 'Epoch', 'Domain']).agg(**{
        'IoU': ('IoU', 'mean'),
        'DSC': ('DSC', 'mean'),
        'Surface DSC': ('Surface DSC', 'mean'),
        'Optimizer': ('Optimizer', 'first'),
        'Normalization': ('Normalization', 'first'),
        'Data Aug.': ('Data Aug.', 'first'),
        **{
            column: (column, 'mean') for column in columns_measurements_train
        }
    }).reset_index()
    print(stats_train)
    print(stats_train_meaned_over_layer)

    for measurement, score in itertools.product(columns_measurements_train, ['IoU', 'Surface DSC']):
        hlp.plot_scattered_and_layered(add_async_task, stats_train, stats_train_meaned_over_layer, measurement, output_dir, col='Normalization', row='Optimizer', hue='fold_train', style='Data Aug.', score=score, yscale='log', x_line='# Block')
        hlp.plot_scattered_and_layered(add_async_task, stats_train, stats_train_meaned_over_layer, measurement, output_dir, col='Optimizer', hue='Normalization', style='Data Aug.', score=score, x_line='# Block')#, yscale='log')

    join_async_tasks()



task = 'Task601_cc359_all_training'
trainers = [
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120_noDA__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120_nogamma__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120_nomirror__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120_norotation__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120_noscaling__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120_noDA__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120_nogamma__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120_nomirror__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120_norotation__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120_noscaling__nnUNetPlansv2.1',

    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_bn_SGD_ep120__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_bn_SGD_ep120_noDA__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_bn_ep120__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_bn_ep120_noDA__nnUNetPlansv2.1',
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


# calc_roughness_and_confidence_grouped(task, trainers, folds, epochs)

plot_roughness_and_confidence_grouped(
    'data/csv/activations-roughness-and-confidence/activations-roughness-and-confidence-grouped-relu-and-residual-pca10c-gt-s6-b48-testaug*.csv',
    'data/fig/activations-roughness-and-confidence/pca10c-gt-s6-b48-testaug'
)