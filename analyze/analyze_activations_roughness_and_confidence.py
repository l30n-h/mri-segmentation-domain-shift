import numpy as np
import torch
import pandas as pd
import matplotlib as mpl
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



def calc_roughness_and_confidence_grouped(task, trainers, folds, epochs):
    def get_activations_dicts_merged(activations_paths):
        activations_dict_concat = dict()
        for path in activations_paths:
            activations_dict = torch.load(
                path,
                map_location=torch.device('cpu')
            )
            for key, value in activations_dict.items():
                activations_dict_concat.setdefault(key, []).extend(
                    value
                )
        
        return dict(map(
            lambda item: (
                item[0],
                torch.stack(item[1])
            ),
            activations_dict_concat.items()
        ))
    def get_roughness_and_confidence_scores(scores):
        # assert len(fold_train.unique) == 1
        fold_train = scores['fold_train'].unique()[0]
        scores_dict = {
            (fold_test, is_validation): scores[
                (scores['fold_test'] == fold_test) & (scores['is_validation'] == is_validation)
            ] for fold_test, is_validation in itertools.product(
                scores['fold_test'].unique(),
                scores['is_validation'].unique()
            )
        }
        activations_dict_dict = {
            key: get_activations_dicts_merged(
                scores['activations_path'].tolist()
            ) for key, scores in scores_dict.items()
        }
        layers = ['input'] + list(filter(
            lambda x: '.instnorm' in x or 'tu.' in x or 'seg_outputs.' in x,
            hlp.get_layers_ordered()
        ))
        for layer_pre, layer_cur in zip(layers, layers[1:]):
            activations_dict_train = activations_dict_dict[(fold_train, 0)]
            data_src = activations_dict_train.get(layer_pre, None)
            data_dst = activations_dict_train.get(layer_cur, None)
            if data_src is None or data_dst is None:
                continue
            layer_config = hlp.get_layer_config(layer_cur)
            if 'kernel_size' not in layer_config:
                continue
            data_gt = activations_dict_train['gt']

            #data_src_gt = torch.zeros_like(data_src[:,:1,:,:])
            #data_dst_gt = torch.zeros_like(data_dst[:,:1,:,:])
            data_src_gt = torch.nn.functional.interpolate(data_gt, data_src.shape[2:], mode='bilinear').ceil()
            data_dst_gt = torch.nn.functional.interpolate(data_gt, data_dst.shape[2:], mode='bilinear').ceil()
            
            kernel_size=layer_config['kernel_size']
            stride=layer_config['stride']
            MAX_SAMPLES=6*4*255*195 // 40 # paper code batchsize 50 * volsize [64,64,16] / 80000

            print(layer_cur)

            if '.instnorm' in layer_pre:
                data_src = torch.where(data_src > 0, data_src, data_src * 0.01)
            if layer_pre.startswith('tu.'):
                d = int(layer_pre.replace('tu.', ''))
                conv_blocks_context = activations_dict_train['conv_blocks_context.{}.blocks.1.instnorm'.format(3-d)]
                conv_blocks_context = torch.where(conv_blocks_context > 0, conv_blocks_context, conv_blocks_context * 0.01)
                data_src = torch.cat((data_src, conv_blocks_context), axis=1)
            

            data_src = data_src.movedim(1, -1).numpy()
            data_src_gt = data_src_gt.movedim(1, -1).numpy()
            data_dst = data_dst.movedim(1, -1).numpy()
            data_dst_gt = data_dst_gt.movedim(1, -1).numpy()

            pt_src, pt_src_gt, pt_dst_pred, pt_dst_gt = align_and_subsample(
                data_src,
                data_src_gt,
                data_dst,
                data_dst_gt,
                MAX_SAMPLES=MAX_SAMPLES,
                kernel_size=kernel_size,
                stride=stride
            )

            PCA_components = 0.8 #10
            kmeans_components = 5

            pca_src, kmeans_src, mixtures_src = learn_manifold(
                pt_src,
                min(PCA_components, pt_src.shape[1]),
                kmeans_components
            )
            pca_dst, kmeans_dst, mixtures_dst = learn_manifold(
                pt_dst_pred,
                min(PCA_components, pt_dst_pred.shape[1]),
                kmeans_components
            )

            pca_src_scores = { 'pca_src_{}'.format(key): value for key, value in get_pca_scores(pca_src).items() }
            pca_dst_scores = { 'pca_dst_{}'.format(key): value for key, value in get_pca_scores(pca_dst).items() }
            train_metrics = {
                **pca_src_scores,
                **pca_dst_scores,
                'kmeans_src_inertia': kmeans_src.inertia_,
                'kmeans_src_n_iter': kmeans_src.n_iter_,
                'kmeans_dst_inertia': kmeans_dst.inertia_,
                'kmeans_dst_n_iter': kmeans_dst.n_iter_
            }

            for (fold_test, is_validation), activations_dict in activations_dict_dict.items():
                data_src = activations_dict[layer_pre]
                data_dst = activations_dict[layer_cur]
                data_gt = activations_dict['gt']

                #data_src_gt = torch.zeros_like(data_src[:,:1,:,:])
                #data_dst_gt = torch.zeros_like(data_dst[:,:1,:,:])
                data_src_gt = torch.nn.functional.interpolate(data_gt, data_src.shape[2:], mode='bilinear').ceil()
                data_dst_gt = torch.nn.functional.interpolate(data_gt, data_dst.shape[2:], mode='bilinear').ceil()

                if '.instnorm' in layer_pre:
                    data_src = torch.where(data_src > 0, data_src, data_src * 0.01)
                if layer_pre.startswith('tu.'):
                    d = int(layer_pre.replace('tu.', ''))
                    conv_blocks_context = activations_dict['conv_blocks_context.{}.blocks.1.instnorm'.format(3-d)]
                    conv_blocks_context = torch.where(conv_blocks_context > 0, conv_blocks_context, conv_blocks_context * 0.01)
                    data_src = torch.cat((data_src, conv_blocks_context), axis=1)
                
                
                data_src = data_src.movedim(1, -1).numpy()
                data_src_gt = data_src_gt.movedim(1, -1).numpy()
                data_dst = data_dst.movedim(1, -1).numpy()
                data_dst_gt = data_dst_gt.movedim(1, -1).numpy()

                test_src, test_src_gt, test_dst_pred, test_dst_gt = align_and_subsample(
                    data_src,
                    data_src_gt,
                    data_dst,
                    data_dst_gt,
                    MAX_SAMPLES=MAX_SAMPLES*2,
                    kernel_size=kernel_size,
                    stride=stride
                )
                pt_src_transformed, _, _ = project_manifold(test_src, pca_src, kmeans_src, mixtures_src)
                pt_dst_transformed, dst_clustidx, _ = project_manifold(test_dst_pred, pca_dst, kmeans_dst, mixtures_dst)

                ## COMPUTE CLUSTERING USING PT_PRED --> This is the roughness metric
                db_roughness = sklearn.metrics.davies_bouldin_score(
                    pt_src_transformed,
                    dst_clustidx
                )

                ## COMPUTE CONFIDENCE -->
                scores, score_weight = compute_confidence(mixtures_src, pt_src_transformed, pt_dst_transformed)
                wscores = scores * score_weight
                conf = scores.mean()
                wconf = wscores.mean()
                scores_sub = scores_dict[(fold_test, is_validation)]
                yield {
                    'fold_test': fold_test,
                    'is_validation': is_validation,
                    'name': layer_cur,
                    'roughness': db_roughness,
                    'confidence': conf,
                    'confidence_weighted': wconf,
                    'iou_score_mean': scores_sub['iou_score'].mean(),
                    'iou_score_std': scores_sub['iou_score'].std(),
                    'dice_score_mean': scores_sub['dice_score'].mean(),
                    'dice_score_std': scores_sub['dice_score'].std(),
                    'sdice_score_mean': scores_sub['sdice_score'].mean(),
                    'sdice_score_std': scores_sub['sdice_score'].std(),
                    **train_metrics
                }

    output_dir = 'data/csv/activations-roughness-and-confidence'
    os.makedirs(output_dir, exist_ok=True)

    for trainer, fold_train, epoch in itertools.product(trainers, folds, epochs):
        print(task, trainer, fold_train, epoch)

        id_path_map = dict(map(
            lambda p: (hlp.get_id_from_filename(os.path.basename(p)), p),
            glob.iglob(os.path.join(
                hlp.get_testdata_dir(task, trainer, fold_train, epoch),
                'activations-small-fullmap',
                '*_activations.pkl'
            ))
        ))
        
        scores = hlp.get_scores(task, trainer, fold_train, epoch)

        scores = scores[scores['id'].isin(set(id_path_map.keys()))]
        scores['activations_path'] = scores['id'].apply(lambda x: id_path_map.get(x))
        #scores = scores.sort_values('id')
        
        stats = pd.DataFrame(
            get_roughness_and_confidence_scores(scores)
        )
        stats['trainer'] = trainer
        stats['fold_train'] = fold_train
        stats['epoch'] = epoch

        stats.to_csv(
            os.path.join(
                output_dir,
                'activations-roughness-and-confidence-grouped-relu-and-residual{}-{}-{}-{}.csv'.format(
                    '-pca80p-gt',#'',
                    trainer,
                    fold_train,
                    epoch
                )
            )
        )


def plot_roughness_and_confidence_grouped():
    stats = pd.concat([
        pd.read_csv(path) for path in glob.iglob('data/csv/activations-roughness-and-confidence/activations-roughness-and-confidence-grouped-relu-and-residual-pca80p-gt-*.csv', recursive=False)
    ])
    stats.rename(
        columns={
            'name':'layer'
        },
        inplace = True
    )

    stats['trainer_short'] = stats['trainer'].apply(hlp.get_trainer_short)
    layers_position_map = hlp.get_layers_position_map()
    stats['layer_pos'] = stats['layer'].apply(lambda d: layers_position_map.get(d))

    stats_weights = pd.concat([
        pd.read_csv(path) for path in glob.iglob('data/csv/weights/weights-*.csv', recursive=False)
    ])
    stats_weights['name'] = stats_weights['name'].str.replace('.weight', '')
    stats_weights = stats_weights[['trainer', 'fold_train', 'epoch', 'name', 'norm_frob', 'norm_spectral']]

    # TODO
    stats = stats[~stats['layer'].str.startswith('seg_outputs')]
    stats = stats[~stats['layer'].str.startswith('tu')]
    stats['layer'] = stats['layer'].str.replace('.instnorm', '.conv')
    # 
    stats = stats.join(stats_weights.set_index(['trainer', 'fold_train', 'epoch', 'name']), on=['trainer', 'fold_train', 'epoch', 'layer'])

    columns = [
        'roughness',
        'confidence',
        'confidence_weighted',
    ]
    for column in columns:
        stats[column + '_frob'] = stats[column] / stats['norm_frob']
        stats[column + '_spectral'] = stats[column] / stats['norm_spectral']
    columns = [
        'roughness',
        'confidence',
        'confidence_weighted',
        'roughness_frob',
        'confidence_frob',
        'confidence_weighted_frob',
        'roughness_spectral',
        'confidence_spectral',
        'confidence_weighted_spectral',
        *list(filter(
            lambda c: c.startswith('pca_') or c.startswith('kmeans_'),
            stats.columns.values.tolist()
        ))
    ]


    stats_meaned = stats.groupby(['trainer_short', 'fold_train', 'epoch', 'fold_test', 'is_validation']).agg(
        iou_score_mean=('iou_score_mean', 'first'),
        dice_score_mean=('dice_score_mean', 'first'),
        sdice_score_mean=('sdice_score_mean', 'first'),
        roughness_mean=('roughness', 'mean'),
        confidence_mean=('confidence', 'mean'),
        confidence_weighted_mean=('confidence_weighted', 'mean'),
        roughness_frob_mean=('roughness_frob', 'mean'),
        confidence_frob_mean=('confidence_frob', 'mean'),
        confidence_weighted_frob_mean=('confidence_weighted_frob', 'mean'),
        roughness_spectral_mean=('roughness_spectral', 'mean'),
        confidence_spectral_mean=('confidence_spectral', 'mean'),
        confidence_weighted_spectral_mean=('confidence_weighted_spectral', 'mean'),
        pca_src_variance_ratio_cumsum_index_max_mean=('pca_src_variance_ratio_cumsum_index_max', 'mean'),
        pca_dst_variance_ratio_cumsum_index_max_mean=('pca_dst_variance_ratio_cumsum_index_max', 'mean'),
    ).reset_index()
    stats_meaned['subplot'] = stats_meaned['trainer_short'] + '_' + stats_meaned['fold_train']
    stats_meaned['psize'] = stats_meaned['epoch'] / 10 + (stats_meaned['fold_train'] == stats_meaned['fold_test']).astype(int) * 5


    stats_layered = stats.copy()
    stats_layered['subplot'] = stats_layered['trainer_short'] + '_' + stats_layered['fold_train']
    stats_layered['line'] = stats_layered['epoch'].map(str) + '_' + stats_layered['fold_test'] + '_' + stats_layered['is_validation'].map(str)
    stats_layered_colors = stats_layered.groupby(['subplot', 'line']).agg(
        iou_score_mean=('iou_score_mean', 'first'),
        dice_score_mean=('dice_score_mean', 'first'),
        sdice_score_mean=('sdice_score_mean', 'first'),
    )
    layered_line_columns = stats_layered['line'].unique()
    stats_layered = stats_layered.pivot(
        index=['subplot', 'layer_pos'],
        columns=['line']
    ).reset_index(col_level=1)

    stats_train = stats.copy()
    stats_train = stats_train[(stats_train['fold_train'] == stats_train['fold_test']) & (~stats_train['is_validation'])]
    stats_train['subplot'] = stats_train['trainer_short'] + '_' + stats_train['fold_train']
    stats_train = stats_train.pivot(
        index=['subplot', 'layer_pos'],
        columns=['epoch']
    ).reset_index(col_level=1)
    stats_train.columns = ['_'.join(str(s).strip() for s in col if s) for col in stats_train.columns]
    
    print(stats_train.columns.values)

    output_dir = 'data/fig/activations-roughness-and-confidence'
    os.makedirs(output_dir, exist_ok=True)

    for column in [
        'roughness',
        'confidence',
        'confidence_weighted',
        'roughness_frob',
        'confidence_frob',
        'confidence_weighted_frob',
        'roughness_spectral',
        'confidence_spectral',
        'confidence_weighted_spectral',
        'pca_src_variance_ratio_cumsum_index_max',
        'pca_dst_variance_ratio_cumsum_index_max'
    ]:
        print(column)
        fig, axes = hlp.create_plot(
            stats_meaned,
            column_x='{}_mean'.format(column),
            column_y='iou_score_mean',
            #column_y='sdice_score_mean',
            column_subplots='subplot',
            column_size='psize',
            column_color='iou_score_mean',
            #column_color='sdice_score_mean',
            ncols=2,
            figsize=(42, 24),
            lim_same_x=True,
            lim_same_y=True,
            lim_same_c=True,
            colormap='cool',
            fig_num='activations-roughness-and-confidence',
            fig_clear=True
        )
        fig.savefig(
            os.path.join(
                output_dir,
                'activations-roughness-and-confidence-grouped-relu-and-residual-meaned-{}.png'.format(column)
            ),
            bbox_inches='tight',
            pad_inches=0
        )
        plt.close(fig)

        stats_layered_filtered = stats_layered[['', column]]
        stats_layered_filtered.columns = stats_layered_filtered.columns.droplevel(0)
        fig, axes = hlp.create_plot(
            stats_layered_filtered,
            column_x='layer_pos',
            column_y=layered_line_columns,
            kind='line',
            column_subplots='subplot',
            column_color='iou_score_mean',
            #column_color='sdice_score_mean',
            colors=stats_layered_colors,
            ncols=2,
            figsize=(42, 24),
            lim_same_x=True,
            lim_same_y=True,
            lim_same_c=True,
            colormap='cool',
            legend=False,
            fig_num='activations-roughness-and-confidence',
            fig_clear=True
        )
        fig.savefig(
            os.path.join(
                output_dir,
                'activations-roughness-and-confidence-grouped-relu-and-residual-layered-{}.png'.format(column)
            ),
            bbox_inches='tight',
            pad_inches=0
        )
        plt.close(fig)
    
    for column in ['variance_ratio_cumsum_m', 'noise_variance', 'inertia', 'n_iter', 'norm', 'index_max']:
        fig, axes = hlp.create_plot(
            stats_train,
            column_x='layer_pos',
            column_y=list(filter(
                lambda c: column in c,
                stats_train.columns.values.tolist()
            )),
            kind='line',
            column_subplots='subplot',
            ncols=2,
            figsize=(42, 24),
            lim_same_x=True,
            lim_same_y=True,
            lim_same_c=True,
            colormap='cool',
            legend=True,
            fig_num='activations-roughness-and-confidence',
            fig_clear=True
        )
        fig.savefig(
            os.path.join(
                output_dir,
                'activations-roughness-and-confidence-grouped-relu-and-residual-layered-{}.png'.format(column)
            ),
            bbox_inches='tight',
            pad_inches=0
        )
        plt.close(fig)



task = 'Task601_cc359_all_training'
trainers = [
    # 'nnUNetTrainerV2_MA_noscheduler_depth5_SGD_ep120__nnUNetPlansv2.1',
    # 'nnUNetTrainerV2_MA_noscheduler_depth5_SGD_ep120_noDA__nnUNetPlansv2.1',
    # 'nnUNetTrainerV2_MA_noscheduler_depth5_ep120__nnUNetPlansv2.1',
    # 'nnUNetTrainerV2_MA_noscheduler_depth5_ep120_noDA__nnUNetPlansv2.1',

    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120_noDA__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120_noDA__nnUNetPlansv2.1',
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
#TODO paper uses mask in test data to sample well distributed samples. Still label free??
#TODO which layers to consider?
#     model 600 referenced in paper seems to define a layer as conv-norm-relu...
#     here only .conv .tu .seg_outputs
#TODO number of pca components needed for 80% variance seems to correlate with epochs and optimizer
#     at least its lower for adam than for sgd
#     its lower for 120 epochs for SGD and higher for 120 epochs for Adam
#     its slightly lower for both if DA is used (execept sgd ge3)

#TODO check for seg_output shift