import numpy as np
import torch
import pandas as pd
import itertools
import os
import sys
import glob

import helpers as hlp

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)  
sys.path.append(os.path.join(parent, 'lipEstimation'))
import lipschitz_approximations as lipapp
import lipschitz_utils as liputils

fold_id_mapping = hlp.get_fold_id_mapping()

def get_weights_noise_estimate(weights):
    a = weights.reshape(
        weights.shape[0] * weights.shape[1],
        1,
        *weights.shape[2:4]
    )
    noise_means = hlp.get_noise_estimates(a)
    return noise_means.reshape(*weights.shape[0:2])

def get_toeplitz(kernel_flat):
    #TODO input length and padding not considered
    dim = len(kernel_flat.shape) - 1
    k_len = kernel_flat.shape[dim]
    shifts = torch.arange(k_len, device=kernel_flat.device)
    idx1 = shifts.view((k_len, 1)).repeat((1, k_len))
    idx2 = (idx1 + shifts) % k_len
    assert dim <= 1
    if dim == 1:
        return torch.gather(
            kernel_flat.unsqueeze(1).repeat(1, k_len, 1),
            dim + 1,
            idx2.unsqueeze(0).repeat(kernel_flat.shape[0], 1, 1)
        )
    return torch.gather(
        kernel_flat.repeat(k_len, 1),
        1,
        idx2
    )


def generate_stats_weights(parameters):
    def get_stats(features):
        # for conv [out_ch, in_ch, k_h, k_w]
        if features is None or len(features.shape) == 0:
            return {}
        
        if len(features.shape) < 4:
            features = features[(...,)+(None,)*(4-len(features.shape))]

        features = features.to(device='cuda').to(dtype=torch.float32)

        features_noise = get_weights_noise_estimate(features)


        features_flat = features.flatten(1)
        features_flat_normed, features_flat_norm = hlp.get_normed(features_flat, dim=1)

        a = features_flat_normed
        # Idea was to append norm of feature vector to measure similarity and considering the length
        # very dependent on the length so tried to norm with feature vector size...
        # features_flat_norm = (features_flat_norm/features_flat_normed.shape[1])
        # a = torch.cat((features_flat_normed, features_flat_norm), dim=1)
        similarity_matrix = hlp.get_cosine_similarity(a, a)
        #similarity_matrix = hlp.get_l2_similarity(a, a)

        diagonal = similarity_matrix.diagonal()
        diagonal_zero = (diagonal < 0.001).count_nonzero().float()
        triu_ids = torch.triu_indices(similarity_matrix.shape[0], similarity_matrix.shape[1], 1, device=similarity_matrix.device)
        coocurence = similarity_matrix[triu_ids[0], triu_ids[1]]


        features_flat2 = features.flatten(2)
        features_flat2_normed, features_flat2_norm = hlp.get_normed(features_flat2, dim=1)

        similarity_matrices2 = torch.stack([ 
            hlp.get_cosine_similarity(f, f) for f in features_flat2_normed
        ])

        diagonals2 = similarity_matrices2.diagonal(dim1=1, dim2=2)
        diagonals2_zero = (diagonals2 < 0.001).count_nonzero(dim=0).float()
        triu_ids = torch.triu_indices(similarity_matrices2.shape[1], similarity_matrices2.shape[2], 1, device=similarity_matrices2.device)
        coocurences2 = similarity_matrices2[:,triu_ids[0], triu_ids[1]]

        # TODO tried to get condition of matrix to check numerical stability
        # very slow and almost unbounded...
        # toeplitz = get_toeplitz(features_flat)
        # toeplitz_norm = toeplitz.flatten(1).norm(dim=1, p='fro')
        # toeplitz_det = toeplitz.det().abs()
        # n = toeplitz.shape[0]
        # #x = torch.sqrt(1-(n/(toeplitz_norm*toeplitz_norm)).pow(n) * toeplitz_det * toeplitz_det)
        # #cond_est = torch.sqrt((1+x)/(1-x))
        # cond_est = 2 / toeplitz_det * (toeplitz_norm / torch.tensor(n).sqrt()).pow(n)
        # #cond = torch.linalg.cond(toeplitz, p='fro')


        features_base = {
            'features': features,
            'features_noise': features_noise,
            'features_zero': diagonal_zero[None],
            'features_norm': features_flat_norm,
            'similarity_di_abs': diagonal.abs(),
            'similarity_co_abs': coocurence.abs(),
            'features2_zero': diagonals2_zero,
            'features2_norm': features_flat2_norm,
            'similarity2_di_abs': diagonals2.abs(),
            'similarity2_co_abs': coocurences2.abs(),
        }

        features_moments = {
            k: v for name, value in features_base.items() for k, v in hlp.get_moments_recursive(name, value)
        }
        return {
            'num_features': features_flat_norm.shape[0],
            **dict(map(lambda d: (d[0], d[1].item()), features_moments.items()))
        }

    for name in set(map(lambda x: x.rsplit('.', 1)[0], parameters.keys())):
        features_weight = parameters.get('{}.weight'.format(name), None)
        features_bias = parameters.get('{}.bias'.format(name), None)
        stats_weight = get_stats(features_weight)
        stats_bias = get_stats(features_bias)
        out = {
            'name': name,
            **dict(map(lambda d: ('{}_weight'.format(d[0]), d[1]), stats_weight.items())),
            **dict(map(lambda d: ('{}_bias'.format(d[0]), d[1]), stats_bias.items())),
            
            # 'toe_norm_mean': toeplitz_norm.mean().item(),
            # 'toe_norm_std': toeplitz_norm.std().item(),
            # 'toe_det_mean': toeplitz_det.mean().item(),
            # 'toe_det_std': toeplitz_det.std().item(),
            # 'cond_est_mean': cond_est.mean().item(),
            # 'cond_est_std': cond_est.std().item(),
            
        }
        yield out

def calc_weight_norms(network):
    liputils.compute_module_input_sizes(network, (1, 1, 256, 192))
    lipapp.lipschitz_spectral_ub(network)
    lipapp.lipschitz_frobenius_ub(network)

    stats_norms = pd.DataFrame()
    for name, module in network.named_modules():
        if hasattr(module, 'frob_norm'):
            for k, v in module.frob_norm.items():
                stats_norms.loc[name, 'norm_frob_' + k] = v.item()
        if hasattr(module, 'spectral_norm'):
            stats_norms.loc[name, 'norm_spectral'] = module.spectral_norm.item()
    return stats_norms

def calc_weight_stats(trainers, folds, epochs):
    output_dir = 'data/csv/weights'
    os.makedirs(output_dir, exist_ok=True)

    for trainer, fold, epoch in itertools.product(trainers, folds, epochs):
        print(trainer, fold, epoch)

        tmodel = hlp.load_model(trainer, fold, epoch)
        tmodel.network.do_ds = False
        for p in tmodel.network.parameters():
            p.requires_grad = False
        for name, module in tmodel.network.named_modules():
            module.full_name = name
            if hasattr(module, 'inplace'):
                module.inplace = False
        
        parameters = dict(tmodel.network.named_parameters())
        
        
        stats_norms = calc_weight_norms(tmodel.network)

        stats = pd.DataFrame(generate_stats_weights(
            parameters
        ))
        stats = stats.join(
            stats_norms,
            how='outer',
            on='name'
        )

        stats['trainer'] = trainer
        stats['fold_train'] = fold
        stats['epoch'] = epoch
        print(stats)
        stats.to_csv(
            os.path.join(
                output_dir,
                'weights-{}-{}-{}.csv'.format(trainer, fold, epoch)
            )
        )

def plot_weight_stats():
    stats = pd.concat([
        pd.read_csv(path) for path in glob.iglob('data/csv/weights/weights-*.csv', recursive=False)
    ]).reset_index(drop=True)

    index = ['trainer', 'fold_train', 'epoch']
    scores = pd.concat([
        hlp.get_scores(
            'Task601_cc359_all_training',
            *ids
        ) for ids in stats[index].drop_duplicates().values.tolist()
    ]).reset_index(drop=True)
    scores = scores.groupby(index).agg(
        optimizer=('optimizer', 'first'),
        wd=('wd', 'first'),
        DA=('DA', 'first'),
        bn=('bn', 'first'),
        dice_score_mean=('dice_score', 'mean'),
        dice_score_std=('dice_score', 'std'),
        iou_score_mean=('iou_score', 'mean'),
        iou_score_std=('iou_score', 'std'),
        sdice_score_mean=('sdice_score', 'mean'),
        sdice_score_std=('sdice_score', 'std')
    ).reset_index()

    stats = stats.join(scores.set_index(index), on=index)

    stats.rename(columns={ 'name': 'layer' }, inplace=True)
    layers_position_map = hlp.get_layers_position_map()
    stats['layer_pos'] = stats['layer'].apply(lambda d: layers_position_map.get(d))

    stats['trainer_short'] = stats['trainer'].apply(hlp.get_trainer_short)    
    stats['wd_bn'] = stats['wd'].apply(lambda x: 'wd=' + str(x)) + ' ' + stats['bn'].apply(lambda x: 'bn=' + str(x))

    stats['stable_rank'] = (stats['norm_frob_weight'] ** 2) / ((stats['norm_spectral'] ** 2) + 0.00001)
    
    stats_base = stats
    print(stats)

    output_dir = 'data/fig/weights'
    os.makedirs(output_dir, exist_ok=True)

    add_async_task, join_async_tasks = hlp.get_async_queue(num_threads=12)


    columns = [
        'stable_rank',
        *list(filter(lambda c: any(s in c for s in [
            'norm_',
            'features',
            'similarity',
        ]), stats.columns.values.tolist()))
    ]
    print(columns)

    stats = stats_base.copy()
    print(stats)
    stats = stats[stats['layer_pos'] >= 0]
    stats = stats[~stats['layer'].str.endswith('.lrelu')]
    stats = stats[~stats['layer'].str.endswith('.instnorm')]
    for column in columns:
        add_async_task(
            hlp.relplot_and_save,
            outpath=os.path.join(
                output_dir,
                'weights-layer_pos-{}-dice_score-trainer_short.png'.format(column)
            ),
            data=stats,
            kind='line',
            x='layer_pos',
            y=column,
            col='fold_train',
            col_order=stats['fold_train'].sort_values().unique(),
            row='trainer_short',
            row_order=stats['trainer_short'].sort_values().unique(),
            hue='dice_score_mean',
            palette='cool',
            #style='fold_train',
            size='epoch',
            height=6,
            aspect=2,
            estimator=None,
            ci=None,
            facet_kws=dict(
                sharey='row'
            )
        )


    # stats = stats_base.copy()
    # stats['layer'] = stats['layer'].fillna('all')
    # stats = stats[~stats['layer'].str.endswith('.lrelu')]
    # for layer in stats['layer'].drop_duplicates():
    #     stats_l = stats[stats['layer'] == layer]
    #     add_async_task(
    #             hlp.relplot_and_save,
    #             outpath=os.path.join(
    #                 output_dir,
    #                 'weights-norm_spectral-dice_score-trainer_short-{}.png'.format(layer)
    #             ),
    #             data=stats_l,
    #             kind='scatter',
    #             x='norm_spectral',
    #             y='dice_score_mean',
    #             col='optimizer',
    #             col_order=stats['optimizer'].sort_values().unique(),
    #             row='wd_bn',
    #             row_order=stats['wd_bn'].sort_values().unique(),
    #             hue='fold_train',
    #             #palette='cool',
    #             style='DA',
    #             size='epoch',
    #             height=6,
    #             aspect=2,
    #             estimator=None,
    #             ci=None,
    #         )
    
    join_async_tasks()


def plot_weights(trainers, folds, epochs):
    layers_position_map = hlp.get_layers_position_map()
    for trainer, fold_train, epoch in itertools.product(trainers, folds, epochs):
        print(trainer, fold_train, epoch)

        tmodel = hlp.load_model(trainer, fold_train, epoch)
        parameters_dict = dict(tmodel.network.named_parameters())

        for name, parameters in parameters_dict.items():
            if len(parameters.shape) < 4:
                print(name)
                continue
            layer_id = layers_position_map.get(name.replace('.weight', ''))
            merged = torch.nn.functional.pad(
                parameters,
                (2,2,2,2)
            ).swapaxes(
                0,
                1
            ).flatten(
                start_dim=1,
                end_dim=2
            ).swapaxes(
                0,
                1
            ).flatten(
                start_dim=1,
                end_dim=2
            )
            mval = merged.abs().max()
            plt.imsave(
                'data/fig/weights_pure-{}-{}-{}-{}.jpg'.format(
                    trainer, fold_train, epoch, layer_id
                ),
                merged,
                vmin=-mval,
                vmax=mval
            )

            # fig, axes = plt.subplots(figsize=(32, 24), nrows=1, ncols=1)
            # axes.imshow(merged, cmap='viridis', interpolation='none')
            # #axes.colorbar()

            # fig.savefig('data/fig/weights_pure-{}-{}-{}-{}.jpg'.format(
            #     trainer, fold_train, epoch, layer_id
            # ))
            # plt.close(fig)
        


trainers = [
    'nnUNetTrainerV2_MA_noscheduler_depth5_ep120__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_ep120_noDA__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_SGD_ep120__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_SGD_ep120_noDA__nnUNetPlansv2.1',
    
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120_noDA__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120_noDA__nnUNetPlansv2.1',

    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_bn_ep120__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_bn_ep120_noDA__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_bn_SGD_ep120__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_bn_SGD_ep120_noDA__nnUNetPlansv2.1',
]
folds = [
    'siemens15',
    'siemens3',
    'ge15',
    'ge3',
    'philips15',
    'philips3'
]
epochs = [10, 20, 30, 40, 80, 120]

#calc_weight_stats(trainers, folds, epochs)

plot_weight_stats()

#plot_weights(trainers, folds, epochs)