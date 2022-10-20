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
    scores['same_domain'] = scores['fold_train'] == scores['fold_test']
    scores['domain_val'] = scores['same_domain'].apply(lambda x: 'same' if x else 'other') + ' ' + scores['is_validation'].apply(lambda x: 'validation' if x else '')
    scores = scores.groupby([*index, 'domain_val']).agg(
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
    stats = scores.join(stats.set_index(index), on=index)

    stats.rename(columns={ 'name': 'layer' }, inplace=True)
    layers_position_map = hlp.get_layers_position_map()
    stats['layer_pos'] = stats['layer'].apply(lambda d: layers_position_map.get(d))
    stats['wd_bn'] = stats['wd'].apply(lambda x: 'wd=' + str(x)) + ' ' + stats['bn'].apply(lambda x: 'bn=' + str(x))
    stats['optimizer_wd_bn'] = stats['optimizer'] + ' ' + stats['wd_bn']
    stats['trainer_short'] = stats['trainer'].apply(hlp.get_trainer_short)
    
    stats['stable_rank'] = (stats['norm_frob_weight'] ** 2) / ((stats['norm_spectral'] ** 2) + 0.00001)
    
    stats_base = stats
    print(stats)

    output_dir = 'data/fig/weights'
    os.makedirs(output_dir, exist_ok=True)

    add_async_task, join_async_tasks = hlp.get_async_queue(num_threads=12)

    #stats = stats[stats['domain_val'] == 'same validation']
    stats = stats[stats['domain_val'] == 'other ']
    stats = stats[~stats['wd']]
    stats = stats[stats['layer_pos'] >= 0]
    stats = stats[~stats['layer'].str.endswith('.lrelu')]
    stats = stats[~stats['layer'].str.endswith('.instnorm')]
    #stats = stats[(stats['bn']) | (~stats['layer'].str.endswith('.instnorm'))]
    print(stats)

    columns_measurements = [
        'stable_rank',
        'norm_frob_weight',
        'norm_spectral'
        # *list(filter(lambda c: any(s in c for s in [
        #     'norm_',
        #     # 'features',
        #     # 'similarity',
        # ]), stats.columns.values.tolist()))
    ]
    print(columns_measurements)

    stats_meaned_over_layer = stats.groupby(['trainer_short', 'fold_train', 'epoch']).agg(
        iou_score_mean=('iou_score_mean', 'first'),
        dice_score_mean=('dice_score_mean', 'first'),
        sdice_score_mean=('sdice_score_mean', 'first'),
        optimizer=('optimizer', 'first'),
        optimizer_wd_bn=('optimizer_wd_bn', 'first'),
        **{
            '{}_mean'.format(column): (column, 'mean') for column in columns_measurements
        }
    ).reset_index()

    for measurement, score in itertools.product(columns_measurements, ['dice_score_mean', 'sdice_score_mean']):
        hlp.plot_scattered_and_layered(add_async_task, stats, stats_meaned_over_layer, measurement, output_dir, row=None, score=score, yscale='log')
        hlp.plot_scattered_and_layered(add_async_task, stats, stats_meaned_over_layer, measurement, output_dir, row='trainer_short', score=score, yscale='log')
        hlp.plot_scattered_and_layered(add_async_task, stats, stats_meaned_over_layer, measurement, output_dir, row='optimizer', score=score, yscale='log')
        hlp.plot_scattered_and_layered(add_async_task, stats, stats_meaned_over_layer, measurement, output_dir, row='optimizer_wd_bn', score=score, yscale='log')
    
    join_async_tasks()


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
    'philips15',
    'philips3'
]
epochs = [10, 20, 30, 40, 80, 120]

#calc_weight_stats(trainers, folds, epochs)

plot_weight_stats()