from nnunet.training.model_restore import load_model_and_checkpoint_files
import numpy as np
import torch
import pandas as pd
import itertools
import os
import sys
import matplotlib.pyplot as plt
import glob

import helpers as hlp

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)  
sys.path.append(os.path.join(parent, 'lipEstimation'))
import lipschitz_approximations as lipapp
import lipschitz_utils as liputils

fold_id_mapping = hlp.get_fold_id_mapping()

def get_flattened(features):
    #TODO check that shape always [out_ch, in_ch, k_h, k_w]
    #TODO what to check input or output??
    #return features.movedim(1, 0).flatten(1)
    return features.flatten(1)

def get_normed(features, dim=1):
    norm = features.norm(dim=dim, keepdim=True)
    scale = 1.0 / (norm + 1e-12)
    zero = norm > 1e-6
    return features * scale * zero, norm * zero

def apply_measure_on_all_combinations(features_a, features_b, measure_fn):
    #a = features_a.repeat_interleave(features_a.shape[0], dim=0)
    #b = features_b.repeat(features_b.shape[0], 1)
    #similarity = measure_fn(a, b)
    #return similarity.view(features_a.shape[0], features_b.shape[0])
    # below is faster and does the same
    triu_ids = torch.triu_indices(features_a.shape[0], features_b.shape[0], device=features_a.device)
    a = features_a[triu_ids[0]]
    b = features_b[triu_ids[1]]
    similarity = measure_fn(a, b)
    z = torch.zeros(features_a.shape[0], features_b.shape[0], device=features_a.device)
    z[triu_ids[0], triu_ids[1]] = similarity
    z[triu_ids[1], triu_ids[0]] = similarity
    return z

def get_cosine_similarity(features_a, features_b):
    return apply_measure_on_all_combinations(
        features_a,
        features_b,
        lambda a,b: torch.nn.functional.cosine_similarity(a, b, dim=1)
    )

def get_l2_similarity(features_a, features_b):
    return apply_measure_on_all_combinations(
        features_a,
        features_b,
        lambda a,b: 1.0 / (1 + (a - b).norm(dim=1))
    )

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

def get_num_similar_features(similarity_matrix, threshold):
    diagonal = similarity_matrix.diagonal()
    # if diagonal contains zeros => dead kernels...
    indices_zero = (diagonal < 0.001).nonzero()
    # if except diagonal close to one => filters similar
    diagonal_matrix = torch.diag(diagonal)

    indices_similar_positive_diag = (diagonal_matrix > threshold).nonzero()
    indices_similar_negative_diag = (diagonal_matrix < -threshold).nonzero()
    triu_ids = torch.triu_indices(similarity_matrix.shape[0], similarity_matrix.shape[1], 1, device=similarity_matrix.device)
    coocurence = similarity_matrix[triu_ids[0], triu_ids[1]]
    indices_similar_positive_co = (coocurence >  threshold).nonzero()
    indices_similar_negative_co = (coocurence < -threshold).nonzero()

    return {
        'num_zero': indices_zero.shape[0],
        # 'num_similar_positive_diag': indices_similar_positive_diag.shape[0],
        # 'num_similar_negative_diag': indices_similar_negative_diag.shape[0],
        # 'num_similar_positive_co': (coocurence >  threshold).nonzero().shape[0],
        # 'num_similar_negative_co': (coocurence < -threshold).nonzero().shape[0],
        # 'mean_co': coocurence.mean().item(),
        # 'std_co': coocurence.std().item(),
        'mean_abs_di': diagonal.abs().mean().item(),
        'std_abs_di': diagonal.abs().std().item(),
        'mean_abs_co': coocurence.abs().mean().item(),
        'std_abs_co': coocurence.abs().std().item()
    }


def generate_stats_compare(parameters_a, parameters_b):
    for name in parameters_a.keys():
        features_a = parameters_a[name]
        features_b = parameters_b[name]
        if len(features_a.shape) == 0:
            continue
        if len(features_a.shape) == 1:
            features_a = torch.unsqueeze(features_a, 1)
            features_b = torch.unsqueeze(features_b, 1)
            #continue

        features_a_flat = get_flattened(features_a)
        features_a_flat_normed, features_a_flat_norm = get_normed(features_a_flat, dim=1)

        features_b_flat = get_flattened(features_b)
        features_b_flat_normed, features_b_flat_norm = get_normed(features_b_flat, dim=1)

        a = features_a_flat_normed
        b = features_b_flat_normed
        # Idea was to append norm of feature vector to measure similarity and considering the length
        # very dependent on the length so tried to norm with feature vector size...
        # features_a_flat_norm = (features_a_flat_norm/features_a_flat_normed.shape[1])
        # features_b_flat_norm = (features_b_flat_norm/features_b_flat_normed.shape[1])
        # a = torch.cat((features_a_flat_normed, features_a_flat_norm), dim=1)
        # b = torch.cat((features_b_flat_normed, features_b_flat_norm), dim=1)
        similarity_matrix = get_cosine_similarity(a, b)
        #similarity_matrix = get_l2_similarity(a, b)
        nums_sim = get_num_similar_features(similarity_matrix, threshold=0.7)


        # TODO tried to get condition of matrix to check numerical stability
        # very slow and almost unbounded...
        # toeplitz_a = get_toeplitz(features_a_flat)
        # toeplitz_a_norm = toeplitz_a.flatten(1).norm(dim=1, p='fro')
        # toeplitz_a_det = toeplitz_a.det().abs()
        # n = toeplitz_a.shape[0]
        # #x = torch.sqrt(1-(n/(toeplitz_a_norm*toeplitz_a_norm)).pow(n) * toeplitz_a_det * toeplitz_a_det)
        # #cond_est_a = torch.sqrt((1+x)/(1-x))
        # cond_est_a = 2 / toeplitz_a_det * (toeplitz_a_norm / torch.tensor(n).sqrt()).pow(n)
        # #cond_a = torch.linalg.cond(toeplitz_a, p='fro')

        yield {
            'name': name,
            **(nums_sim if len(features_a.shape) == 4 else {}),
            'num_features': features_a_flat_norm.shape[0],
            'mean_a': features_a_flat.mean().item(),
            'std_a': features_a_flat.std().item(),
            'mean_b': features_b_flat.mean().item(),
            'std_b': features_b_flat.std().item(),
            'norm_mean_a': features_a_flat_norm.mean().item(),
            'norm_std_a': features_a_flat_norm.std().item(),
            'norm_mean_b': features_b_flat_norm.mean().item(),
            'norm_std_b': features_b_flat_norm.std().item(),
            
            # 'toe_norm_mean_a': toeplitz_a_norm.mean().item(),
            # 'toe_norm_std_a': toeplitz_a_norm.std().item(),
            # 'toe_det_mean_a': toeplitz_a_det.mean().item(),
            # 'toe_det_std_a': toeplitz_a_det.std().item(),
            # 'cond_est_mean_a': cond_est_a.mean().item(),
            # 'cond_est_std_a': cond_est_a.std().item(),
            
        }

def calc_weight_norms(network):
    liputils.compute_module_input_sizes(network, (1, 1, 256, 192))
    lipapp.lipschitz_spectral_ub(network)
    lipapp.lipschitz_frobenius_ub(network)

    stats_norms = pd.DataFrame()
    for name, module in network.named_modules():
        if hasattr(module, 'frob_norm'):
            for k, v in module.frob_norm.items():
                stats_norms.loc[name + '.' + k, 'norm_frob'] = v.item()
        if hasattr(module, 'spectral_norm'):
            stats_norms.loc[name + '.weight', 'norm_spectral'] = module.spectral_norm.item()
    return stats_norms

def calc_weight_stats():
    folder_test = 'archive/old/nnUNet-container/data/testout/Task601_cc359_all_training'
    folder_train = 'archive/old/nnUNet-container/data/nnUNet_trained_models/nnUNet/2d/Task601_cc359_all_training'
    #folder_train = 'data/nnUNet_trained_models/nnUNet/2d/Task601_cc359_all_training'
    trainers = [
        # 'nnUNetTrainerV2_MA_Lab_ResidualUNet_Lovasz_relu_bn_ep40_noDA__nnUNetPlansv2.1',
        # 'nnUNetTrainerV2_MA_Lab_ResidualUNet_Lovasz_relu_bn_ep40__nnUNetPlansv2.1',
        # 'nnUNetTrainerV2_MA_Lab_Lovasz_ep40_noDA__nnUNetPlansv2.1',
        # 'nnUNetTrainerV2_MA_Lab_Lovasz_ep40__nnUNetPlansv2.1',
        # 'nnUNetTrainerV2_MA_Lovasz_noscheduler_depth5_SGD_ep40_noDA__nnUNetPlansv2.1',
        # 'nnUNetTrainerV2_MA_Lovasz_noscheduler_depth5_SGD_ep40__nnUNetPlansv2.1',
        # 'nnUNetTrainerV2_MA_Lovasz_noscheduler_depth5_SGD_ep40_noDA-it2__nnUNetPlansv2.1',
        # 'nnUNetTrainerV2_MA_Lovasz_noscheduler_depth5_SGD_ep40-it2__nnUNetPlansv2.1',
        # 'nnUNetTrainerV2_MA_Lovasz_noscheduler_ep40_noDA__nnUNetPlansv2.1',
        # 'nnUNetTrainerV2_MA_Lovasz_noscheduler_ep40__nnUNetPlansv2.1',
        # 'nnUNetTrainerV2_MA_Lovasz_noscheduler_depth5_ep40_noDA__nnUNetPlansv2.1',
        # 'nnUNetTrainerV2_MA_Lovasz_noscheduler_depth5_ep40__nnUNetPlansv2.1',
        # 'nnUNetTrainerV2_MA_noscheduler_depth5_ep40_noDA__nnUNetPlansv2.1',
        # 'nnUNetTrainerV2_MA_noscheduler_depth5_ep40__nnUNetPlansv2.1',
        # 'nnUNetTrainerV2_MA_noscheduler_depth5_ep40_noDA-it2__nnUNetPlansv2.1',
        # 'nnUNetTrainerV2_MA_noscheduler_depth5_ep40-it2__nnUNetPlansv2.1',
        # 'nnUNetTrainerV2_MA_noscheduler_depth5_ep40_nomirror__nnUNetPlansv2.1',
        # 'nnUNetTrainerV2_MA_noscheduler_depth5_ep40_norotation__nnUNetPlansv2.1',
        # 'nnUNetTrainerV2_MA_noscheduler_depth5_ep40_nogamma__nnUNetPlansv2.1',
        # 'nnUNetTrainerV2_MA_noscheduler_depth5_ep40_noscaling__nnUNetPlansv2.1',


        'nnUNetTrainerV2_MA_noscheduler_depth5_ep120__nnUNetPlansv2.1',
        'nnUNetTrainerV2_MA_noscheduler_depth5_ep120_noDA__nnUNetPlansv2.1',
        'nnUNetTrainerV2_MA_noscheduler_depth5_SGD_ep120__nnUNetPlansv2.1',
        'nnUNetTrainerV2_MA_noscheduler_depth5_SGD_ep120_noDA__nnUNetPlansv2.1',
        'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120__nnUNetPlansv2.1',
        'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120_noDA__nnUNetPlansv2.1',
        'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120__nnUNetPlansv2.1',
        'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120_noDA__nnUNetPlansv2.1',
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
    def load_parameters(trainer, fold, epoch):
        return torch.load(
            os.path.join(
                folder_train,
                trainer,
                'fold_{}'.format(fold_id_mapping[fold]),
                'model_ep_{:0>3}.model'.format(epoch)
            ),
            map_location=torch.device('cpu')
        )['state_dict']

    # folder_train = 'code/lab_med_viz/baseline_results1'
    # trainers = [
    #     'baseline_focal_lovasz_Adam_None_None_frontback',
    #     'baseline_focal_lovasz_Adam_rand_aug_None_frontback'
    # ]
    # folds = [
    #     'siemens15',
    #     'siemens3',
    #     'ge15',
    #     'ge3',
    #     'philips15',
    #     'philips3'
    # ]
    # epochs = [39]#[9, 19, 29, 39]
    # def load_parameters(trainer, fold, epoch):
    #     return torch.load(
    #         os.path.join(
    #             folder_train,
    #             trainer,
    #             'mode_{}'.format(fold_id_mapping[fold]),
    #             'e_{}.pth'.format(epoch)
    #         ),
    #         map_location=torch.device('cpu')
    #     )['model_state_dict']

    for trainer, fold, epoch in itertools.product(trainers, folds, epochs):
        print(trainer, fold, epoch)

        tmodel, params = load_model_and_checkpoint_files(
            os.path.join(folder_train, trainer),
            fold_id_mapping[fold],
            mixed_precision=True,
            checkpoint_name='model_ep_{:0>3}'.format(epoch)
        )
        if epoch > 0:
            tmodel.load_checkpoint_ram(params[0], False)
        tmodel.network.do_ds = False
        for p in tmodel.network.parameters():
            p.requires_grad = False
        
        parameters = dict(tmodel.network.named_parameters())
        #parameters = load_parameters(trainer, fold, epoch)
        
        
        stats_norms = calc_weight_norms(tmodel.network)
        

        stats = pd.DataFrame(generate_stats_compare(
            parameters,
            parameters
        )).set_index('name')
        print(set(stats_norms.index) - set(stats.index))
        stats = stats.join(stats_norms)

        stats['trainer'] = trainer
        stats['fold_train'] = fold
        stats['epoch'] = epoch
        stats.reset_index().to_csv('data/csv/weights-{}-{}-{}.csv'.format(trainer, fold, epoch))

def plot_weight_stats():
    stats = pd.concat([
        pd.read_csv(path) for path in glob.iglob('data/csv/weights-*.csv', recursive=False)
    ])
    stats['trainer_short'] = stats['trainer'].apply(hlp.get_trainer_short)

    index = ['trainer', 'fold_train', 'epoch']
    domainshift_scores = pd.concat([
        hlp.load_domainshift_scores(
            'Task601_cc359_all_training',
            *ids
        ) for ids in stats[index].drop_duplicates().values.tolist()
    ])

    stats = stats.join(domainshift_scores.set_index(index), on=index)

    stats['stable_rank'] = stats['norm_frob'] / (stats['norm_spectral'] + 0.00001)
    stats = stats[stats['name'].str.endswith('.weight')]
    stats['name'] = stats['name'].str.replace('.weight', '')

    layers_set = set(hlp.get_layers_ordered())
    layers_position_map = hlp.get_layers_position_map()
    stats.rename(
        columns={
            'name': 'layer'
        },
        inplace=True
    )
    stats = stats[stats['layer'].isin(layers_set)]
    stats['layer_pos'] = stats['layer'].apply(lambda d: layers_position_map.get(d))

    import random
    random.seed(0)
    stats['layer_pos'] = stats['layer_pos'].apply(lambda d: d + (random.random() - 0.5) * 0.7)

    columns = [
        'num_features',
        'num_zero',
        'mean_abs_di',
        'std_abs_di',
        'mean_abs_co',
        'std_abs_co',
        'norm_mean_a',
        'norm_std_a',
        'norm_frob',
        'norm_spectral',
        'stable_rank'
    ]

    joined = stats.groupby(['trainer_short', 'fold_train', 'epoch', 'layer_pos']).agg([
        'mean'#, 'std', 'sum', 'min', 'max'
    ]).reset_index()
    joined['psize'] = joined['epoch'] / 10
    column_color = 'sdice_score_diff_mean'

    for column in columns:
        fig, axes = hlp.create_scatter_plot(
            joined,
            column_x='layer_pos',
            column_y=(column, 'mean'),
            column_subplots='trainer_short',
            column_size='psize',
            column_color=column_color,
            ncols=1,
            figsize=(32, 24),
            lim_same_x=True,
            lim_same_y=True,
            lim_same_c=True,
            colormap='cool'
        )

        fig.savefig('data/fig/weights-{}.png'.format(column))
        plt.close(fig)


#calc_weight_stats()

plot_weight_stats()
