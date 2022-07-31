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
    for name, features in parameters.items():
        # for conv [out_ch, in_ch, k_h, k_w]
        if len(features.shape) == 0:
            continue
        
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
            'features_noise': features_noise[..., None, None],
            'features_zero': diagonal_zero[None, None, None, None],
            'features_norm': features_flat_norm[..., None, None],
            'similarity_di_abs': diagonal.abs()[..., None, None, None],
            'similarity_co_abs': coocurence.abs()[..., None, None, None],
            'features2_zero': diagonals2_zero[..., None, None, None],
            'features2_norm': features_flat2_norm[..., None],
            'similarity2_di_abs': diagonals2.abs()[..., None, None],
            'similarity2_co_abs': coocurences2.abs()[..., None, None],
        }

        features_moments = {
            k: v for name, value in features_base.items() for k, v in hlp.get_moments_recursive(name, value)
        }

        out = {
            'name': name,
            'num_features': features_flat_norm.shape[0],
            **dict(map(lambda d: (d[0], d[1].item()), features_moments.items()))
            
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
                stats_norms.loc[name + '.' + k, 'norm_frob'] = v.item()
        if hasattr(module, 'spectral_norm'):
            stats_norms.loc[name + '.weight', 'norm_spectral'] = module.spectral_norm.item()
    return stats_norms

def calc_weight_stats(folder_train, trainers, folds, epochs):
    output_dir = 'data/csv/weights'
    os.makedirs(output_dir, exist_ok=True)

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
        

        stats = pd.DataFrame(generate_stats_weights(
            parameters
        )).set_index('name')
        print(set(stats_norms.index) - set(stats.index))
        stats = stats.join(stats_norms)

        stats['trainer'] = trainer
        stats['fold_train'] = fold
        stats['epoch'] = epoch
        stats.reset_index().to_csv(
            os.path.join(
                output_dir,
                'weights-{}-{}-{}.csv'.format(trainer, fold, epoch)
            )
        )

def plot_weight_stats():
    stats = pd.concat([
        pd.read_csv(path) for path in glob.iglob('data/csv/weights/weights-*.csv', recursive=False)
    ])
    stats['trainer_short'] = stats['trainer'].apply(hlp.get_trainer_short)

    index = ['trainer', 'fold_train', 'epoch']
    scores = pd.concat([
        hlp.get_scores(
            'Task601_cc359_all_training',
            *ids
        ) for ids in stats[index].drop_duplicates().values.tolist()
    ])
    #index = [*index, 'fold_test']
    scores = scores.groupby(index).agg({
        'dice_score': ['mean', 'std'],
        'sdice_score': ['mean', 'std']
    }).reset_index()
    stats = stats.join(scores.set_index(index), on=index)
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

    # import random
    # random.seed(0)
    # stats['layer_pos'] = stats['layer_pos'].apply(lambda d: d + (random.random() - 0.5) * 0.7)



    stats['stable_rank'] = stats['norm_frob'] / (stats['norm_spectral'] + 0.00001)
    

    columns = [
        'stable_rank',
        *list(filter(lambda c: any(s in c for s in ['norm_', 'features', 'similarity', 'norm_']), stats.columns.values.tolist()))
    ]

    stats = hlp.numerate_nested(stats, [
        'trainer_short',
        'fold_train',
        'epoch',
    ])

    stats['psize'] = stats['epoch'] / 10

    column_color = ('sdice_score', 'mean')

    output_dir = 'data/fig/weights'
    os.makedirs(output_dir, exist_ok=True)

    for column in columns:
        # fig, axes = hlp.create_plot(
        #     joined,
        #     column_x='layer_pos',
        #     column_y=(column, 'mean'),
        #     column_subplots='trainer_short',
        #     column_size='psize',
        #     column_color=column_color,
        #     ncols=1,
        #     figsize=(32, 24),
        #     lim_same_x=True,
        #     lim_same_y=True,
        #     lim_same_c=True,
        #     colormap='cool'
        # )
        fig, axes = hlp.create_plot(
            stats,
            column_x='x',
            column_y=column,
            column_subplots='layer_pos',
            column_size='psize',
            column_color=column_color,
            ncols=2,
            figsize=(42, 24*4),
            lim_same_x=False,
            lim_same_y=False,
            lim_same_c=True,
            colormap='cool',
            fig_num='weights',
            fig_clear=True
        )
        fig.savefig(
            os.path.join(
                output_dir,
                'weights-{}.png'.format(column)
            ),
            bbox_inches='tight',
            pad_inches=0
        )
        plt.close(fig)


def plot_weights(trainers, folds, epochs, load_parameters):
    layers_position_map = hlp.get_layers_position_map()
    for trainer, fold_train, epoch in itertools.product(trainers, folds, epochs):
        print(trainer, fold_train, epoch)

        parameters_dict = load_parameters(trainer, fold_train, epoch)

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




#calc_weight_stats(folder_train, trainers, folds, epochs)

#plot_weight_stats()
#plot_weights(trainers, folds, epochs, load_parameters)