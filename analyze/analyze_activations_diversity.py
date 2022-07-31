import numpy as np
import torch
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
import os
import glob
import helpers as hlp


def generate_stats_activation_maps(activations_dict):
    for name, activation_maps in activations_dict.items():
        activation_maps = activation_maps.to(device='cuda').to(dtype=torch.float32)

        maps_flat = activation_maps.flatten(2)
        maps_flat_normed, maps_flat_norm = hlp.get_normed(
            maps_flat,
            dim=2
        )

        similarity_matrices = torch.stack([ 
            hlp.get_cosine_similarity(
                map_flat_normed,
                map_flat_normed
            ) for map_flat_normed in maps_flat_normed
        ])

        diagonals = similarity_matrices.diagonal(dim1=1, dim2=2)
        diagonals_zero = (diagonals < 0.001).count_nonzero(dim=0).float()
        triu_ids = torch.triu_indices(similarity_matrices.shape[1], similarity_matrices.shape[2], 1, device=similarity_matrices.device)
        coocurences = similarity_matrices[:,triu_ids[0], triu_ids[1]]

        out = {
            'name': name,
            'num_slices': activation_maps.shape[0],
            'num_feats': activation_maps.shape[1],
            'num_zero': diagonals_zero.mean(dim=0).item(),
            'di_abs_mean_1_mean_0': diagonals.abs().mean(dim=1).mean(dim=0).item(),
            'di_abs_std_1_mean_0': diagonals.abs().std(dim=1).mean(dim=0).item(),
            'di_abs_mean_1_std_0': diagonals.abs().mean(dim=1).std(dim=0).item(),
            'di_abs_std_1_std_0': diagonals.abs().std(dim=1).std(dim=0).item(),

            'co_abs_mean_1_mean_0': coocurences.abs().mean(dim=1).mean(dim=0).item(),
            'co_abs_std_1_mean_0': coocurences.abs().std(dim=1).mean(dim=0).item(),
            'co_abs_mean_1_std_0': coocurences.abs().mean(dim=1).std(dim=0).item(),
            'co_abs_std_1_std_0': coocurences.abs().std(dim=1).std(dim=0).item(),

            'norm_mean_1_mean_0': maps_flat_norm.mean(dim=1).mean(dim=0).item(),
            'norm_std_1_mean_0': maps_flat_norm.std(dim=1).mean(dim=0).item(),
            'norm_mean_1_std_0': maps_flat_norm.mean(dim=1).std(dim=0).item(),
            'norm_std_1_std_0': maps_flat_norm.std(dim=1).std(dim=0).item(),
        }
        yield out


def calc_diversity_stats(task, trainers, folds, epochs):
    def get_stats_per_id(file_id, file_path):
        activations_dict = torch.load(
            file_path,
            map_location=torch.device('cpu')
        )
        stats = pd.DataFrame(
            generate_stats_activation_maps(activations_dict)
        )
        stats['id'] = file_id
        return stats
    

    output_dir = 'data/csv/activations-diversity'
    os.makedirs(output_dir, exist_ok=True)

    for trainer, fold_train, epoch in itertools.product(trainers, folds, epochs):
        print(task, trainer, fold_train, epoch)

        directory_activations = os.path.join(
            hlp.get_testdata_dir(task, trainer, fold_train, epoch),
            'activations-small-fullmap'
        )

        id_path_map = dict(map(
            lambda p: (hlp.get_id_from_filename(os.path.basename(p)), p),
            glob.iglob(os.path.join(directory_activations, '*_activations.pkl'))
        ))
        
        scores = hlp.get_scores(task, trainer, fold_train, epoch)

        stats = pd.concat([
            get_stats_per_id(file_id, file_path) for file_id, file_path in id_path_map.items()
        ])

        stats = stats.join(scores.set_index('id'), on='id')
        stats.to_csv(
            os.path.join(
                output_dir,
                'activations-diversity-{}-{}-{}.csv'.format(trainer, fold_train, epoch)
            )
        )


def print_scores():
    for trainer, fold_train, epoch in itertools.product(trainers, folds, epochs):
        print(task, trainer, fold_train, epoch)

        scores = hlp.get_scores(task, trainer, fold_train, epoch)

        ids = set([
            'CC0001',
            'CC0009',
            'CC0061',
            'CC0069',
            'CC0121',
            'CC0129',
            'CC0181',
            'CC0189',
            'CC0241',
            'CC0249',
            'CC0301',
            'CC0309',
        ])
        print(scores[scores['id'].isin(ids)].sort_values(['sdice_score']))

def plot_activations():
    layers_position_map = hlp.get_layers_position_map()
    for trainer, fold_train, epoch in itertools.product(trainers, folds, epochs):
        print(task, trainer, fold_train, epoch)

        directory_activations = os.path.join(
            hlp.get_testdata_dir(task, trainer, fold_train, epoch),
            'activations-small-fullmap'
        )

        id_path_map = dict(map(
            lambda p: (hlp.get_id_from_filename(os.path.basename(p)), p),
            glob.iglob(os.path.join(directory_activations, '*_activations.pkl'))
        ))

        file_ids = set([
            'CC0001',
            'CC0009',
            'CC0061',
            'CC0069',
            'CC0121',
            'CC0129',
            'CC0181',
            'CC0189',
            'CC0241',
            'CC0249',
            'CC0301',
            'CC0309',
        ])

        for file_id in file_ids:
            activations_dict = torch.load(
                id_path_map.get(file_id),
                map_location=torch.device('cpu')
            )
            for name, activation_maps in activations_dict.items():
                layer_id = layers_position_map.get(name)
                activation_maps = activation_maps.to(dtype=torch.float32)

                #activation_maps = activation_maps
                #activation_maps = activation_maps / activation_maps.abs().flatten(2).max(dim=2)[0][:,:,None, None]
                activation_maps = activation_maps / activation_maps.abs().flatten(1).max(dim=1)[0][:,None,None, None]

                merged = torch.nn.functional.pad(
                    activation_maps,
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

                stats_name = 'activation_maps_pure_slicenormed'
                output_dir = 'data/fig/activation_maps/{}-{}'.format(stats_name, trainer)
                os.makedirs(output_dir, exist_ok=True)
                plt.imsave(
                    os.path.join(output_dir, '{}-{}-{}-{}-{}-{}.jpg'.format(
                        stats_name, trainer, fold_train, epoch, file_id, layer_id
                    )),
                    merged,
                    vmin=-mval,
                    vmax=mval
                )


def plot_activations_moments():
    layers_position_map = hlp.get_layers_position_map()
    for trainer, fold_train, epoch in itertools.product(trainers, folds, epochs):
        print(task, trainer, fold_train, epoch)

        directory_activations = os.path.join(
            hlp.get_testdata_dir(task, trainer, fold_train, epoch),
            'activations-small-fullmap'
        )

        id_path_map = dict(map(
            lambda p: (hlp.get_id_from_filename(os.path.basename(p)), p),
            glob.iglob(os.path.join(directory_activations, '*_activations.pkl'))
        ))

        file_ids = set([
            'CC0001',
            'CC0009',
            'CC0061',
            'CC0069',
            'CC0121',
            'CC0129',
            'CC0181',
            'CC0189',
            'CC0241',
            'CC0249',
            'CC0301',
            'CC0309',
        ])

        for file_id in file_ids:
            activations_dict = torch.load(
                id_path_map.get(file_id),
                map_location=torch.device('cpu')
            )
            for name, activation_maps in activations_dict.items():
                layer_id = layers_position_map.get(name)
                activation_maps = activation_maps.to(dtype=torch.float32)

                activation_moments = torch.cat(
                    (
                        activation_maps.mean(dim=1, keepdim=True),
                        activation_maps.std(dim=1, keepdim=True),
                        hlp.standardized_moment(activation_maps, order=3, dim=1, keepdim=True),
                        hlp.standardized_moment(activation_maps, order=4, dim=1, keepdim=True)
                    ),
                    dim=1
                )
                activation_moments_normed = activation_moments / activation_moments.abs().flatten(2).max(dim=2)[0][:,:,None, None]
                merged = torch.nn.functional.pad(
                    activation_moments_normed,
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

                stats_name = 'activation_maps_pure_moments_normed'
                output_dir = 'data/fig/activation_maps/{}-{}'.format(stats_name, trainer)
                os.makedirs(output_dir, exist_ok=True)
                plt.imsave(
                    os.path.join(output_dir, '{}-{}-{}-{}-{}-{}.jpg'.format(
                        stats_name, trainer, fold_train, epoch, file_id, layer_id
                    )),
                    merged,
                    vmin=-mval,
                    vmax=mval
                )


def plot_activations_moments_combined():
    layers_position_map = hlp.get_layers_position_map()
    for trainer, fold_train, epoch in itertools.product(trainers, folds, epochs):
        print(task, trainer, fold_train, epoch)

        directory_activations = os.path.join(
            hlp.get_testdata_dir(task, trainer, fold_train, epoch),
            'activations-small-fullmap'
        )

        id_path_map = dict(map(
            lambda p: (hlp.get_id_from_filename(os.path.basename(p)), p),
            glob.iglob(os.path.join(directory_activations, '*_activations.pkl'))
        ))

        ids = set([
            'CC0001',
            'CC0009',
            'CC0061',
            'CC0069',
            'CC0121',
            'CC0129',
            'CC0181',
            'CC0189',
            'CC0241',
            'CC0249',
            'CC0301',
            'CC0309',
        ])

        scores = hlp.get_scores(task, trainer, fold_train, epoch)
        file_ids = scores[scores['id'].isin(ids)].sort_values(['sdice_score'])['id']

        activations_dict_merged = dict()

        for file_id in file_ids:
            activations_dict = torch.load(
                id_path_map.get(file_id),
                map_location=torch.device('cpu')
            )
            for name, activation_maps in activations_dict.items():
                activations_dict_merged.setdefault(name, []).append(
                    activation_maps
                )
            
        for name, activation_maps in activations_dict_merged.items():
            layer_id = layers_position_map.get(name)
            activation_maps = torch.stack(activation_maps)
            activation_maps = activation_maps.to(dtype=torch.float32)

            activation_moments_all = torch.cat(
                (
                    activation_maps.mean(dim=2, keepdim=True),
                    activation_maps.std(dim=2, keepdim=True),
                    hlp.standardized_moment(activation_maps, order=3, dim=2, keepdim=True),
                    hlp.standardized_moment(activation_maps, order=4, dim=2, keepdim=True)
                ),
                dim=2
            ).swapaxes(0, 2)
            for i, metric in enumerate(['mean', 'std', 'skewness', 'kurtosis']):
                activation_moments = activation_moments_all[i]
                activation_moments_normed = activation_moments#activation_moments / activation_moments.abs().flatten(2).max(dim=2)[0][:,:,None, None]
                merged = torch.nn.functional.pad(
                    activation_moments_normed,
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

                stats_name = 'activation_maps_pure_moments_normed_combined'
                output_dir = 'data/fig/activation_maps/{}-{}'.format(stats_name, trainer)
                os.makedirs(output_dir, exist_ok=True)
                plt.imsave(
                    os.path.join(output_dir, '{}-{}-{}-{}-{}-{}.jpg'.format(
                        stats_name, metric, trainer, fold_train, epoch, layer_id
                    )),
                    merged,
                    vmin=-mval,
                    vmax=mval
                )


def plot_diversity():
    stats = pd.concat([
        pd.read_csv(path) for path in glob.iglob('data/csv/activations-diversity/activations-diversity-*.csv', recursive=False)
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
    stats['same_fold'] = pd.factorize(
        stats['fold_train'] == stats['fold_test'],
        sort=True
    )[0]
    
    # stats['layer_pos'] = stats['layer_pos'] + (pd.factorize(stats['id'], sort=True)[0] * 0.7 / stats['id'].nunique())
    # import random
    # random.seed(0)
    # stats['layer_pos'] = stats['layer_pos'].apply(lambda d: d + (random.random() - 0.5) * 0.7)

    #stats = stats[stats['is_validation']]
    #stats = stats[~stats['layer'].str.contains('.instnorm')]
    #stats = stats[~stats['layer'].str.contains('.conv')]
    #stats = stats[~stats['layer'].str.contains('seg_outputs.')]
    #stats = stats[~stats['layer'].str.contains('tu.')]

    #stats = stats[stats['fold_train'].str.contains('ge3')]

    columns = [
        'dice_score',
        'sdice_score',
        *list(filter(lambda c: any(s in c for s in ['num_', 'di_', 'co_', 'norm_']), stats.columns.values.tolist()))
    ]

    stats = hlp.numerate_nested(stats, [
        'trainer_short',
        'epoch',
        'fold_train',
        'same_fold',
        'id',
    ])

    column_color = 'sdice_score'
    stats['psize'] = (stats['is_validation'] & stats['same_fold'])*6 + 2 #stats['epoch'] / 20
    # stats['fold_train_id'] = pd.factorize(stats['fold_train'], sort=True)[0]
    # column_color = 'fold_train_id'
    # stats['fold_test_id'] = pd.factorize(stats['fold_test'], sort=True)[0]
    # column_color = 'fold_test_id'
    # stats['same_fold'] = pd.factorize(
    #     stats['fold_train'] == stats['fold_test'],
    #     sort=True
    # )[0]
    # column_color = 'same_fold'
    # stats['is_validation_factor'] = pd.factorize(stats['is_validation'], sort=True)[0]
    # column_color = 'is_validation_factor'

    output_dir = 'data/fig/activations-diversity'
    os.makedirs(output_dir, exist_ok=True)

    for column in columns:
        print(column)
        # fig, axes = hlp.create_plot(
        #     stats,
        #     column_x='layer_pos',
        #     column_y=column,
        #     column_subplots='trainer_short',
        #     column_size='psize',
        #     column_color=column_color,
        #     ncols=1,
        #     figsize=(42, 24),
        #     lim_same_x=True,
        #     lim_same_y=True,
        #     lim_same_c=True,
        #     colormap='cool',
        #     fig_num='activations_diversity',
        #     fig_clear=True
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
            fig_num='activations_diversity',
            fig_clear=True
        )
        fig.savefig(
            os.path.join(
                output_dir,
                'activations-diversity-{}.png'.format(column)
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
epochs = [40, 120] #[10,20,30,40,80,120]


#calc_diversity_stats(task, trainers, folds, epochs)

#plot_diversity()

plot_activations()
#plot_activations_moments()
#plot_activations_moments_combined()

print_scores()