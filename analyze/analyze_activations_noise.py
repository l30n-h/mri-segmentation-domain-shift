import numpy as np
import torch
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
import os
import glob
import helpers as hlp


def get_noise_estimates(input):
    #assert len(input.shape) == 3
    out = torch.nn.functional.conv2d(
        input,
        torch.tensor(
            [[[
                [ 1, -2,  1],
                [-2,  4, -2],
                [ 1, -2,  1]
            ]]],
            device=input.device,
            dtype=input.dtype
        )
    ).to(dtype=torch.float32)
    norm_const = 0.20888 # == torch.sqrt(torch.tensor(0.5 * torch.pi)) / 6.0
    noise_mean = out.abs().mean(
        dim=(-2, -1)
    ) * norm_const
    return noise_mean

def get_activations_noise_estimate(activations):
    a = activations.reshape(
        activations.shape[0] * activations.shape[1],
        1,
        *activations.shape[2:4]
    )
    noise_means = get_noise_estimates(a)
    #noise_means = noise_means.reshape(*activations.shape[0:2], -1, -1)
    return noise_means.mean(), noise_means.std()

def generate_stats_activation_maps(activations_dict):
    for name, activation_map in activations_dict.items():

        #todo depends on network params
        layers_position_map = hlp.get_layers_position_map()
        layer_div = 2 ** (layers_position_map[name] // 4)
        activation_map2d = activation_map.reshape(
            *activation_map.shape[0:2],
            max(16, 256 // layer_div),
            max(24, 192 // layer_div)
        )

        noise_mean, noise_std = get_activations_noise_estimate(
            activation_map2d.to(device='cuda')
        )

        activation_map = activation_map2d.to(dtype=torch.float32)
        
        activations_mean = activation_map.mean(dim=(-2, -1))
        activations_std = activation_map.mean(dim=(-2, -1))

        # activations_mean_feat_mean = activations_mean.mean(dim=0)
        # activations_mean_feat_std = activations_mean.std(dim=0)
        # activations_mean_slice_mean = activations_mean.mean(dim=1)
        # activations_mean_slice_std = activations_mean.std(dim=1)

        out = {
            'name': name,
            'num_slices': activation_map.shape[0],
            'num_feats':activation_map.shape[1],
            'mean_mean': activations_mean.mean().item(),
            'mean_std': activations_mean.std().item(),
            'std_mean': activations_std.mean().item(),
            'std_std': activations_std.std().item(),

            # 'mean_feat_mean_std': activations_mean_feat_mean.std().item(),
            # 'mean_feat_std_mean': activations_mean_feat_std.mean().item(),
            # # could be used to find best where all real layer with lrelu feat_std_std->mean->[mean std] is max.
            # 'mean_feat_std_std': activations_mean_feat_std.std().item(),
            # 'mean_slice_mean_std': activations_mean_slice_mean.std().item(),
            # 'mean_slice_std_mean': activations_mean_slice_std.mean().item(),
            # 'mean_slice_std_std': activations_mean_slice_std.std().item(),
            
            'noise_mean': noise_mean.item(),
            'noise_std': noise_std.item()
        }
        yield out


def calc_noise_stats(task, trainers, folds, epochs):
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
    
    for trainer, fold_train, epoch in itertools.product(trainers, folds, epochs):
        print(task, trainer, fold_train, epoch)

        directory_activations = os.path.join(
            hlp.get_testdata_dir(task, trainer, fold_train, epoch),
            'activations-small'
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
        stats.to_csv('data/csv/activations-noise-{}-{}-{}.csv'.format(trainer, fold_train, epoch))


def plot_activation_maps_as_image(scores, id_path_map, layer_name, out_path):
    scores = scores.set_index('id')
    scores_fold_test = scores.groupby('fold_test').agg({
        'sdice_score': ['mean', 'max']
    })

    scores = scores[scores.index.isin(set(id_path_map.keys()))]
    scores_filtered = scores[scores['is_validation']]
    file_ids = scores_filtered.reset_index().groupby('fold_test').agg({
        'id': ['first']
    })[('id', 'first')].tolist()

    folds_test = scores['fold_test']
    num_folds_test = len(folds_test.unique())
    fig, axes = plt.subplots(figsize=(32, 24), nrows=num_folds_test // 2, ncols=2)
    axes_flat = axes.flat

    fold_id_mapping = hlp.get_fold_id_mapping()

    for file_id in file_ids:
        scores_id = scores.loc[file_id]
        fold_test = scores_id['fold_test']
        fold_test_id = fold_id_mapping[fold_test]
        
        activations_dict = torch.load(
            id_path_map[file_id],
            map_location=torch.device('cpu')
        )
        activations = activations_dict[layer_name]

        #todo depends on network params
        layers_position_map = hlp.get_layers_position_map()
        layer_div = 2 ** (layers_position_map[layer_name] // 4)
        activations = activations.reshape(
            *activations.shape[0:2], max(16, 256 // layer_div), max(24, 192 // layer_div)
        )
        
        noise_mean, noise_std = get_activations_noise_estimate(
            activations.to(device='cuda')
        )

        y = activations.mean(dim=0)[0].numpy()

        axes_flat[fold_test_id].imshow(y)
        axes_flat[fold_test_id].set_title("{} / {} / {}  {} / {}".format(
            scores_id['sdice_score'].round(4),
            scores_fold_test.loc[fold_test][('sdice_score', 'mean')].round(4),
            scores_fold_test.loc[fold_test][('sdice_score', 'max')].round(4),
            noise_mean.item(),
            noise_std.item()
        ))
    
    fig.savefig(out_path)
    plt.close(fig)

def plot_activation_maps(task, trainers, folds, epochs):
    layers_ordered = hlp.get_layers_ordered()

    for trainer, fold_train, epoch in itertools.product(trainers, folds, epochs):
        print(task, trainer, fold_train, epoch)

        directory_activations = os.path.join(
            hlp.get_testdata_dir(task, trainer, fold_train, epoch),
            'activations-small'
        )

        scores = hlp.get_scores(task, trainer, fold_train, epoch)

        id_path_map = dict(map(
            lambda p: (hlp.get_id_from_filename(os.path.basename(p)), p),
            glob.iglob(os.path.join(directory_activations, '*_activations.pkl'))
        ))

        layer_name = layers_ordered[19]#[1]
        plot_activation_maps_as_image(
            scores,
            id_path_map,
            layer_name,
            'data/fig/activation_maps-{}-{}-{}-{}.png'.format(trainer, fold_train, epoch, layer_name)
        )


def plot_noise():
    stats = pd.concat([
        pd.read_csv(path) for path in glob.iglob('data/csv/activations-noise-*.csv', recursive=False)
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

    import random
    random.seed(0)
    stats['layer_pos'] = stats['layer_pos'].apply(lambda d: d + (random.random() - 0.5) * 0.7)

    print(stats.head().to_string())
    stats = stats[stats['is_validation']]
    stats = stats[~stats['layer'].str.contains('.conv')]

    columns = [
        #'dice_score',
        'sdice_score',
        # 'num_slices',
        # 'num_feats',
        'mean_mean',
        'mean_std',
        'std_mean',
        'std_std',
        'noise_mean',
        'noise_std'
    ]

    stats_grouped = stats.groupby([
        'trainer_short', 'fold_train', 'epoch', 'layer_pos'
    ])[columns].agg([
        'mean'#, 'std', 'sum', 'min', 'max'
    ]).reset_index()
    stats_grouped['psize'] = stats_grouped['epoch'] / 10
    column_color = ('sdice_score', 'mean')
    #stats_grouped['fold_train_id'] = pd.factorize(stats_grouped['fold_train'], sort=True)[0]
    column_color = 'fold_train_id'

    for column in columns:
        fig, axes = hlp.create_scatter_plot(
            stats_grouped,
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

        fig.savefig('data/fig/activations-noise-{}.png'.format(column))
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


#calc_noise_stats(task, trainers, folds, epochs)

plot_activation_maps(task, trainers, folds, epochs)
plot_noise()