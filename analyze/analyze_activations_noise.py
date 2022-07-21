import numpy as np
import torch
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
import os
import glob
import helpers as hlp


def get_activations_noise_estimate(activations):
    a = activations.reshape(
        activations.shape[0] * activations.shape[1],
        1,
        *activations.shape[2:4]
    )
    noise_means = hlp.get_noise_estimates(a)
    return noise_means.reshape(*activations.shape[0:2])

def generate_stats_activation_maps(activations_dict):
    

    for name, activation_map in activations_dict.items():
        activation_map = activation_map.to(device='cuda')

        activation_noise = get_activations_noise_estimate(
            activation_map
        )

        activation_map = activation_map.to(dtype=torch.float32)

        activations_base = {
            'activations': activation_map,
            'activations_abs': activation_map.abs(),
            'activations_positiv': (activation_map > 0).float(),
            'activations_noise': activation_noise[:, :, None, None]
        }
        
        activation_moments = {
            k: v for name, value in activations_base.items() for k, v in hlp.get_moments_recursive(name, value)
        }

        #TODO SSIM index (https://en.wikipedia.org/wiki/Structural_similarity)
        #TODO maybe mean/std somewhat snr
        #TODO blob detection (https://scikit-image.org/docs/stable/api/skimage.feature.html)
        

        # activations_mean_feat_mean = activations_mean.mean(dim=0)
        # activations_mean_feat_std = activations_mean.std(dim=0)
        # activations_mean_slice_mean = activations_mean.mean(dim=1)
        # activations_mean_slice_std = activations_mean.std(dim=1)

        out = {
            'name': name,
            'num_slices': activation_map.shape[0],
            'num_feats': activation_map.shape[1],
            **dict(map(lambda d: (d[0], d[1].item()), activation_moments.items()))

            # 'mean_feat_mean_std': activations_mean_feat_mean.std().item(),
            # 'mean_feat_std_mean': activations_mean_feat_std.mean().item(),
            # # could be used to find best where all real layer with lrelu feat_std_std->mean->[mean std] is max.
            # 'mean_feat_std_std': activations_mean_feat_std.std().item(),
            # 'mean_slice_mean_std': activations_mean_slice_mean.std().item(),
            # 'mean_slice_std_mean': activations_mean_slice_std.mean().item(),
            # 'mean_slice_std_std': activations_mean_slice_std.std().item(),
            
            # 'noise_mean': noise_mean.item(),
            # 'noise_std': noise_std.item()
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
    
    folder_suffixes = ['-encoder', '-decoder']
    for trainer, fold_train, epoch, folder_suffix in itertools.product(trainers, folds, epochs, folder_suffixes):
        print(task, trainer, fold_train, epoch, folder_suffix)

        directory_activations = os.path.join(
            hlp.get_testdata_dir(task, trainer, fold_train, epoch),
            'activations-small{}'.format(folder_suffix)
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
        stats.to_csv('data/csv/activations-noise{}-{}-{}-{}.csv'.format(
            folder_suffix,
            trainer,
            fold_train,
            epoch
        ))


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
        
        activation_noise = get_activations_noise_estimate(
            activations.to(device='cuda')
        )
        activation_noise.mean(), activation_noise.std()

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
            'activations-small-decoder'
        )

        scores = hlp.get_scores(task, trainer, fold_train, epoch)

        id_path_map = dict(map(
            lambda p: (hlp.get_id_from_filename(os.path.basename(p)), p),
            glob.iglob(os.path.join(directory_activations, '*_activations.pkl'))
        ))

        layer_name = layers_ordered[-2]#[19]#[1]
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

    stats['layer_pos'] = stats['layer_pos'] + (pd.factorize(stats['id'], sort=True)[0] * 0.7 / stats['id'].nunique())

    stats = stats[stats['is_validation']]
    #stats = stats[~stats['layer'].str.contains('.instnorm')]
    #stats = stats[~stats['layer'].str.contains('.conv')]
    #stats = stats[~stats['layer'].str.contains('seg_outputs.')]
    #stats = stats[~stats['layer'].str.contains('tu.')]

    #stats = stats[stats['fold_train'].str.contains('ge3')]

    
    columns = [
        #'dice_score',
        'sdice_score',
        *list(filter(lambda c: c.startswith('activation'), stats.columns.values.tolist()))[1000:]
    ]

    stats['psize'] = stats['epoch'] / 10
    column_color = 'sdice_score'
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

    output_dir = 'data/fig/activations-noise'
    os.makedirs(output_dir, exist_ok=True)

    for column in columns:
        print(column)
        fig, axes = hlp.create_plot(
            stats,
            column_x='layer_pos',
            column_y=column,
            column_subplots='trainer_short',
            column_size='psize',
            column_color=column_color,
            ncols=1,
            figsize=(42, 24),
            lim_same_x=True,
            lim_same_y=True,
            lim_same_c=True,
            colormap='cool',
            fig_num='activations_noise',
            fig_clear=True
        )

        fig.savefig(os.path.join(
            output_dir,
            'activations-noise-{}.png'.format(column)
        ))
        plt.close(fig)

def plot_noise_layerwise():
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
    stats['same_fold'] = pd.factorize(
        stats['fold_train'] == stats['fold_test'],
        sort=True
    )[0]


    columns = [
        #'dice_score',
        'sdice_score',
        *list(filter(
            lambda c: c.startswith('activation'),
            filter(
                lambda c: c.endswith('_mean_0') and 'skewness' not in c and 'kurtosis' not in c,
                stats.columns.values.tolist()
            )
        ))
    ]

    
    
    print(stats)
    #stats = stats[stats['fold_train'].str.contains('siemens')]
    #stats = stats[stats['trainer_short'].str.contains('SGD')]

    stats = hlp.numerate_nested(stats, [
        'trainer_short',
        'epoch',
        'fold_train',
        'same_fold',
        'id',
    ])

    column_color = 'sdice_score'
    stats['psize'] = (stats['is_validation'] & stats['same_fold'])*6 + 2 #stats['epoch'] / 20
    
    #'trainer_short' 'epoch' 'same_fold' 'id'

    output_dir = 'data/fig/activations-noise-layerwise'
    os.makedirs(output_dir, exist_ok=True)

    for column in columns:
        print(column)
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
            fig_num='activations_noise',
            fig_clear=True
        )

        fig.savefig(
            os.path.join(
                output_dir,
                'activations-noise-layerwise-{}.png'.format(column)
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


#calc_noise_stats(task, trainers, folds, epochs)

#plot_activation_maps(task, trainers, folds, epochs)
#plot_noise()
plot_noise_layerwise()




# activations_abs_mean  ..._0-1_mean_2-3:
# activations_mean  ..._0_mean_2-3_std_1:
#   clear seperation of epochs in segout layer for Adam
#   less clear but still visible for SGD

# activations_abs_mean  ..._1_mean_2-3_std_0 (..._2-3_std_0_mean_1):
#   SGD segout layer seems higher for better sdice
#   not visible for Adam

# activations_positiv   ..._std_1_std_2-3_mean_0:
#   segout layer smaller 0.5 seem to be good
# activations_positiv   ..._std_2-3_mean_0-1:
#   segout layer DA seems to reduce max value
# activations_positiv   ...mean_1_std_2-3:
#   segout layer Adam lower

# activations_std_1_mean_2-3_mean_0:
#   std higher for higher epochs
#   std higher in higher layers for Adam

# activations_abs mean:
#   segout activation mean higher for more epochs and especially higher for Adam

# activations_positiv_mean
# activations_positiv_mean_1_std_2-3
# activations_positiv_std_1_mean_2-3
# activations_positiv_std_1_std_2-3
#   segout to much deviation from same_fold seems to result lower sdice score