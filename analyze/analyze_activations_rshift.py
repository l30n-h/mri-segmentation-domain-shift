import numpy as np
import torch
import pandas as pd
import itertools
import glob
import os
import matplotlib.pyplot as plt
import helpers as hlp

def get_kullback_liebler_divergence(P, Q):
    return torch.sum(P * torch.log(P / (Q + 1e-3)))

from scipy.stats import wasserstein_distance
def get_wasserstein_distance(P, Q):
    return torch.tensor([
        wasserstein_distance(P[:, k], Q[:, k])
        for k in range(P.shape[1])
    ]).mean()

def generate_stats_compare(parameters_a, parameters_b):
    for name in parameters_a.keys():
        features_a = parameters_a[name].to(torch.float32)
        features_b = parameters_b[name].to(torch.float32)

        if len(features_a.shape) == 0:
            continue
        if len(features_a.shape) == 1:
            features_a = torch.unsqueeze(features_a, 1)
            features_b = torch.unsqueeze(features_b, 1)
            #continue

        diff = features_a - features_b

        rshift = get_wasserstein_distance(features_a.numpy(), features_b.numpy())

        yield {
            'name': name,
            'rshift': rshift.round(decimals=4).item(),
            'a_mean': features_a.mean().item(),
            'a_std': features_a.std().item(),
            'b_mean': features_b.mean().item(),
            'b_std': features_b.std().item(),
            'diff_mean': diff.mean().item(),
            'diff_std': diff.std().item(),
            'diff_mean_abs': diff.abs().mean().item(),
            'diff_std_abs': diff.abs().std().item()
        }

def get_activations_dict(activations_paths, num_slices=None):
    def get_slices_subset(activations_dict):
        indices = torch.randint(
            low=0,
            high=next(iter(activations_dict.values())).shape[0],
            size=(num_slices,)
        )
        return dict(map(
            lambda item: (
                item[0],
                torch.index_select(
                    item[1],
                    0,
                    indices
                )
            ),
            activations_dict.items()
        ))

    def get_slices_mean(activations_dict):
        return dict(
            map(
                lambda x: (x[0], x[1].mean(dim=0, keepdim=True)),
                activations_dict.items()
            )
        )
    
    activations_dict_concat = dict()
    for path in activations_paths:
        activations_dict = torch.load(
            path,
            map_location=torch.device('cpu')
        )
        # TODO for each layer if not flattened and meaned than .flatten(2).mean(dim=2)
        activations_dict = get_slices_mean(
            activations_dict
        ) if num_slices is None else get_slices_subset(
            activations_dict
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

def get_activations_id_path_map(directory_activations):
    activation_paths = [ path for path in glob.iglob(
        os.path.join(
            directory_activations,
            '*_activations.pkl'
        ),
        recursive=False
    ) ]

    return dict(map(
        lambda path: (
            hlp.get_id_from_filename(os.path.basename(path)),
            path
        ),
        activation_paths
    ))


def calc_rshift():
    task = 'Task601_cc359_all_training'
    trainers = [
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
        'philips15',
        'philips3'
    ]
    epochs = [10,20,30,40,80,120] #[40]

    output_dir = 'data/csv/activations-rshift'
    os.makedirs(output_dir, exist_ok=True)

    for trainer, fold_train, epoch in itertools.product(trainers, folds, epochs):
        id_path_map = get_activations_id_path_map(
            os.path.join(
                hlp.get_testdata_dir(task, trainer, fold_train, epoch),
                'activations-mean'
            )
        )
        
        scores = hlp.get_scores(
            task,
            trainer,
            fold_train,
            epoch
        )

        scores = scores[scores['id'].isin(set(id_path_map.keys()))]
        scores['activations_path'] = scores['id'].apply(lambda x: id_path_map.get(x))

        scores_train = scores[scores['fold_test'] == fold_train]
        scores_train = scores_train[~scores_train['is_validation']]
        #TODO more subsets could be used
        activation_paths_train = scores_train.nlargest(
            12,
            'sdice_score'
        )['activations_path'].tolist()
        
        
        num_slices = 64
        activations_ref = get_activations_dict(
            activation_paths_train,
            num_slices
        )

        for fold_test in folds:
            print(trainer, fold_train, fold_test, epoch)

            scores_test = scores[scores['fold_test'] == fold_test]
            #TODO more subsets could be used but already overlaps with train but random slices used!!
            scores_test = scores_test.sort_values('is_validation', ascending=False).head(12)

            activation_paths_test = scores_test['activations_path'].tolist()

            activations = get_activations_dict(
                activation_paths_test,
                num_slices
            )
            stats = pd.DataFrame(generate_stats_compare(
                activations_ref,
                activations
            ))
            stats['trainer'] = trainer
            stats['fold_train'] = fold_train
            stats['epoch'] = epoch
            stats['fold_test'] = fold_test
            print(stats)
            stats.to_csv(
                os.path.join(
                    output_dir,
                    'activations-rshift-{}-{}-{}-{}.csv'.format(trainer, fold_train, epoch, fold_test)
                )
            )



def plot_rshift():
    stats = pd.concat([
        pd.read_csv(path) for path in glob.iglob('data/csv/activations-rshift/activations-rshift-*.csv', recursive=False)
    ])
    stats['trainer_short'] = stats['trainer'].apply(hlp.get_trainer_short)

    index = ['trainer', 'fold_train', 'epoch']
    scores = pd.concat([
        hlp.get_scores(
            'Task601_cc359_all_training',
            *ids
        ) for ids in stats[index].drop_duplicates().values.tolist()
    ])
    index = [*index, 'fold_test']
    scores = scores.groupby(index).agg(
        iou_score_mean=('iou_score', 'mean'),
        dice_score_mean=('dice_score', 'mean'),
        sdice_score_mean=('sdice_score', 'mean'),
    ).reset_index()
    stats = stats.join(scores.set_index(index), on=index)

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
    stats['same_fold'] = (stats['fold_train'] == stats['fold_test']).astype(int)


    columns = [
        'rshift',
        'a_mean',
        'a_std',
        'b_mean',
        'b_std',
        'diff_mean',
        'diff_std',
        'diff_mean_abs',
        'diff_std_abs',
    ]

    stats = hlp.numerate_nested(stats, [
        'epoch',
        'trainer_short',
        'fold_train',
        'same_fold',
        'fold_test',
    ])

    stats['psize'] = stats['epoch'] / 10 + stats['same_fold'] * 5

    column_color = ('sdice_score_mean')
    
    #stats = stats[stats['same_fold'] == 0]
    stats = stats[stats['epoch'] >= 40]
    #stats = stats[stats['epoch'] == 40]

    output_dir = 'data/fig/activations-rshift'
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
            fig_num='activations_rshift',
            fig_clear=True
        )
        fig.savefig(
            os.path.join(
                output_dir,
                'activations-rshift-{}.png'.format(column)
            ),
            bbox_inches='tight',
            pad_inches=0
        )
        plt.close(fig)



#calc_rshift()

plot_rshift()