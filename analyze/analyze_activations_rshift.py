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
        'nnUNetTrainerV2_MA_noscheduler_depth5_SGD_ep120__nnUNetPlansv2.1',
        'nnUNetTrainerV2_MA_noscheduler_depth5_SGD_ep120_noDA__nnUNetPlansv2.1',
        'nnUNetTrainerV2_MA_noscheduler_depth5_ep120__nnUNetPlansv2.1',
        'nnUNetTrainerV2_MA_noscheduler_depth5_ep120_noDA__nnUNetPlansv2.1',
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

    for trainer, fold_train, epoch in itertools.product(trainers, folds, epochs):
        id_path_map = get_activations_id_path_map(
            os.path.join(
                hlp.get_testdata_dir(task, trainer, fold_train, epoch),
                'activations'
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
            scores_test = scores_test.sort_values('is_validation').head(12)

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
            stats.to_csv('data/csv/activations-rshift-{}-{}-{}-{}.csv'.format(trainer, fold_train, epoch, fold_test))



def plot_rshift():
    stats = pd.concat([
        pd.read_csv(path) for path in glob.iglob('data/csv/activations-rshift-*.csv', recursive=False)
    ])
    stats['trainer_short'] = stats['trainer'].apply(hlp.get_trainer_short)

    index = ['trainer', 'fold_train', 'epoch', 'fold_test']
    domainshift_scores = pd.concat([
        hlp.load_domainshift_scores(
            'Task601_cc359_all_training',
            *ids
        ) for ids in stats[index].drop_duplicates().values.tolist()
    ])

    stats = stats.join(domainshift_scores.set_index(index), on=index)


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

    stats_grouped = stats.groupby([
        'trainer_short', 'fold_train', 'epoch', 'fold_test', 'layer_pos'
    ]).agg([
        'mean',# 'std', 'sum', 'min', 'max'
    ]).reset_index()

    stats_grouped['psize'] = stats_grouped['epoch'] / 10
    stats_grouped['same_fold'] = pd.factorize(
        stats_grouped['fold_train'] == stats_grouped['fold_test'],
        sort=True
    )[0]

    column_color = 'same_fold'
    #column_color = ('sdice_score_other_mean', 'mean')
    stats_grouped = stats_grouped[stats_grouped['epoch'] == 40]

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

        fig.savefig('data/fig/activations-{}.png'.format(column))
        plt.close(fig)



#calc_rshift()

plot_rshift()