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
def get_representation_shift(act_ref, act_test):
    return torch.tensor([
        wasserstein_distance(act_ref[:, k].numpy(), act_test[:, k].numpy())
        for k in range(act_ref.shape[1])
    ]).mean()


def get_rshift_stats(act_ref, act_test):
    act_ref = act_ref[0 : act_test.shape[0]]
    diff = act_ref - act_test
    rshift = get_representation_shift(act_ref, act_test)
    return {
        'rshift': rshift.round(decimals=4).item(),
        'ref_mean': act_ref.mean().item(),
        'ref_std': act_ref.std().item(),
        'test_mean': act_test.mean().item(),
        'test_std': act_test.std().item(),
        'diff_mean': diff.mean().item(),
        'diff_std': diff.std().item(),
        'diff_mean_abs': diff.abs().mean().item(),
        'diff_std_abs': diff.abs().std().item()
    }

def load_data_live(task, trainer, fold_train, epoch, layers, dataset_keys, batches_per_scan=64):
    layers_set = set(layers)
    def is_module_tracked(name, module):
        return name in layers_set

    def extract_activations(name, activations, slices_and_tiles_count, activations_dict):
        activations_dict.setdefault(name, []).append(
            activations[0].float().mean(dim=(1,2))
        )
        return activations_dict

    def merge_activations_dict(activations_dict):
        return dict(map(
            lambda item: (
                item[0],
                torch.stack(item[1]).cpu()
            ),
            activations_dict.items()
        ))

    
    tmodel = hlp.load_model(trainer, fold_train, epoch)
    hlp.apply_prediction_data_filter_monkey_patch(tmodel, batches_per_scan=batches_per_scan)

    out = hlp.get_activations_dicts_merged([ activations_dict for id, prediction, activations_dict in hlp.generate_predictions_ram(
        trainer=tmodel,
        dataset_keys=dataset_keys,
        activations_extractor_kwargs=dict(
            extract_activations=extract_activations,
            is_module_tracked=is_module_tracked,
            merge_activations_dict=merge_activations_dict
        )
    )])
    return out

def calc_rshift(task, trainers, folds, epochs, output_dir):
    SCANS_PER_FOLD=24
    BATCHES_PER_SCAN=64
    LAYERS = hlp.get_layers_ordered()

    def get_ids(scores, fold_test, is_validation):
        return scores[
            (scores['fold_test'] == fold_test) & (scores['is_validation'] == is_validation)
        ].sort_values('id').head(SCANS_PER_FOLD)['id_long']
    
    def generate_rshift_stats_per_model(ids_train, folds_ids_dict_test, load_activations_means):
        activations_ref_per_layer = load_activations_means(ids_train)

        for (fold_test, is_validation), ids_test in folds_ids_dict_test.items():
            activations_test_per_layer = load_activations_means(ids_test)

            for layer, act_test in activations_test_per_layer.items():
                yield {
                    'fold_test': fold_test,
                    'is_validation': is_validation,
                    'name': layer,
                    **get_rshift_stats(
                        activations_ref_per_layer[layer],
                        activations_test_per_layer[layer]
                    )
                }

    output_dir = os.path.join(
        output_dir,
        '{}spf-{}bps'.format(
            SCANS_PER_FOLD,
            BATCHES_PER_SCAN,
        )
    )
    os.makedirs(output_dir, exist_ok=True)

    for trainer, fold_train, epoch in itertools.product(trainers, folds, epochs):
        scores = hlp.get_scores(
            task,
            trainer,
            fold_train,
            epoch
        )
        ids_train = get_ids(scores, fold_train, False)
        folds_ids_dict_test = dict(map(
            lambda key: (key, get_ids(scores, *key)),
            itertools.product(
                scores['fold_test'].unique(),
                scores['is_validation'].unique()
            )
        ))

        stats = pd.DataFrame(generate_rshift_stats_per_model(
            ids_train,
            folds_ids_dict_test,
            lambda keys: load_data_live(
                task,
                trainer,
                fold_train,
                epoch,
                LAYERS,
                keys,
                batches_per_scan=BATCHES_PER_SCAN
            )
        ))
        stats['task'] = task
        stats['trainer'] = trainer
        stats['fold_train'] = fold_train
        stats['epoch'] = epoch
        print(stats)
        stats.to_csv(
            os.path.join(
                output_dir,
                'activations-rshift-{}-{}-{}.csv'.format(
                    trainer,
                    fold_train,
                    epoch
                )
            )
        )



def plot_rshift(stats_path):
    stats = pd.concat([
        pd.read_csv(path) for path in glob.iglob(os.path.join(stats_path, '*'), recursive=False)
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

    layers_position_map = hlp.get_layers_position_map()
    layers_position_map['input'] = -2
    layers_position_map['gt'] = -1
    stats.rename(
        columns={
            'name': 'layer'
        },
        inplace=True
    )
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

    output_dir = os.path.join(
        'data/fig/activations-rshift',
        os.path.basename(stats_path)
    )
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
    #'philips15',
    #'philips3'
]
epochs = [10,20,30,40,80,120] #[40]

#calc_rshift(task, trainers, folds, epochs, output_dir='data/csv/activations-rshift')

plot_rshift('data/csv/activations-rshift/24spf-64bps')


#TODO instnorm
#  batchnorm makes use of train mean and std
#  instancenorm shifts each slice to 0-mean 1-std (rsp beta-mean gamma-std)
#  rshift paper uses resnet18/inception-v3/googlenet with bn and tracking stats
#  btw unet2d used in lab also uses batchnorm