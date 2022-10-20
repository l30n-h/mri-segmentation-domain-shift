import numpy as np
import torch
import pandas as pd
import itertools
import glob
import os
import helpers as hlp

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
    SCANS_PER_FOLD=6
    BATCHES_PER_SCAN=64
    LAYERS = hlp.get_layers_ordered()

    def get_ids(scores, fold_test, is_validation):
        return scores[
            (scores['fold_test'] == fold_test) & (scores['is_validation'] == is_validation)
        ].sort_values('id').head(SCANS_PER_FOLD)['id_long']
    
    def generate_rshift_stats_per_model(ids_train, folds_ids_dict_test, load_activations_means):
        print('train', fold_train)
        activations_ref_per_layer = load_activations_means(ids_train)

        for (fold_test, is_validation), ids_test in folds_ids_dict_test.items():
            print('evaluate', fold_test, is_validation)
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
        '{}spf-{}bps-testaug'.format(
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
        scores = scores[(scores['fold_test'] == fold_train) | (~scores['is_validation'])]
        ids_train = get_ids(scores, fold_train, False)
        folds_ids_dict_test = dict(map(
            lambda key: (tuple(key), get_ids(scores, *key)),
            scores[['fold_test', 'is_validation']].drop_duplicates().values.tolist()
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



def plot_rshift(stats_glob, output_dir):
    stats = pd.concat([
        pd.read_csv(path) for path in glob.iglob(stats_glob, recursive=False)
    ]).reset_index(drop=True)
    stats.rename(
        columns={
            'name': 'layer'
        },
        inplace=True
    )
    layers_position_map = hlp.get_layers_position_map()
    layers_position_map['input'] = -2
    layers_position_map['gt'] = -1
    stats['layer_pos'] = stats['layer'].apply(lambda d: layers_position_map.get(d))

    index = ['trainer', 'fold_train', 'epoch']
    scores = pd.concat([
        hlp.get_scores(
            'Task601_cc359_all_training',
            *ids
        ) for ids in stats[index].drop_duplicates().values.tolist()
    ]).reset_index(drop=True)
    index = [*index, 'fold_test', 'is_validation']
    scores = scores.groupby(index).agg(
        fold_test_base=('fold_test_base', 'first'),
        optimizer=('optimizer', 'first'),
        wd=('wd', 'first'),
        DA=('DA', 'first'),
        bn=('bn', 'first'),
        test_augmentation=('test_augmentation', 'first'),
        dice_score_mean=('dice_score', 'mean'),
        dice_score_std=('dice_score', 'std'),
        iou_score_mean=('iou_score', 'mean'),
        iou_score_std=('iou_score', 'std'),
        sdice_score_mean=('sdice_score', 'mean'),
        sdice_score_std=('sdice_score', 'std')
    ).reset_index()
    scores['same_domain'] = scores['fold_train'] == scores['fold_test']
    scores['same_domain_base'] = scores['fold_train'] == scores['fold_test_base']
    scores['domain_val'] = scores['same_domain'].apply(lambda x: 'same' if x else 'other') + scores['is_validation'].apply(lambda x: ' validation' if x else '')
    scores['domain_val_base'] = scores['same_domain_base'].apply(lambda x: 'same' if x else 'other') + scores['is_validation'].apply(lambda x: ' validation' if x else '')
    scores['domain_val_testaug'] = scores['domain_val'] + scores['test_augmentation'].apply(lambda x: '' if x == 'None' else ' testaug')
    scores['trainer_short'] = scores['trainer'].apply(hlp.get_trainer_short)
    scores['has_testaug'] = scores['test_augmentation'] != 'None'
    stats = stats.join(scores.set_index(index), on=index)
    stats['is_validation'] = stats['is_validation'].astype(bool)


    stats = stats[~stats['wd']]
    stats = stats[stats['DA'].isin(['none', 'full'])]
    stats = stats[stats['same_domain'] | (~stats['is_validation'])]


    stats['Data Aug.'] = stats['DA'].str.replace(r'^no([^n].+)$', r'no-\1', n=1, regex=True)
    stats['Domain'] = stats['domain_val_testaug'].str.replace('same validation', 'Validation').str.replace('same', 'Training').str.replace('other testaug', 'Other w/ test aug.').str.replace('other', 'Other w/o test aug.')
    stats['Base Domain'] = stats['domain_val_base'].str.replace('same validation', 'Validation').str.replace('same', 'Training').str.replace('other', 'Other') + stats['test_augmentation'].apply(lambda x: '' if x == 'None' else ' w/ test aug.')
    stats['Optimizer'] = stats['optimizer']
    stats['DSC'] = stats['dice_score_mean']
    stats['Surface DSC'] = stats['sdice_score_mean']
    stats['IoU'] = stats['iou_score_mean']
    stats['Epoch'] = stats['epoch']
    stats['Layer'] = stats['layer']
    stats['# Layer'] = stats['layer_pos']
    stats['Normalization'] = stats['bn'].apply(lambda x: 'Batch' if x else 'Instance')
    stats['Representation shift'] = stats['rshift']

    # TODO
    stats = stats[(stats['Normalization'] == 'Batch') | (~stats['Layer'].str.endswith('.instnorm'))]

    stats = stats[(~stats['Layer'].str.endswith('.instnorm'))]
    stats = stats[stats['# Layer'] >= 0]
    #

    stats = stats[stats['Surface DSC'] > 0.03]
    output_dir = output_dir + '-gt_0_03'

    columns_measurements = ['Representation shift']

    stats_meaned_over_layer = stats.groupby(['trainer_short', 'fold_train', 'Epoch', 'fold_test', 'is_validation']).agg(**{
        'IoU': ('IoU', 'first'),
        'DSC': ('DSC', 'first'),
        'Surface DSC': ('Surface DSC', 'first'),
        'Optimizer': ('Optimizer', 'first'),
        'Data Aug.': ('Data Aug.', 'first'),
        'Normalization': ('Normalization', 'first'),
        'Domain': ('Domain', 'first'),
        'Base Domain': ('Base Domain', 'first'),
        'domain_val_base': ('domain_val_base', 'first'),
        'has_testaug': ('has_testaug', 'first'),
        **{
            column: (column, 'mean') for column in columns_measurements
        }
    }).reset_index()

    print(
        stats.groupby(['trainer_short', 'fold_train', 'Epoch', 'is_validation'])['fold_test'].agg(['count', 'nunique']).to_string()
    )

    for groupby in [
        [],
        ['Optimizer',],
        ['Normalization'],
        ['Data Aug.'],
        ['Optimizer', 'Normalization'],
        ['Optimizer', 'Normalization', 'Data Aug.'],

        ['fold_train'],
        ['Optimizer', 'fold_train'],
        ['Normalization', 'fold_train'],
        ['Optimizer', 'Normalization', 'fold_train'],

        ['Domain'],
        ['Domain', 'Optimizer', 'Normalization'],

        ['Optimizer', 'Normalization', 'has_testaug']
    ]:
        print(groupby)
        print(
            hlp.get_corr_stats(
                stats=stats_meaned_over_layer,
                groupby=groupby,
                columns=['Representation shift', 'Surface DSC', 'Epoch']
            ).to_string()
        )
    
    print(stats_meaned_over_layer.groupby(['Domain', 'Optimizer', 'Normalization'])['Surface DSC'].agg(['mean', 'std', 'count']).to_string())

    print(stats)
    os.makedirs(output_dir, exist_ok=True)
    add_async_task, join_async_tasks = hlp.get_async_queue(num_threads=4)

    for measurement, score in itertools.product(columns_measurements, ['DSC', 'Surface DSC']):
        print(stats)
        hlp.plot_scattered_and_layered(add_async_task, stats, stats_meaned_over_layer, measurement, output_dir, col='Normalization', row='Optimizer', hue='Base Domain', style='Data Aug.', score=score, yscale='log', share_measurement='col')
        hlp.plot_scattered_and_layered(add_async_task, stats, stats_meaned_over_layer, measurement, output_dir, col='Normalization', row='Optimizer', hue='fold_train', style='Data Aug.', score=score, share_measurement='col')
        hlp.plot_scattered_and_layered(add_async_task, stats, stats_meaned_over_layer, measurement, output_dir, col='Normalization', row='Optimizer', hue='Data Aug.', style=None, score=score, yscale='log', share_measurement='col')
        hlp.plot_scattered_and_layered(add_async_task, stats, stats_meaned_over_layer, measurement, output_dir, col='Normalization', row='Optimizer', hue='has_testaug', style='Data Aug.', score=score, yscale='log', share_measurement='col')


        col='Normalization'
        row='Optimizer'
        hue='fold_train'
        style='Data Aug.'
        score=score

        stats_corr = hlp.get_corr_stats(
            stats,
            groupby=['# Layer', col, row, hue, style],
            columns=[measurement, score],
            column_combinations=None
        ).reset_index()
        measurement_pval = '{}-{}_pval'.format(measurement, score)
        measurement_corr = '{}-{}_corr'.format(measurement, score)
        stats_corr.loc[stats_corr[measurement_pval] >= 0.01, measurement] = np.nan
        stats_corr[measurement] = stats_corr[measurement_corr]
        suffix = '{}-{}-{}-{}'.format(
            'single' if col is None else col,
            'single' if row is None else row,
            'single' if hue is None else hue,
            'single' if style is None else style,
        )
        add_async_task(
            hlp.relplot_and_save,
            outpath=os.path.join(
                output_dir,
                '{}-{}-layered-{}.png'.format(measurement, score, suffix)
            ),
            data=stats_corr,
            kind='line',
            x='# Layer',
            y=measurement,
            row=row,
            row_order=None if row is None else stats_corr[row].sort_values().unique(),
            col=col,
            col_order=None if col is None else stats_corr[col].sort_values().unique(),
            style=style if style in stats_corr else None,
            hue=hue,
            palette='cool',
            aspect=2,
            height=6,
        )
    join_async_tasks()



task = 'Task601_cc359_all_training'
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
    #'philips15',
    #'philips3'
]
epochs = [10,20,30,40,80,120]

#calc_rshift(task, trainers, folds, epochs, output_dir='data/csv/activations-rshift')

plot_rshift(
    'data/csv/activations-rshift/6spf-64bps-testaug/*.csv',
    'data/fig/activations-rshift/6spf-64bps-testaug'
)