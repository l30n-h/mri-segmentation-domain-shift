import numpy as np
import torch
import pandas as pd
import itertools
import glob
import os
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
    ]).reset_index(drop=True)
    index = ['trainer', 'fold_train', 'epoch']
    scores = pd.concat([
        hlp.get_scores(
            'Task601_cc359_all_training',
            *ids
        ) for ids in stats[index].drop_duplicates().values.tolist()
    ]).reset_index(drop=True)
    index = [*index, 'fold_test', 'is_validation']
    scores = scores.groupby(index).agg(
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
    stats = stats.join(scores.set_index(index), on=index)

    layers_position_map = hlp.get_layers_position_map()
    layers_position_map['input'] = -2
    layers_position_map['gt'] = -1
    stats.rename(columns={ 'name': 'layer' }, inplace=True)
    stats['layer_pos'] = stats['layer'].apply(lambda d: layers_position_map.get(d))
    stats['wd_bn'] = stats['wd'].apply(lambda x: 'wd=' + str(x)) + ' ' + stats['bn'].apply(lambda x: 'bn=' + str(x))
    stats['optimizer_wd_bn'] = stats['optimizer'] + ' ' + stats['wd_bn']
    stats['wd_bn_DA'] = stats['wd_bn'] + ' ' + stats['DA']
    stats['same_domain'] = stats['fold_train'] == stats['fold_test']
    stats['domain_val'] = stats['same_domain'].apply(lambda x: 'same' if x else 'other') + ' ' + stats['is_validation'].apply(lambda x: 'validation' if x else '')
    stats['trainer_short'] = stats['trainer'].apply(hlp.get_trainer_short)
    stats['trainer_short_epoch'] = stats['trainer_short'] + " " + stats['epoch'].apply(lambda x: str(x).rjust(3, '0'))

    print(stats)
    print(stats.head(20).to_string())

    output_dir = 'data/fig/activations-rshift'
    os.makedirs(output_dir, exist_ok=True)

    add_async_task, join_async_tasks = hlp.get_async_queue(num_threads=12)

    columns = ['dice_score']
    aggs = ['mean']
    #stats = stats[stats['fold_train'] == 'siemens15']
    #stats = stats[stats['epoch'] == 120]
    #stats = stats[stats['layer_pos'] <= 39]
    #stats = stats[stats['fold_test'] == 'siemens3']
    stats = stats[stats['DA'].isin(['none', 'full'])]
    print(stats)
    # print(stats[[
    #     'task',
    #     'trainer',
    #     'fold_train',
    #     'epoch',
    #     'fold_test',
    #     'is_validation',
    #     'layer',
    #     'optimizer',
    #     'wd',
    #     'DA',
    #     'bn'
    # ]].to_string())
    # for column, agg in itertools.product(columns, aggs):
    #     add_async_task(
    #         hlp.relplot_and_save,
    #         outpath=os.path.join(output_dir, 'rshift-layer_pos-fold_test-{}-{}.png'.format(column, agg)),
    #         data=stats,
    #         kind='scatter',
    #         x='rshift',
    #         y='{}_{}'.format(column, agg),
    #         # col='trainer_short',
    #         # col_order=stats['trainer_short'].sort_values().unique(),
    #         col='optimizer_wd_bn',
    #         col_order=stats['optimizer_wd_bn'].sort_values().unique(),
    #         row='layer_pos',
    #         row_order=stats['layer_pos'].sort_values().unique(),
    #         #hue='fold_train',
    #         hue='DA',
    #         #hue='same_domain',
    #         style='domain_val',
    #         size='epoch',
    #         height=3,
    #         aspect=2,
    #         ci=None,
    #         facet_kws=dict(
    #             #sharex='col'
    #             sharex=False
    #         )
    #     )
    #stats = stats[stats['fold_train'] == 'siemens15']
    stats = stats[stats['epoch'].isin([20, 40, 120])]
    #stats = stats[stats['layer_pos'] <= 39]
    #stats = stats[stats['layer_pos'] > 4]
    #stats = stats[stats['fold_test'] == 'siemens3']
    stats = stats[stats['DA'].isin(['none', 'full'])]
    stats = stats[stats['domain_val'] != 'other validation']
    print(stats)
    # add_async_task(
    #     hlp.relplot_and_save,
    #     outpath=os.path.join(output_dir, 'rshift-layer_pos-epoch-rshift.png'),
    #     data=stats,
    #     kind='line',
    #     x='epoch',
    #     y='rshift',
    #     col='trainer_short',
    #     col_order=stats['trainer_short'].sort_values().unique(),
    #     row='layer_pos',
    #     row_order=stats['layer_pos'].sort_values().unique(),
    #     hue='fold_test',
    #     style='domain_val',
    #     size='fold_train',
    #     #units='',
    #     height=3,
    #     aspect=2,
    #     ci=None,
    #     # facet_kws=dict(
    #     #     sharex='col'
    #     # )
    # )

    # add_async_task(
    #     hlp.relplot_and_save,
    #     outpath=os.path.join(output_dir, 'rshift-epoch-layer_pos-rshift.png'),
    #     data=stats,
    #     kind='line',
    #     x='layer_pos',
    #     y='rshift',
    #     col='optimizer_wd_bn',
    #     col_order=stats['optimizer_wd_bn'].sort_values().unique(),
    #     row='epoch',
    #     row_order=stats['epoch'].sort_values().unique(),
    #     hue='dice_score_mean',
    #     palette='cool',
    #     style='DA',
    #     #style='domain_val',
    #     #size='fold_train',
    #     #units='fold_test',
    #     height=6,
    #     aspect=2,
    #     estimator=None,
    #     ci=None,
    #     facet_kws=dict(
    #         sharey='none'
    #     )
    # )
    #stats['rshift_sqrt'] = stats['rshift'] ** (0.5)
    for bn in stats['bn'].sort_values().unique():
        stats2 = stats[stats['bn'] == bn]
        if not bn:
            stats2 = stats2[~stats2['layer'].str.endswith('.instnorm')]
        add_async_task(
            hlp.relplot_and_save,
            outpath=os.path.join(output_dir, 'rshift-dice_score_mean-layer_pos-rshift-bn={}.png'.format(bn)),
            data=stats2,
            kind='line',
            x='layer_pos',
            y='rshift',
            yscale='log',
            col='trainer_short',
            col_order=stats2['trainer_short'].sort_values().unique(),
            row='fold_train',
            row_order=stats2['fold_train'].sort_values().unique(),
            hue='dice_score_mean',
            #hue='epoch',
            palette='cool',
            style='fold_test',
            size='epoch',
            #units='fold_test',
            height=8,
            aspect=4,
            estimator=None,
            ci=None,
            # facet_kws=dict(
            #     sharey='none'
            # )
        )
    
    join_async_tasks()



task = 'Task601_cc359_all_training'
trainers = [
    # 'nnUNetTrainerV2_MA_noscheduler_depth5_SGD_ep120__nnUNetPlansv2.1',
    # 'nnUNetTrainerV2_MA_noscheduler_depth5_SGD_ep120_noDA__nnUNetPlansv2.1',
    # 'nnUNetTrainerV2_MA_noscheduler_depth5_ep120__nnUNetPlansv2.1',
    # 'nnUNetTrainerV2_MA_noscheduler_depth5_ep120_noDA__nnUNetPlansv2.1',

    # 'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120__nnUNetPlansv2.1',
    # 'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120_noDA__nnUNetPlansv2.1',
    # 'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120__nnUNetPlansv2.1',
    # 'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120_noDA__nnUNetPlansv2.1',

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
epochs = [10,20,30,40,80,120] #[40]

#calc_rshift(task, trainers, folds, epochs, output_dir='data/csv/activations-rshift')

plot_rshift('data/csv/activations-rshift/24spf-64bps')


#TODO instnorm
#  batchnorm makes use of train mean and std
#  instancenorm shifts each slice to 0-mean 1-std (rsp beta-mean gamma-std)
#  rshift paper uses resnet18/inception-v3/googlenet with bn and tracking stats
#  btw unet2d used in lab also uses batchnorm