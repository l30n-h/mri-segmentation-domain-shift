import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import glob
import os
import helpers as hlp
import json
import os

def store_activation_maps(task, trainers, fold_trains, epochs, output_directory_base='data/fig/activation_maps/'):
    batches_per_scan=24
    layers = hlp.get_layers_ordered()
    dataset_keys = [
        "CC0001_philips_15_55_M",
        "CC0009_philips_15_69_M",
        "CC0061_philips_3_55_F",
        "CC0069_philips_3_56_F",
        "CC0121_siemens_15_61_M",
        "CC0129_siemens_15_54_F",
        "CC0181_siemens_3_36_F",
        "CC0189_siemens_3_66_F",
        "CC0241_ge_15_60_F",
        "CC0249_ge_15_58_F",
        "CC0301_ge_3_53_F",
        "CC0309_ge_3_42_M",
    ]

    layers_set = set(layers)
    def is_module_tracked(name, module):
        return name in layers_set

    def extract_activations_full(name, activations, slices_and_tiles_count, activations_dict):
        activations_dict.setdefault(name, []).append(activations[0].clone())
        return activations_dict

    def merge_activations_dict(activations_dict):
        return dict(map(
            lambda item: (
                item[0],
                torch.stack(item[1]).cpu()
            ),
            activations_dict.items()
        ))

    for trainer, fold_train, epoch in itertools.product(trainers, fold_trains, epochs):
    
        tmodel = hlp.load_model(trainer, fold_train, epoch)
        hlp.apply_prediction_data_filter_monkey_patch(tmodel, batches_per_scan=batches_per_scan)

        out = hlp.get_activations_dicts_merged([ activations_dict for id, prediction, activations_dict in hlp.generate_predictions_ram(
            trainer=tmodel,
            dataset_keys=dataset_keys,
            activations_extractor_kwargs=dict(
                extract_activations=extract_activations_full,
                is_module_tracked=is_module_tracked,
                merge_activations_dict=merge_activations_dict
            )
        )])

        layers_position_map = hlp.get_layers_position_map()
        layers_position_map['input'] = -2
        layers_position_map['gt'] = -1

        for name, activation_maps_merged in out.items():
            layer_id = layers_position_map.get(name)
            activation_maps_merged = activation_maps_merged.to(dtype=torch.float32)

            for i, key in enumerate(dataset_keys):
                activation_maps = activation_maps_merged[i*batches_per_scan:(i+1)*batches_per_scan]
                #activation_maps = activation_maps
                #activation_maps = activation_maps / 
                scale = 1.0
                #scale = torch.nan_to_num(1.0 / activation_maps.abs().flatten(1).max(dim=1)[0][:,None,None, None], 1.0)
                #scale = torch.nan_to_num(1.0 / activation_maps.abs().flatten(2).max(dim=2)[0][:,:,None, None], 1.0)
                
                activation_maps = activation_maps * scale
                print(key, name, activation_maps.shape, activation_maps.min(), activation_maps.max(), activation_maps.mean(), activation_maps.std())
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

                stats_name = 'extraction-test'
                output_dir = os.path.join(
                    output_directory_base,
                    '{}-{}'.format(stats_name, trainer)
                )
                os.makedirs(output_dir, exist_ok=True)
                plt.imsave(
                    os.path.join(output_dir, '{}-{}-{}-{}-f{}-l{}.jpg'.format(
                        stats_name, trainer, fold_train, str(epoch).rjust(3, '0'), key.split('_')[0], str(layer_id).rjust(3, '0')
                    )),
                    merged,
                    vmin=-mval,
                    vmax=mval
                )

def store_activations(data, path):
    print(path)
    output_dir = os.path.dirname(path)
    os.makedirs(output_dir, exist_ok=True)
    return torch.save(data, path)


def extract_featurewise_measurements(
    task,
    trainers,
    folds_train,
    epochs,
    output_directory_base,
    dataset_keys=None
):
    layers_set = set(hlp.get_layers_ordered())
    def is_module_tracked(name, module):
        return name in layers_set

    def extract_activations_full(name, activations, slices_and_tiles_count, activations_dict):
        activations = activations[0]
        activations_float32 = activations.float()
        activations_float32_abs = activations_float32.abs()

        activations_normed, activations_norm = hlp.get_normed(
            activations_float32.flatten(1),
            dim=1
        )
        similarity_matrix = hlp.get_cosine_similarity(
                activations_normed,
                activations_normed
        )
        diagonals = similarity_matrix.diagonal(dim1=0, dim2=1)
        coocurences = similarity_matrix.masked_select(
            torch.eye(similarity_matrix.shape[0], device=similarity_matrix.device) == 0
        ).reshape(similarity_matrix.shape[0], similarity_matrix.shape[0] - 1)
        if coocurences.shape[1] == 0:
            coocurences = torch.full((coocurences.shape[0], 1), float('nan'), device=coocurences.device)
        coocurences_abs = coocurences.abs()

        results = {
            "min": activations_float32.amin(dim=(1,2)),
            "max": activations_float32.amax(dim=(1,2)),
            "mean": activations_float32.mean(dim=(1,2)),
            "std": activations_float32.std(dim=(1,2)),
            "skewness": hlp.standardized_moment(activations_float32, 3, dim=(1,2)),
            "kurtosis": hlp.standardized_moment(activations_float32, 4, dim=(1,2)),
            "abs_min": activations_float32_abs.amin(dim=(1,2)),
            "abs_max": activations_float32_abs.amax(dim=(1,2)),
            "abs_mean": activations_float32_abs.mean(dim=(1,2)),
            "abs_std": activations_float32_abs.std(dim=(1,2)),
            "abs_skewness": hlp.standardized_moment(activations_float32_abs, 3, dim=(1,2)),
            "abs_kurtosis": hlp.standardized_moment(activations_float32_abs, 4, dim=(1,2)),
            
            "noise": hlp.get_noise_estimates(
                activations.reshape(activations.shape[0], 1, *activations.shape[1:])
            ).reshape((activations.shape[0])),

            "l2_norm": activations_norm[:,0],
            "similarity_dia": diagonals,
            "similarity_coo_min": coocurences.amin(dim=1),
            "similarity_coo_max": coocurences.amax(dim=1),
            "similarity_coo_mean": coocurences.mean(dim=1),
            "similarity_coo_std": coocurences.std(dim=1),
            "similarity_coo_skewness": hlp.standardized_moment(coocurences, 3, dim=1),
            "similarity_coo_kurtosis": hlp.standardized_moment(coocurences, 4, dim=1),
            "similarity_coo_abs_min": coocurences_abs.amin(dim=1),
            "similarity_coo_abs_max": coocurences_abs.amax(dim=1),
            "similarity_coo_abs_mean": coocurences_abs.mean(dim=1),
            "similarity_coo_abs_std": coocurences_abs.std(dim=1),
            "similarity_coo_abs_skewness": hlp.standardized_moment(coocurences_abs, 3, dim=1),
            "similarity_coo_abs_kurtosis": hlp.standardized_moment(coocurences_abs, 4, dim=1),
        }
        activations_dict.setdefault('data', {}).setdefault(name, []).append(
            torch.stack(tuple(results.values()),dim=0).half()
        )
        activations_dict.setdefault('measurement_names', tuple(results.keys()))
        return activations_dict

    def merge_activations_dict(activations_dict):
        activations_dict['data'] = dict(map(
            lambda item: (
                item[0],
                torch.stack(item[1]).cpu()
            ),
            activations_dict['data'].items()
        ))
        return activations_dict
    

    add_async_task, join_async_tasks = hlp.get_async_queue()

    for trainer, fold_train, epoch in itertools.product(trainers, folds_train, epochs):
        tmodel = hlp.load_model(trainer, fold_train, epoch)
        
        hlp.apply_prediction_data_filter_monkey_patch(tmodel, batches_per_scan=32)

        for id, prediction, activations_dict in hlp.generate_predictions_ram(
            trainer=tmodel,
            dataset_keys=dataset_keys,
            activations_extractor_kwargs=dict(
                extract_activations=extract_activations_full,
                is_module_tracked=is_module_tracked,
                merge_activations_dict=merge_activations_dict
            )
        ):
            print(id, activations_dict.keys(), activations_dict['data']['input'].shape)
            print(hlp.get_activations_dict_memsize_estimate(activations_dict['data']) / 1024 / 1024)
            add_async_task(
                store_activations,
                activations_dict,
                os.path.join(output_directory_base, task, trainer, fold_train, str(epoch).rjust(3, '0'), id),
            )
        join_async_tasks()


def load_data(task, trainer, fold_train, epoch, id_long):
    path = os.path.join(
        'archive/old/nnUNet-container/data/measurements',
        task,
        trainer,
        fold_train,
        str(epoch).rjust(3, '0'),
        '{}.pt'.format(id_long)
    )
    if not os.path.exists(path):
        return
    print(path)
    measurements = torch.load(path)
    measurement_id_map = dict(map(reversed, enumerate(measurements['measurement_names'])))
    for layer, data in measurements['data'].items():
        data_agg = {
            'min': data.amin(dim=2).mean(dim=0),
            'max': data.amax(dim=2).mean(dim=0),
            'mean': data.mean(dim=2).mean(dim=0),
            'std': data.std(dim=2).mean(dim=0),
        }
        yield {
            'trainer': trainer,
            'fold_train': fold_train,
            'epoch': epoch,
            'id_long': id_long,
            'layer': layer,
            **{ 
                '{}_{}'.format(name, agg): data_agg[agg][measurement_id_map[name]].item() for name, agg in itertools.product(
                    measurement_id_map.keys(),
                    data_agg.keys()
                )
            }
        }


def create_stats_scanwise(task, trainers, fold_trains, epochs):
    output_dir = 'data/csv/activations-misc'
    os.makedirs(output_dir, exist_ok=True)

    scores = pd.concat([
        hlp.get_scores(
            task, trainer, fold_train, epoch
        ) for trainer, fold_train, epoch in itertools.product(trainers, folds, epochs)
    ])
    
    stats = pd.DataFrame(itertools.chain.from_iterable([ 
        list(load_data(task, trainer, fold_train, epoch, id_long)) for trainer, fold_train, epoch, id_long in scores[['trainer', 'fold_train', 'epoch', 'id_long']].drop_duplicates().values.tolist()
    ])).join(scores.set_index(['trainer', 'fold_train', 'epoch', 'id_long']), on=['trainer', 'fold_train', 'epoch', 'id_long'])
    stats.to_csv(os.path.join(output_dir, 'scanwise.csv'))
    print(stats)



def plot():
    stats = pd.read_csv('data/csv/activations-misc/scanwise.csv')
    
    columns = [
        # 'min',
        # 'max',
        # 'mean',
        # 'std',
        # 'skewness',
        # 'kurtosis',
        # 'abs_min',
        # 'abs_max',
        # 'abs_mean',
        # 'abs_std',
        # 'abs_skewness',
        # 'abs_kurtosis',
        'noise',
        # 'l2_norm',
        # 'similarity_dia',
        # 'similarity_coo_min',
        # 'similarity_coo_max',
        # 'similarity_coo_mean',
        # 'similarity_coo_std',
        # 'similarity_coo_skewness',
        # 'similarity_coo_kurtosis',
        'similarity_coo_abs_min',
        'similarity_coo_abs_max',
        'similarity_coo_abs_mean',
        # 'similarity_coo_abs_std',
        # 'similarity_coo_abs_skewness',
        # 'similarity_coo_abs_kurtosis',
    ]

    print(stats)
    
    stats = stats.groupby(['trainer', 'fold_train', 'epoch', 'fold_test', 'is_validation', 'layer']).agg(
        optimizer=('optimizer', 'first'),
        wd=('wd', 'first'),
        DA=('DA', 'first'),
        #bn=('bn', 'first'),
        dice_score_mean=('dice_score', 'mean'),
        dice_score_std=('dice_score', 'std'),
        iou_score_mean=('iou_score', 'mean'),
        iou_score_std=('iou_score', 'std'),
        sdice_score_mean=('sdice_score', 'mean'),
        sdice_score_std=('sdice_score', 'std'),
        **{
            '{}_mean_mean'.format(column): ('{}_mean'.format(column), 'mean') for column in columns
        },
    ).reset_index()

    print(stats)

    layers_position_map = hlp.get_layers_position_map()
    layers_position_map['input'] = -2
    layers_position_map['gt'] = -1
    stats['layer_pos'] = stats['layer'].apply(lambda d: layers_position_map.get(d))
    stats['trainer_short'] = stats['trainer'].apply(hlp.get_trainer_short)
    stats['same_domain'] = stats['fold_train'] == stats['fold_test']
    stats['domain_val'] = stats['same_domain'].apply(lambda x: 'same' if x else 'other') + ' ' + stats['is_validation'].apply(lambda x: 'validation' if x else '')

    output_dir = 'data/fig/activations-misc'
    os.makedirs(output_dir, exist_ok=True)

    stats = stats[stats['epoch'].isin([10, 40, 120])]
    #stats = stats[stats['is_validation']]

    stats = stats[~stats['layer'].str.startswith('seg_outputs')]
    stats = stats[~stats['layer'].str.startswith('tu')]
    
    print(stats)
    stats_meaned_over_layer = stats.groupby(['trainer_short', 'fold_train', 'epoch', 'fold_test', 'domain_val']).agg(
        iou_score_mean=('iou_score_mean', 'first'),
        dice_score_mean=('dice_score_mean', 'first'),
        sdice_score_mean=('sdice_score_mean', 'first'),
        **{
            '{}_mean_mean'.format(column): ('{}_mean_mean'.format(column), 'mean') for column in columns
        },
    ).reset_index()

    add_async_task, join_async_tasks = hlp.get_async_queue(num_threads=12)
    for column in columns:
        add_async_task(
            hlp.relplot_and_save,
            outpath=os.path.join(output_dir, 'meaned-single-{}_mean_mean-dice_score.png'.format(column)),
            data=stats_meaned_over_layer,
            kind='scatter',
            x='{}_mean_mean'.format(column),
            y='dice_score_mean',
            hue='dice_score_mean',
            palette='cool',
            style='domain_val',
            size='epoch',
            aspect=2,
            height=6,
            #estimator=None,
            #ci=None,
        )
        add_async_task(
            hlp.relplot_and_save,
            outpath=os.path.join(output_dir, 'meaned-trainer-{}_mean_mean-dice_score.png'.format(column)),
            data=stats_meaned_over_layer,
            kind='scatter',
            x='{}_mean_mean'.format(column),
            y='dice_score_mean',
            col='fold_train',
            col_order=stats['fold_train'].sort_values().unique(),
            row='trainer_short',
            row_order=stats['trainer_short'].sort_values().unique(),
            hue='dice_score_mean',
            palette='cool',
            style='domain_val',
            size='epoch',
            aspect=2,
            height=6,
            #estimator=None,
            #ci=None,
        )
        add_async_task(
            hlp.relplot_and_save,
            outpath=os.path.join(output_dir, 'layered-single-{}_mean_mean-dice_score.png'.format(column)),
            data=stats,
            kind='line',
            x='layer_pos',
            y='{}_mean_mean'.format(column),
            hue='dice_score_mean',
            palette='cool',
            style='domain_val',
            size='epoch',
            aspect=2,
            height=6,
            #estimator=None,
            #ci=None,
        )
        add_async_task(
            hlp.relplot_and_save,
            outpath=os.path.join(output_dir, 'layered-trainer-{}_mean_mean-dice_score.png'.format(column)),
            data=stats,
            kind='line',
            x='layer_pos',
            y='{}_mean_mean'.format(column),
            col='fold_train',
            col_order=stats['fold_train'].sort_values().unique(),
            row='trainer_short',
            row_order=stats['trainer_short'].sort_values().unique(),
            hue='dice_score_mean',
            palette='cool',
            style='domain_val',
            size='epoch',
            aspect=2,
            height=6,
            #estimator=None,
            #ci=None,
        )
    join_async_tasks()


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
dataset_keys = None
with open('code/analyze/ids_small.json') as f:
    dataset_keys = json.load(f)


# extract_featurewise_measurements(
#     task=task,
#     trainers=trainers,
#     fold_trains=folds,
#     epochs=epochs,
#     output_directory_base='archive/old/nnUNet-container/data/measurements',
#     dataset_keys=dataset_keys
#)

# store_activation_maps(
#     task=task,
#     trainers=trainers,
#     fold_trains=folds,
#     epochs=epochs,
#     output_directory_base='data/fig/activation_maps/'
# )

# create_stats_scanwise(
#     task=task,
#     trainers=trainers,
#     fold_trains=folds,
#     epochs=epochs,
# )

plot()