import numpy as np
import torch
import pandas as pd
import itertools
import glob
import os
import helpers as hlp
import json

from multiprocessing import Pool
import os

def store_data(output_path, data):
    torch.save(data, '{}.pt'.format(output_path))

def get_async_store_queue(output_directory_base, store_fn=store_data, num_threads=8):
    results = []
    pool = Pool(num_threads)
    def store_data_async(output_path_sub, data):
        output_filename = os.path.join(output_directory_base, output_path_sub)
        output_dir = os.path.dirname(output_filename)
        os.makedirs(output_dir, exist_ok=True)
        results.append(pool.apply_async(
            store_fn, (
                output_filename,
                data,
            )
        ))
    def join_all_tasks():
        return [i.get() for i in results]
    return store_data_async, join_all_tasks




def extract_central_patch(trainer, data):
    num_batches = 64
    patch_size = np.array(trainer.patch_size)
    patch_size_halfed = np.array(trainer.patch_size) // 2
    start = np.array(data.shape[2:4]) // patch_size_halfed * patch_size_halfed
    end = start + patch_size
    batches = np.linspace(0, data.shape[1]-1, num_batches).astype(int)
    return data[:, batches, start[0] : end[0], start[1] : end[1]]

def apply_prediction_data_filter_monkey_patch(trainer):
    predict_original = trainer.predict_preprocessed_data_return_seg_and_softmax
    def predict_patched(*args, **kwargs):
        # extract patch here to only get one patch in feature extraction per slice
        # => evaluation faster and more consistent
        data = extract_central_patch(trainer, args[0])
        return predict_original(data, *args[1:], **kwargs)
    trainer.predict_preprocessed_data_return_seg_and_softmax = predict_patched
    return predict_original


def get_tensor_memsize_estimate(t):
    return t.nelement() * t.element_size()

def get_activations_dict_memsize_estimate(activations_dict):
    return sum(map(get_tensor_memsize_estimate, activations_dict.values()))




def extract_featurewise_scalar_measurements(
    task,
    trainers,
    folds,
    epochs,
    output_directory_base,
    dataset_keys=None
    ):
    layers_set = set(hlp.get_layers_ordered())
    def is_module_tracked(name, module):
        return name in layers_set

    def extract_activations_full(name, activations, slices_and_tiles_count, activations_dict):
        data = activations_dict.setdefault(name, [])
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
        data.append(
            torch.cat(tuple(results.values()),dim=0).half()
        )
        #data['measurement_names'] = tuple(results.keys())
        return activations_dict
    

    store_data_async, join_all_tasks = get_async_store_queue(
        output_directory_base=output_directory_base,
        num_threads=8
    )

    for trainer, fold, epoch in itertools.product(trainers, folds, epochs):
        tmodel = hlp.load_model(trainer, fold, epoch)
        
        apply_prediction_data_filter_monkey_patch(tmodel)

        for id, prediction, activations_dict in hlp.generate_predictions_ram(
            trainer=tmodel,
            dataset_keys=dataset_keys,
            activations_extractor_kwargs=dict(
                extract_activations=extract_activations_full,
                is_module_tracked=is_module_tracked
            )
        ):
            print(id, activations_dict.keys(), activations_dict['input'].shape)
            print(get_activations_dict_memsize_estimate(activations_dict) / 1024 / 1024)
            activations_dict = dict(map(
                lambda kv: (kv[0], kv[1].cpu()),
                activations_dict.items()
            ))
            store_data_async(
                os.path.join(task, trainer, fold, str(epoch).rjust(3, '0'), id),
                activations_dict
            )
        join_all_tasks()


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


extract_featurewise_scalar_measurements(
    task=task,
    trainers=trainers,
    folds=folds,
    epochs=epochs,
    output_directory_base='archive/old/nnUNet-container/data/measurements',
    dataset_keys=dataset_keys
)