import pandas as pd
import nibabel as nib
import numpy as np
import glob
import os
import multiprocessing
from functools import partial
import math

from spottunet.utils import sdice
from dpipe.im.metrics import dice_score

### CONFIG ###
dataset_base_path = 'data/nnUNet_raw/nnUNet_raw_data/Task601_cc359_all_training'
dataset_meta_path = os.path.join(dataset_base_path, 'meta.csv')

predictions_base_path_glob = 'archive/old/nnUNet-container/data/testout/Task601_cc359_all_training/MA_*'
#predictions_base_path_glob = 'data/testout/MA_*'
only_full_datasets = True
overwrite = False
### ------ ###

def parallelize(data, func, num_of_processes=8):
    data_split = np.array_split(data, num_of_processes)
    pool = multiprocessing.Pool(num_of_processes)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data

def apply_on_rows_expand(func, data):
    return data.apply(func, axis='columns', result_type='expand')

def apply_on_rows_expand_parallelized(func, data, num_of_processes=8):
    return parallelize(
        data, 
        partial(
            apply_on_rows_expand,
            func
        ),
        num_of_processes
    )


def get_binary_prediction(x, threshold=0.5):
    return x > threshold

def dice_metric(prediction, target):
    return dice_score(
        get_binary_prediction(prediction),
        get_binary_prediction(target)
    )

def sdice_metric(prediction, target, voxel_spacing):
    sdice_tolerance = 1
    return sdice(
        get_binary_prediction(prediction),
        get_binary_prediction(target),
        voxel_spacing,
        sdice_tolerance
    )


def get_nii_scores(meta):
    nii_prediction = nib.load(meta.prediction_path)
    nii_target = nib.load(os.path.join(dataset_base_path, meta.brain_mask))
    prediction = nii_prediction.get_fdata()
    target = nii_target.get_fdata()
    voxel_spacing = np.array([meta.x, meta.y, meta.z])
    dice_score = dice_metric(target, prediction)
    sdice_score = sdice_metric(target, prediction, voxel_spacing)
    return dice_score, sdice_score

def calc_scores_per_row(meta):
    dice_score, sdice_score = get_nii_scores(meta)
    out = {
        "id": meta.id,
        "fold": meta.fold,
        "tomograph_model": meta.tomograph_model,
        "tesla_value": meta.tesla_value,
        "dice_score": dice_score,
        "sdice_score": sdice_score,
    }
    return out



def calc_scores(meta, predictions_base_path, output_path):
    meta['predictions_base_path'] = meta.apply(
        lambda d: predictions_base_path,
        axis='columns'
    )
    meta['prediction_path'] = meta.apply(
        lambda d: d['brain_mask'].replace('labelsTs', predictions_base_path),
        axis='columns'
    )
    out = apply_on_rows_expand_parallelized(
        calc_scores_per_row,
        meta,
        num_of_processes=math.ceil(multiprocessing.cpu_count()*0.8)
    )
    out.to_csv(output_path)


meta = pd.read_csv(dataset_meta_path)

for predictions_base_path in glob.iglob(predictions_base_path_glob, recursive=False):
    print(predictions_base_path)
    output_path = os.path.join(predictions_base_path, 'scores.csv')
    predictions_base_path = os.path.join(predictions_base_path, 'prediction_raw')
    if not overwrite and os.path.exists(output_path):
        print('already exists. skipped...')
        continue
    
    ids_available = set(map(
        lambda x: x.split('_')[0],
        map(
            os.path.basename,
            os.listdir(predictions_base_path)
        )
    ))
    if only_full_datasets and len(ids_available) < len(meta.index):
        print('prediction not finished. skipped...')
        continue
    meta = meta[meta['id'].isin(ids_available)]

    calc_scores(meta.copy(), predictions_base_path, output_path)