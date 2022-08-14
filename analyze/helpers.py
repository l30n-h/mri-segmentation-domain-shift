import os
import pandas as pd
import numpy as np
import torch
from functools import lru_cache
from nnunet.training.model_restore import load_model_and_checkpoint_files
from nnunet.training.network_training.masterarbeit.inference.predict_preprocessed import predict_preprocessed_ram
import nnunet.training.network_training.masterarbeit.inference.activations_extraction as act_ext

def get_layers_ordered():
    return [
        'conv_blocks_context.0.blocks.0.conv',
        'conv_blocks_context.0.blocks.0.instnorm',
        'conv_blocks_context.0.blocks.0.relu',
        'conv_blocks_context.0.blocks.1.conv',
        'conv_blocks_context.0.blocks.1.instnorm',
        'conv_blocks_context.0.blocks.1.relu',
        'conv_blocks_context.1.blocks.0.conv',
        'conv_blocks_context.1.blocks.0.instnorm',
        'conv_blocks_context.1.blocks.0.relu',
        'conv_blocks_context.1.blocks.1.conv',
        'conv_blocks_context.1.blocks.1.instnorm',
        'conv_blocks_context.1.blocks.1.relu',
        'conv_blocks_context.2.blocks.0.conv',
        'conv_blocks_context.2.blocks.0.instnorm',
        'conv_blocks_context.2.blocks.0.relu',
        'conv_blocks_context.2.blocks.1.conv',
        'conv_blocks_context.2.blocks.1.instnorm',
        'conv_blocks_context.2.blocks.1.relu',
        'conv_blocks_context.3.blocks.0.conv',
        'conv_blocks_context.3.blocks.0.instnorm',
        'conv_blocks_context.3.blocks.0.relu',
        'conv_blocks_context.3.blocks.1.conv',
        'conv_blocks_context.3.blocks.1.instnorm',
        'conv_blocks_context.3.blocks.1.relu',
        'conv_blocks_context.4.0.blocks.0.conv',
        'conv_blocks_context.4.0.blocks.0.instnorm',
        'conv_blocks_context.4.0.blocks.0.relu',
        'conv_blocks_context.4.1.blocks.0.conv',
        'conv_blocks_context.4.1.blocks.0.instnorm',
        'conv_blocks_context.4.1.blocks.0.relu',
        'tu.0',
        'conv_blocks_localization.0.0.blocks.0.conv',
        'conv_blocks_localization.0.0.blocks.0.instnorm',
        'conv_blocks_localization.0.0.blocks.0.relu',
        'conv_blocks_localization.0.1.blocks.0.conv',
        'conv_blocks_localization.0.1.blocks.0.instnorm',
        'conv_blocks_localization.0.1.blocks.0.relu',
        'seg_outputs.0',
        'tu.1',
        'conv_blocks_localization.1.0.blocks.0.conv',
        'conv_blocks_localization.1.0.blocks.0.instnorm',
        'conv_blocks_localization.1.0.blocks.0.relu',
        'conv_blocks_localization.1.1.blocks.0.conv',
        'conv_blocks_localization.1.1.blocks.0.instnorm',
        'conv_blocks_localization.1.1.blocks.0.relu',
        'seg_outputs.1',
        'tu.2',
        'conv_blocks_localization.2.0.blocks.0.conv',
        'conv_blocks_localization.2.0.blocks.0.instnorm',
        'conv_blocks_localization.2.0.blocks.0.relu',
        'conv_blocks_localization.2.1.blocks.0.conv',
        'conv_blocks_localization.2.1.blocks.0.instnorm',
        'conv_blocks_localization.2.1.blocks.0.relu',
        'seg_outputs.2',
        'tu.3',
        'conv_blocks_localization.3.0.blocks.0.conv',
        'conv_blocks_localization.3.0.blocks.0.instnorm',
        'conv_blocks_localization.3.0.blocks.0.relu',
        'conv_blocks_localization.3.1.blocks.0.conv',
        'conv_blocks_localization.3.1.blocks.0.instnorm',
        'conv_blocks_localization.3.1.blocks.0.relu',
        'seg_outputs.3',
    ]

@lru_cache
def get_layers_position_map():
    return dict(map(reversed, enumerate(get_layers_ordered())))

def get_previous_layer(layer_name):
    layer_position_map = get_layers_position_map()
    pos_pre = layer_position_map[layer_name] - 1
    if pos_pre < 0:
        return 'input'
    layer_pre = get_layers_ordered()[pos_pre]
    if layer_pre.startswith('seg_outputs.'):
        return get_previous_layer(layer_pre)
    return layer_pre

def get_activations_input(layer_name, activations_dict):
    layer_pre = get_previous_layer(layer_name)
    data = get_activations_output(layer_pre, activations_dict)
    if layer_pre.startswith('tu.'):
        d = int(layer_pre.replace('tu.', ''))
        conv_blocks_context = get_activations_output('conv_blocks_context.{}.blocks.1.instnorm'.format(3-d), activations_dict)
        return torch.cat((data, conv_blocks_context), axis=1)
    return data

def get_activations_output(layer_name, activations_dict):
    if layer_name.endswith('.relu'):
        data = get_activations_input(layer_name, activations_dict)
        return torch.where(data > 0, data, data * 0.01)
    return activations_dict.get(layer_name, None)

def get_layer_config(layer_name):
    if layer_name.startswith('seg_outputs.'):
        return { 'kernel_size': [1, 1], 'stride': [1, 1] }

    if layer_name.startswith('tu.'):
        if layer_name == 'tu.0':
            return { 'kernel_size': [2, 1], 'stride': [2, 1] }
        return { 'kernel_size': [2, 2], 'stride': [2, 2] }

    if layer_name.startswith('conv_blocks_localization.'):
        return { 'kernel_size': [3, 3], 'stride': [1, 1] }
    
    if layer_name.startswith('conv_blocks_context.'):
        if 'blocks.1.' in layer_name or '_context.0.' in layer_name or '_context.4.1.' in layer_name:
            return { 'kernel_size': [3, 3], 'stride': [1, 1] }
        if '_context.4.0.' in layer_name:
            return { 'kernel_size': [3, 3], 'stride': [2, 1] }
        return { 'kernel_size': [3, 3], 'stride': [2, 2] }
    return {}
    

def get_trainer_short(x):
    return '{}_wd={}_DA={}'.format(
        'SGD' if 'SGD' in x else 'Adam',
        'wd0' not in x,
        'noDA' not in x
    )


def get_fold_id_mapping():
    return {
        'siemens15': 0,
        'siemens3': 1,
        'ge15': 2,
        'ge3': 3,
        'philips15': 4,
        'philips3': 5
    }

def load_split_all(task):
    return np.load(
        os.path.join(
            'archive/old/nnUNet-container/data/nnUNet_preprocessed/',
            task,
            'splits_final.pkl'
        ),
        allow_pickle=True
    )

def load_split(task, fold):
    return load_split_all(task)[get_fold_id_mapping()[fold]]

def get_id_from_filename(filename):
    return filename.split('_', 1)[0]

def get_testdata_dir(task, trainer, fold_train, epoch, folder_test='archive/old/nnUNet-container/data/testout'):
    tester = trainer.replace('nnUNetTrainerV2_', '').replace('__nnUNetPlansv2.1', '')
    directory_testout = os.path.join(
        folder_test,
        task,
        '{}-ep{:0>3}-{}'.format(tester, epoch, fold_train)
    )
    return directory_testout



def load_model(trainer, fold, epoch, folder_train='archive/old/nnUNet-container/data/nnUNet_trained_models/nnUNet/2d/Task601_cc359_all_training', eval=True, use_test_hooks=True, test_include_gt=True):
    tmodel, params = load_model_and_checkpoint_files(
        os.path.join(folder_train, trainer),
        get_fold_id_mapping()[fold],
        mixed_precision=True,
        checkpoint_name='model_ep_{:0>3}'.format(epoch)
    )
    if epoch > 0:
        tmodel.load_checkpoint_ram(params[0], False)
    if eval == True:
        tmodel.network.eval()
    return tmodel

def generate_predictions_ram(trainer, dataset_keys=None, activations_extractor_kwargs={}):
    return predict_preprocessed_ram(
        trainer=trainer,
        activations_extractor=act_ext.activations_extractor(
            **activations_extractor_kwargs
        ),
        dataset_keys=dataset_keys,
        do_mirroring=False,
        # use_sliding_window=True,
        # step_size=0.5,
        # use_gaussian=True,
        # all_in_gpu=False
    )


def get_scores(task, trainer, fold_train, epoch, **kwargs):
    directory_testout = get_testdata_dir(task, trainer, fold_train, epoch, **kwargs)
    scores = pd.read_csv(os.path.join(directory_testout, 'scores.csv'))
    
    #todo should be set in calc_score
    scores.drop('Unnamed: 0', axis='columns', inplace=True)
    scores['trainer'] = trainer
    scores['epoch'] = epoch
    scores['fold_train'] = fold_train
    scores['fold'] = scores['tomograph_model'].str.cat(scores['tesla_value'].astype(str))
    scores.rename(columns={ 'fold': 'fold_test' }, inplace=True)
    val_set = set(map(
        get_id_from_filename,
        sum(
            map(
                lambda x: x['val'],
                load_split_all(task)
            ),
            []
        )
    ))
    scores['is_validation'] = scores['id'].apply(lambda x: x in val_set)
    scores['iou_score'] = scores['dice_score'].apply(dice_to_iou)
    return scores

def load_domainshift_scores(task, trainer, fold_train, epoch):
    scores = get_scores(task, trainer, fold_train, epoch)

    scores_same = scores[scores['fold_train'] == scores['fold_test']]
    scores_other = scores[scores['fold_train'] != scores['fold_test']]
    return pd.DataFrame({
        'trainer': [trainer],
        'fold_train': [fold_train],
        'epoch': [epoch],
        'dice_score_all_mean': [scores['dice_score'].mean()],
        'dice_score_all_std': [scores['dice_score'].std()],
        'sdice_score_all_mean': [scores['sdice_score'].mean()],
        'sdice_score_all_std': [scores['sdice_score'].std()],
        'dice_score_same_mean': [scores_same['dice_score'].mean()],
        'dice_score_same_std': [scores_same['dice_score'].std()],
        'sdice_score_same_mean': [scores_same['sdice_score'].mean()],
        'sdice_score_same_std': [scores_same['sdice_score'].std()],
        'dice_score_other_mean': [scores_other['dice_score'].mean()],
        'dice_score_other_std': [scores_other['dice_score'].std()],
        'sdice_score_other_mean': [scores_other['sdice_score'].mean()],
        'sdice_score_other_std': [scores_other['sdice_score'].std()],

        'dice_score_diff_mean': [scores_same['dice_score'].mean() - scores_other['dice_score'].mean()],
        'dice_score_diff_std': [scores_same['dice_score'].std() - scores_other['dice_score'].std()],
        'sdice_score_diff_mean': [scores_same['sdice_score'].mean() - scores_other['sdice_score'].mean()],
        'sdice_score_diff_std': [scores_same['sdice_score'].std() - scores_other['sdice_score'].std()]
    })



def numerate_nested(df, columns, column_name_out='x'):
    grouped = df[columns].groupby(columns[0:-1])
    num_local = grouped.value_counts(normalize=True).groupby(columns[0:-1]).cumsum() * 0.7
    shifted = num_local - num_local.groupby(columns[0:-1]).min()
    num_global = shifted + df.assign(group_index=grouped.ngroup()).groupby(columns)['group_index'].first()
    return df.join(
        num_global.reset_index().set_index(columns).rename(columns={0: column_name_out}),
        on=columns
    )

import matplotlib as mpl
import matplotlib.pyplot as plt
def create_plot(
    df,
    column_x,
    column_y,
    column_subplots,
    kind='scatter',
    column_size=None,
    column_color=None,
    colors=None,
    ncols=1,
    figsize=(32, 24),
    lim_same_x=True,
    lim_same_y=True,
    lim_same_c=True,
    colormap='cool',
    legend=True,
    fig_num=None,
    fig_clear=False
):
    colors = df if colors is None else colors
    def nan_or_inf_to(v, to):
        return to if v != v or abs(v) == float('inf') else v
    def get_lim(df, column):
        v_min = nan_or_inf_to(df[column].min().min().item(), 0)
        v_max = nan_or_inf_to(df[column].max().max().item(), 0)
        v_ext = (v_max - v_min) * 0.01
        return v_min - v_ext, v_max + v_ext
    def get_scalar_map(colors):
        if column_color is None:
            return None
        return mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(
                vmin=colors[column_color].min().item(),
                vmax=colors[column_color].max().item()
            ),
            cmap=plt.get_cmap(colormap)
        )
    
    names_subplot = sorted(df[column_subplots].unique())
    fig, axes = plt.subplots(
        num=fig_num,
        clear=fig_clear,
        figsize=figsize,
        nrows=int(np.ceil(len(names_subplot) / ncols)),
        ncols=ncols
    )
    axes = np.atleast_1d(axes)
    axes_flat = axes.flat

    scalar_map = get_scalar_map(colors) if lim_same_c else None
    xlim = get_lim(df, column_x) if lim_same_x else None
    ylim = get_lim(df, column_y) if lim_same_y else None

    def get_attributes(df_filtered, name_subplot):
        if kind == 'line':
            if column_color is None:
                return {}
            colors_filtered = colors.xs(name_subplot)
            scalar_map_filtered = get_scalar_map(colors_filtered) if scalar_map is None else scalar_map
            return {
                'color': dict(map(
                    lambda item: (item[0], scalar_map_filtered.to_rgba(item[1])),
                    colors_filtered[column_color].to_dict().items(),
                ))
            }
        return {
            'c': column_color,
            's': column_size,
            'colormap': colormap,
            'colorbar': (not lim_same_c),
            'vmin': df[column_color].min().item() if lim_same_c else None,
            'vmax': df[column_color].max().item() if lim_same_c else None,
        }
        

    for i, name_subplot in enumerate(names_subplot):
        df_filtered = df[df[column_subplots] == name_subplot]
        df_filtered.plot(
            kind=kind,
            x=column_x,
            y=column_y,
            title=name_subplot,
            legend=legend,
            grid=True,
            ax=axes_flat[i],
            xlim=xlim,
            ylim=ylim,
            **get_attributes(df_filtered, name_subplot)
        )

    if scalar_map is not None:
        fig.colorbar(scalar_map, ax=axes.ravel().tolist())
    
    return fig, axes




import torch

def get_normed(features, dim=1):
    norm = features.norm(dim=dim, keepdim=True)
    scale = 1.0 / (norm + 1e-12)
    zero = norm > 1e-6
    return features * scale * zero, norm * zero

def apply_measure_on_all_combinations(features_a, features_b, measure_fn):
    #a = features_a.repeat_interleave(features_a.shape[0], dim=0)
    #b = features_b.repeat(features_b.shape[0], 1)
    #similarity = measure_fn(a, b)
    #return similarity.view(features_a.shape[0], features_b.shape[0])
    # below is faster and does the same
    triu_ids = torch.triu_indices(features_a.shape[0], features_b.shape[0], device=features_a.device)
    a = features_a[triu_ids[0]]
    b = features_b[triu_ids[1]]
    similarity = measure_fn(a, b)
    z = torch.zeros(features_a.shape[0], features_b.shape[0], device=features_a.device)
    z[triu_ids[0], triu_ids[1]] = similarity
    z[triu_ids[1], triu_ids[0]] = similarity
    return z

def get_cosine_similarity(features_a, features_b):
    return apply_measure_on_all_combinations(
        features_a,
        features_b,
        lambda a,b: torch.nn.functional.cosine_similarity(a, b, dim=1)
    )

def get_l2_similarity(features_a, features_b):
    return apply_measure_on_all_combinations(
        features_a,
        features_b,
        lambda a,b: 1.0 / (1 + (a - b).norm(dim=1))
    )

def standardized_moment(array, order, dim=None, keepdim=False):
    mean = array.mean(dim=dim, keepdim=True)
    array_shifted = (array - mean)
    mean_o = array_shifted.pow(order).mean(dim=dim, keepdim=keepdim)
    var_o = array_shifted.pow(2).mean(dim=dim, keepdim=keepdim).pow(order / 2.0)
    return mean_o / var_o

def get_noise_estimates(input):
    if input.shape[-1] < 3 or input.shape[-2] < 3:
        return torch.full(input.shape[0:2], float('nan'), device=input.device)
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


def dice_to_iou(dice_score):
    return dice_score / (2.0 - dice_score)

def iou_to_dice(iou):
    return 2.0 * iou / (1.0 + iou)


def get_moments(name, value, dim=None, keepdim=False):
    dim_str = "-".join(map(str, dim)) if isinstance(dim, (list, tuple)) else str(dim)
    yield '{}_mean_{}'.format(name, dim_str), value.mean(dim=dim, keepdim=keepdim) # brighness
    yield '{}_std_{}'.format(name, dim_str), value.std(dim=dim, keepdim=keepdim) # rms contrast
    # yield '{}_skewness_{}'.format(name, dim_str), standardized_moment(value, order=3, dim=dim, keepdim=keepdim)
    # yield '{}_kurtosis_{}'.format(name, dim_str), standardized_moment(value, order=4, dim=dim, keepdim=keepdim)

def get_dims(a):
    if a.shape[0] > 1 and a.shape[1] > 1:
        yield (0, 1)
    if a.shape[0] > 1:
        yield 0
    if a.shape[1] > 1:
        yield 1
    if a.shape[2] > 1 or a.shape[3] > 1:
        yield (2, 3)

def get_moments_recursive(name, value):
    if value.numel() == 1:
        yield name, value
    else:
        for dim in get_dims(value):
            for moment_name, moment in get_moments(name, value, dim=dim, keepdim=True):
                yield from get_moments_recursive(moment_name, moment)