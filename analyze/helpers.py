import os
import pandas as pd
import numpy as np

def get_layers_ordered():
    return [
        'conv_blocks_context.0.blocks.0.conv',
        'conv_blocks_context.0.blocks.0.instnorm',
        'conv_blocks_context.0.blocks.1.conv',
        'conv_blocks_context.0.blocks.1.instnorm',
        'conv_blocks_context.1.blocks.0.conv',
        'conv_blocks_context.1.blocks.0.instnorm',
        'conv_blocks_context.1.blocks.1.conv',
        'conv_blocks_context.1.blocks.1.instnorm',
        'conv_blocks_context.2.blocks.0.conv',
        'conv_blocks_context.2.blocks.0.instnorm',
        'conv_blocks_context.2.blocks.1.conv',
        'conv_blocks_context.2.blocks.1.instnorm',
        'conv_blocks_context.3.blocks.0.conv',
        'conv_blocks_context.3.blocks.0.instnorm',
        'conv_blocks_context.3.blocks.1.conv',
        'conv_blocks_context.3.blocks.1.instnorm',
        'conv_blocks_context.4.0.blocks.0.conv',
        'conv_blocks_context.4.0.blocks.0.instnorm',
        'conv_blocks_context.4.1.blocks.0.conv',
        'conv_blocks_context.4.1.blocks.0.instnorm',
        'tu.0',
        'conv_blocks_localization.0.0.blocks.0.conv',
        'conv_blocks_localization.0.0.blocks.0.instnorm',
        'conv_blocks_localization.0.1.blocks.0.conv',
        'conv_blocks_localization.0.1.blocks.0.instnorm',
        'seg_outputs.0',
        'tu.1',
        'conv_blocks_localization.1.0.blocks.0.conv',
        'conv_blocks_localization.1.0.blocks.0.instnorm',
        'conv_blocks_localization.1.1.blocks.0.conv',
        'conv_blocks_localization.1.1.blocks.0.instnorm',
        'seg_outputs.1',
        'tu.2',
        'conv_blocks_localization.2.0.blocks.0.conv',
        'conv_blocks_localization.2.0.blocks.0.instnorm',
        'conv_blocks_localization.2.1.blocks.0.conv',
        'conv_blocks_localization.2.1.blocks.0.instnorm',
        'seg_outputs.2',
        'tu.3',
        'conv_blocks_localization.3.0.blocks.0.conv',
        'conv_blocks_localization.3.0.blocks.0.instnorm',
        'conv_blocks_localization.3.1.blocks.0.conv',
        'conv_blocks_localization.3.1.blocks.0.instnorm',
        'seg_outputs.3',
    ]

def get_trainer_short(x):
    return ('wd0_' if 'wd0' in x else '') + ('SGD' if 'SGD' in x else 'Adam') + ('_noDA' if 'noDA' in x else '')


def get_layers_position_map():
    return dict(map(reversed, enumerate(get_layers_ordered())))


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



import matplotlib.pyplot as plt
def create_scatter_plot(
    df,
    column_x,
    column_y,
    column_subplots,
    column_size=None,
    column_color=None,
    ncols=1,
    figsize=(32, 24),
    lim_same_x=True,
    lim_same_y=True,
    lim_same_c=True,
    colormap='cool'
):
    def nan_to(v, to):
        return to if v != v else v
    def get_lim(df, column):
        v_min = nan_to(df[[column]].min().item(), 0)
        v_max = nan_to(df[[column]].max().item(), 0)
        v_ext = (v_max - v_min) * 0.01
        return v_min - v_ext, v_max + v_ext
    
    names_subplot = df[column_subplots].unique()
    fig, axes = plt.subplots(
        figsize=figsize,
        nrows=len(names_subplot) // ncols,
        ncols=ncols
    )
    axes_flat = axes.flat

    for i, name_subplot in enumerate(names_subplot):
        df_filtered = df[df[column_subplots] == name_subplot]
        df_filtered.plot.scatter(
            x=column_x,
            y=column_y,
            title=name_subplot,
            legend=True,
            ax=axes_flat[i],
            s=column_size,
            c=column_color,
            colormap=colormap,
            colorbar=(not lim_same_c),
            xlim=get_lim(df, column_x) if lim_same_x else None,
            ylim=get_lim(df, column_y) if lim_same_y else None,
            vmin=df[column_color].min().item() if lim_same_c else None,
            vmax=df[column_color].max().item() if lim_same_c else None,
        )

    if lim_same_c:
        fig.colorbar(plt.gca().get_children()[0], ax=axes.ravel().tolist())
    
    return fig, axes