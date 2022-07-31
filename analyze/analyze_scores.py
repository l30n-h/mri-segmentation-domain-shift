import pandas as pd
import matplotlib.pyplot as plt
import itertools
import os

import helpers as hlp

def plot_scores(task, trainers, folds, epochs):
    scores = pd.concat([
        hlp.get_scores(
            task, trainer, fold, epoch
        ) for trainer, fold, epoch in itertools.product(trainers, folds, epochs)
    ])

    print(scores)
    scores['trainer_short'] = scores['trainer'].apply(hlp.get_trainer_short)
    

    scores_epoched = scores.groupby(
        ['trainer_short', 'fold_train', 'fold_test', 'is_validation', 'epoch'],
    ).agg(
        iou_score_mean=('iou_score', 'mean'),
        dice_score_mean=('dice_score', 'mean'),
        sdice_score_mean=('sdice_score', 'mean'),  
        iou_score_std=('iou_score', 'std'),
        dice_score_std=('dice_score', 'std'),
        sdice_score_std=('sdice_score', 'std'), 
    ).reset_index()
    print(scores_epoched)
    scores_epoched['subplot'] = scores_epoched['trainer_short'] + '_' + scores_epoched['fold_train']
    scores_epoched['line'] = scores_epoched['fold_test'] + '_' + scores_epoched['is_validation'].map(str)
    # scores_epoched = scores_epoched.groupby(['subplot', 'line']).agg(
    #     iou_score_mean=('iou_score_mean', 'first'),
    #     dice_score_mean=('dice_score_mean', 'first'),
    #     sdice_score_mean=('sdice_score_mean', 'first'),
    # )
    epoched_line_columns = scores_epoched['line'].unique()
    scores_epoched = scores_epoched.pivot(
        index=['subplot', 'epoch'],
        columns=['line']
    ).reset_index(col_level=1)


    scores_epoched_trainer = scores.groupby(
        ['trainer_short', 'is_validation', 'epoch'],
    ).agg(
        iou_score_mean=('iou_score', 'mean'),
        dice_score_mean=('dice_score', 'mean'),
        sdice_score_mean=('sdice_score', 'mean'),  
        iou_score_std=('iou_score', 'std'),
        dice_score_std=('dice_score', 'std'),
        sdice_score_std=('sdice_score', 'std'), 
    ).reset_index()
    print(scores_epoched_trainer)
    scores_epoched_trainer['subplot'] = scores_epoched_trainer['trainer_short'].str.replace(r'.*?(Adam|SGD).*', lambda m: m.group(1), regex=True)
    scores_epoched_trainer['line'] = scores_epoched_trainer['trainer_short'].str.replace(r'(Adam|SGD)_', '', regex=True) + '_' + scores_epoched_trainer['is_validation'].map(str)
    epoched_trainer_line_columns = scores_epoched_trainer['line'].unique()
    scores_epoched_trainer = scores_epoched_trainer.pivot(
        index=['subplot', 'epoch'],
        columns=['line']
    ).reset_index(col_level=1)


    output_dir = 'data/fig/scores'
    os.makedirs(output_dir, exist_ok=True)

    columns = [
        'dice_score_mean',
        'sdice_score_mean',
        'iou_score_mean',
        'dice_score_std',
        'sdice_score_std',
        'iou_score_std',
    ]
    for column in columns:
        print(column)
        scores_epoched_filtered = scores_epoched[['', column]]
        scores_epoched_filtered.columns = scores_epoched_filtered.columns.droplevel(0)
        fig, axes = hlp.create_plot(
            scores_epoched_filtered,
            column_x='epoch',
            column_y=epoched_line_columns,
            kind='line',
            column_subplots='subplot',
            #column_color='iou_score_mean',
            #column_color='sdice_score_mean',
            #colors=stats_epoched_colors,
            ncols=2,
            figsize=(42, 24*6),
            lim_same_x=True,
            lim_same_y=True,
            lim_same_c=True,
            colormap='cool',
            legend=True,
            fig_num='activations-roughness-and-confidence',
            fig_clear=True
        )
        fig.savefig(
            os.path.join(
                output_dir,
                'scores-epoch-{}.png'.format(column)
            ),
            bbox_inches='tight',
            pad_inches=0
        )
        plt.close(fig)


        scores_epoched_trainer_filtered = scores_epoched_trainer[['', column]]
        scores_epoched_trainer_filtered.columns = scores_epoched_trainer_filtered.columns.droplevel(0)
        fig, axes = hlp.create_plot(
            scores_epoched_trainer_filtered,
            column_x='epoch',
            column_y=epoched_trainer_line_columns,
            kind='line',
            column_subplots='subplot',
            #column_color='iou_score_mean',
            #column_color='sdice_score_mean',
            #colors=stats_epoched_colors,
            ncols=2,
            figsize=(42, 24),
            lim_same_x=True,
            lim_same_y=True,
            lim_same_c=True,
            colormap='cool',
            legend=True,
            fig_num='activations-roughness-and-confidence',
            fig_clear=True
        )
        fig.savefig(
            os.path.join(
                output_dir,
                'scores-epoch-trainer-{}.png'.format(column)
            ),
            bbox_inches='tight',
            pad_inches=0
        )
        plt.close(fig)


task = 'Task601_cc359_all_training'
trainers = [
    'nnUNetTrainerV2_MA_noscheduler_depth5_SGD_ep120__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_SGD_ep120_noDA__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_ep120__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_ep120_noDA__nnUNetPlansv2.1',
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
epochs = [10,20,30,40,80,120]

plot_scores(task, trainers, folds, epochs)