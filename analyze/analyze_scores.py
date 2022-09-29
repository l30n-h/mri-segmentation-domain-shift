import pandas as pd
import itertools
import os

import helpers as hlp


def plot_scores_full_small_compare(task, trainers, folds, epochs):
    scores_full = pd.concat([
        hlp.get_scores(
            task, trainer, fold, epoch,
            folder_test='archive/old/nnUNet-container/data/testout'
        ) for trainer, fold, epoch in itertools.product(trainers, folds, epochs)
    ]).reset_index(drop=True)
    scores_full['is_small'] = False

    scores_testaug = pd.concat([
        hlp.get_scores(
            task, trainer, fold, epoch,
            folder_test='archive/old/nnUNet-container/data/testout2'
        ) for trainer, fold, epoch in itertools.product(trainers, folds, epochs)
    ]).reset_index(drop=True)
    scores_testaug['is_small'] = True
    scores_testaug = scores_testaug[scores_testaug['test_augmentation'] == 'None']
    # filter for ids mostly used for experiments
    SCANS_PER_FOLD = 6
    scores_testaug = scores_testaug.sort_values('id').groupby(['trainer', 'fold_train', 'epoch', 'fold_test', 'is_validation', 'is_small']).head(SCANS_PER_FOLD)
    
    scores = pd.concat([ scores_full, scores_testaug ])
    scores = scores.groupby(['trainer', 'fold_train', 'epoch', 'fold_test', 'is_validation', 'is_small']).agg(
        count=('id_long', 'count'),
        dice_score_mean=('dice_score', 'mean'),
        dice_score_std=('dice_score', 'std'),
        iou_score_mean=('iou_score', 'mean'),
        iou_score_std=('iou_score', 'std'),
        sdice_score_mean=('sdice_score', 'mean'),
        sdice_score_std=('sdice_score', 'std')
    ).reset_index()
    scores['same_domain'] = scores['fold_train'] == scores['fold_test']
    scores['domain_val'] = scores['same_domain'].apply(lambda x: 'same' if x else 'other') + ' ' + scores['is_validation'].apply(lambda x: 'validation' if x else '')
    scores['trainer_short'] = scores['trainer'].apply(hlp.get_trainer_short)

    scores = scores[scores['same_domain'] | (~scores['is_validation'])]
    
    print(scores)

    index = ['trainer', 'fold_train', 'epoch', 'fold_test', 'is_validation']
    metrics = ['dice_score_mean', 'dice_score_std', 'iou_score_mean', 'iou_score_std', 'sdice_score_mean', 'sdice_score_std', 'count']
    scores_small = scores[scores['is_small']].set_index(index)[metrics]
    scores_full = scores[~scores['is_small']].set_index(index)[metrics]
    
    print(scores_small)
    print(scores_full)
    diff = scores_full - scores_small 
    print(diff.min())
    print(diff.max())
    print(diff.mean())
    print(diff.std())
    diff = diff.abs()
    print(diff.min())
    print(diff.max())
    print(diff.mean())
    print(diff.std())

    output_dir = 'data/fig/scores/scores-full-small-compare'
    os.makedirs(output_dir, exist_ok=True)

    add_async_task, join_async_tasks = hlp.get_async_queue(num_threads=12)

    columns = [
        'dice_score',
        'sdice_score',
        #'iou_score',
    ]
    aggs = ['mean', 'std']
    for column, agg in itertools.product(columns, aggs):
        add_async_task(
            hlp.relplot_and_save,
            outpath=os.path.join(output_dir, 'scores-fullsmallcompare-epoch-{}-{}.png'.format(column, agg)),
            data=scores,
            kind='line',
            x='epoch',
            y='{}_{}'.format(column, agg),
            col='domain_val',
            col_order=scores['domain_val'].sort_values().unique(),
            hue='trainer_short',
            style='is_small',
            height=6,
        )

    join_async_tasks()

def plot_scores(task, trainers, folds, epochs):
    scores = pd.concat([
        hlp.get_scores(
            task, trainer, fold, epoch
        ) for trainer, fold, epoch in itertools.product(trainers, folds, epochs)
    ]).reset_index(drop=True)

    scores = scores.groupby(['trainer', 'fold_train', 'epoch', 'fold_test', 'is_validation']).agg(
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

    scores['wd_bn'] = scores['wd'].apply(lambda x: 'wd=' + str(x)) + ' ' + scores['bn'].apply(lambda x: 'bn=' + str(x))
    scores['wd_DA'] = scores['wd'].apply(lambda x: 'wd=' + str(x)) + ' ' + scores['DA']
    scores['optimizer_wd_DA'] = scores['optimizer'] + ' ' + scores['wd_DA']
    scores['optimizer_DA'] = scores['optimizer'] + ' ' + scores['DA']
    scores['wd_bn_DA'] = scores['wd_bn'] + ' ' + scores['DA']
    scores['same_domain'] = scores['fold_train'] == scores['fold_test']
    scores['domain_val'] = scores['same_domain'].apply(lambda x: 'same' if x else 'other') + ' ' + scores['is_validation'].apply(lambda x: 'validation' if x else '')
    scores['trainer_short'] = scores['trainer'].apply(hlp.get_trainer_short)


    scores = scores[scores['same_domain'] | (~scores['is_validation'])]
    #scores = scores[scores['test_augmentation'] == 'None']

    print(scores)

    #output_dir = 'data/fig/scores/scores-overfit'
    output_dir = 'data/fig/scores/scores-testaug'
    #output_dir = 'data/fig/scores/scores-testaug-none'
    os.makedirs(output_dir, exist_ok=True)

    
    add_async_task, join_async_tasks = hlp.get_async_queue(num_threads=12)

    aggs = ['mean', 'std']
    for agg in aggs:
        add_async_task(
            hlp.relplot_and_save,
            outpath=os.path.join(output_dir, 'scores-scores-{}.png'.format(agg)),
            data=scores.melt(
                'dice_score_{}'.format(agg),
                value_vars=[
                    'iou_score_{}'.format(agg),
                    'sdice_score_{}'.format(agg)
                ],
                var_name='score_{}'.format(agg),
                value_name='score_value_{}'.format(agg)
            ),
            kind='line',
            x='dice_score_{}'.format(agg),
            y='score_value_{}'.format(agg),
            hue='score_{}'.format(agg),
            errorbar=None,
            height=6,
            facet_kws=dict(
                xlim=(0.0, 1.0),
                ylim=(0.0, 1.0),
            ),
        )

    columns = [
        'dice_score',
        'sdice_score',
        #'iou_score',
    ]
    for column, agg in itertools.product(columns, aggs):
        add_async_task(
            hlp.relplot_and_save,
            outpath=os.path.join(output_dir, 'scores-fold_test-epoch-{}-{}.png'.format(column, agg)),
            data=scores,
            kind='line',
            x='epoch',
            y='{}_{}'.format(column, agg),
            row='optimizer',
            row_order=scores['optimizer'].sort_values().unique(),
            col='wd_bn_DA',
            col_order=scores['wd_bn_DA'].sort_values().unique(),
            hue='fold_test',
            height=6,
        )

        add_async_task(
            hlp.relplot_and_save,
            outpath=os.path.join(output_dir, 'scores-optimizer-epoch-{}-{}.png'.format(column, agg)),
            data=scores,
            kind='line',
            x='epoch',
            y='{}_{}'.format(column, agg),
            row='wd_bn',
            row_order=scores['wd_bn'].sort_values().unique(),
            col='DA',
            col_order=scores['DA'].sort_values().unique(),
            hue='optimizer',
            height=6,
        )

        add_async_task(
            hlp.relplot_and_save,
            outpath=os.path.join(output_dir, 'scores-optimizer-epoch-{}-{}-fold_test.png'.format(column, agg)),
            data=scores,
            kind='line',
            x='epoch',
            y='{}_{}'.format(column, agg),
            row='wd_bn',
            row_order=scores['wd_bn'].sort_values().unique(),
            col='DA',
            col_order=scores['DA'].sort_values().unique(),
            hue='optimizer',
            style='fold_test',
            height=6,
        )

        add_async_task(
            hlp.relplot_and_save,
            outpath=os.path.join(output_dir, 'scores-DA-epoch-{}-{}.png'.format(column, agg)),
            data=scores,
            kind='line',
            x='epoch',
            y='{}_{}'.format(column, agg),
            row='wd_bn',
            row_order=scores['wd_bn'].sort_values().unique(),
            col='optimizer',
            col_order=scores['optimizer'].sort_values().unique(),
            hue='DA',
            height=6,
        )

        add_async_task(
            hlp.relplot_and_save,
            outpath=os.path.join(output_dir, 'scores-domain-epoch-{}-{}.png'.format(column, agg)),
            data=scores,
            kind='line',
            x='epoch',
            y='{}_{}'.format(column, agg),
            row='wd_bn_DA',
            row_order=scores['wd_bn_DA'].sort_values().unique(),
            col='optimizer',
            col_order=scores['optimizer'].sort_values().unique(),
            hue='same_domain',
            style='domain_val',
            height=6,
        )

        add_async_task(
            hlp.relplot_and_save,
            outpath=os.path.join(output_dir, 'scores-trainer-domain-epoch-{}-{}.png'.format(column, agg)),
            data=scores,
            kind='line',
            x='epoch',
            y='{}_{}'.format(column, agg),
            row='trainer_short',
            row_order=scores['trainer_short'].sort_values().unique(),
            col='optimizer',
            col_order=scores['optimizer'].sort_values().unique(),
            hue='same_domain',
            style='domain_val',
            height=6,
        )

        add_async_task(
            hlp.relplot_and_save,
            outpath=os.path.join(output_dir, 'scores-test_augmentation-epoch-{}-{}.png'.format(column, agg)),
            data=scores,
            kind='line',
            x='epoch',
            y='{}_{}'.format(column, agg),
            row='trainer_short',
            row_order=scores['trainer_short'].sort_values().unique(),
            col='optimizer',
            col_order=scores['optimizer'].sort_values().unique(),
            hue='test_augmentation',
            style='domain_val',
            height=6,
        )

        scores2 = scores[scores['DA'].isin(['full', 'none'])]
        add_async_task(
            hlp.relplot_and_save,
            outpath=os.path.join(output_dir, 'scores-bn-epoch-{}-{}.png'.format(column, agg)),
            data=scores2,
            kind='line',
            x='epoch',
            y='{}_{}'.format(column, agg),
            row='fold_train',
            row_order=scores2['fold_train'].sort_values().unique(),
            col='optimizer_DA',
            col_order=scores2['optimizer_DA'].sort_values().unique(),
            hue='wd_bn',
            style='domain_val',
            height=6,
        )

    join_async_tasks()


task = 'Task601_cc359_all_training'
trainers = [
    # 'nnUNetTrainerV2_MA_noscheduler_depth5_SGD_ep120__nnUNetPlansv2.1',
    # 'nnUNetTrainerV2_MA_noscheduler_depth5_SGD_ep120_noDA__nnUNetPlansv2.1',
    # 'nnUNetTrainerV2_MA_noscheduler_depth5_ep120__nnUNetPlansv2.1',
    # 'nnUNetTrainerV2_MA_noscheduler_depth5_ep120_noDA__nnUNetPlansv2.1',

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

    # 'nnUNetTrainerV2_MA_noscheduler_depth7_bf24_wd0_ep360_noDA__nnUNetPlansv2.1',
    # 'nnUNetTrainerV2_MA_noscheduler_depth5_ep360_noDA__nnUNetPlansv2.1',
]
folds = [
    'siemens15',
    'siemens3',
    'ge15',
    'ge3',
    'philips15',
    'philips3'
]
epochs = [10,20,30,40,80,120,200,280,360]

plot_scores_full_small_compare(task, trainers, folds, epochs)
plot_scores(task, trainers, folds, epochs)