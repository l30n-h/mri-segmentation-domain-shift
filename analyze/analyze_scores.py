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
    scores['domain_val'] = scores['same_domain'].apply(lambda x: 'same' if x else 'other') + scores['is_validation'].apply(lambda x: ' validation' if x else '')
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

    scores['wd_bn'] = scores['wd'].apply(lambda x: 'wd=' + str(x)) + ' ' + scores['bn'].apply(lambda x: 'bn=' + str(x))
    scores['wd_DA'] = scores['wd'].apply(lambda x: 'wd=' + str(x)) + ' ' + scores['DA']
    scores['optimizer_wd_DA'] = scores['optimizer'] + ' ' + scores['wd_DA']
    scores['optimizer_DA'] = scores['optimizer'] + ' ' + scores['DA']
    scores['wd_bn_DA'] = scores['wd_bn'] + ' ' + scores['DA']
    scores['same_domain'] = scores['fold_train'] == scores['fold_test']
    scores['domain_val'] = scores['same_domain'].apply(lambda x: 'same' if x else 'other') + scores['is_validation'].apply(lambda x: ' validation' if x else '')
    scores['domain_val_testaug'] = scores['domain_val'] + scores['test_augmentation'].apply(lambda x: '' if x == 'None' else ' testaug')
    scores['trainer_short'] = scores['trainer'].apply(hlp.get_trainer_short)

    scores['same_base_domain'] = scores['fold_train'] == scores['fold_test_base']
    scores['base_domain_val'] = scores['same_base_domain'].apply(lambda x: 'same' if x else 'other') + scores['is_validation'].apply(lambda x: ' validation' if x else '')

    scores = scores[scores['same_domain'] | (~scores['is_validation'])]
    #scores = scores[scores['same_base_domain'] | (~scores['is_validation'])]
    #scores = scores[scores['test_augmentation'] == 'None']
    scores['Data Aug.'] = scores['DA'].str.replace(r'^no([^n].+)$', r'no-\1', n=1, regex=True)
    scores['Domain'] = scores['domain_val_testaug'].str.replace('same validation', 'Validation').str.replace('same', 'Training').str.replace('other testaug', 'Other w/ test aug.').str.replace('other', 'Other w/o test aug.')
    scores['Optimizer'] = scores['optimizer']
    scores['DSC'] = scores['dice_score_mean']
    scores['Surface DSC'] = scores['sdice_score_mean']
    scores['Epoch'] = scores['epoch']
    scores['Normalization'] = scores['bn'].apply(lambda x: 'Batch' if x else 'Instance')

    scores = scores[~scores['wd']]

    print(scores)

    output_dir = 'data/fig/scores/scores-testaug'
    os.makedirs(output_dir, exist_ok=True)
    columns = ['DSC', 'Surface DSC']
    add_async_task, join_async_tasks = hlp.get_async_queue(num_threads=12)

    for column in columns:
        add_async_task(
            hlp.relplot_and_save,
            outpath=os.path.join(output_dir, 'scores-DA-no_bn-epoch-{}.png'.format(column)),
            data=scores[~scores['bn']],
            kind='line',
            x='Epoch',
            y=column,
            col='Optimizer',
            style='Domain',
            hue='Data Aug.',
            height=6,
        )
        add_async_task(
            hlp.relplot_and_save,
            outpath=os.path.join(output_dir, 'scores-DA-Normalization-da_full_none-epoch-{}.png'.format(column)),
            data=scores[scores['Data Aug.'].isin(['full', 'none'])],
            kind='line',
            x='Epoch',
            y=column,
            col='Normalization',
            style='Domain',
            hue='Data Aug.',
            height=6,
        )
        add_async_task(
            hlp.relplot_and_save,
            outpath=os.path.join(output_dir, 'scores-DA-Optimizer-da_full_none-epoch-{}.png'.format(column)),
            data=scores[scores['Data Aug.'].isin(['full', 'none'])],
            kind='line',
            x='Epoch',
            y=column,
            col='Optimizer',
            style='Domain',
            hue='Data Aug.',
            height=6,
        )
        add_async_task(
            hlp.relplot_and_save,
            outpath=os.path.join(output_dir, 'scores-DA-Optimizer-Normalization-da_full_none-epoch-{}.png'.format(column)),
            data=scores[scores['Data Aug.'].isin(['full', 'none'])],
            kind='line',
            x='Epoch',
            y=column,
            row='Optimizer',
            col='Normalization',
            style='Domain',
            hue='Data Aug.',
            height=6,
        )
        add_async_task(
            hlp.relplot_and_save,
            outpath=os.path.join(output_dir, 'scores-optimizer-da_full_none-epoch-{}.png'.format(column)),
            data=scores[scores['Data Aug.'].isin(['full', 'none'])],
            kind='line',
            x='Epoch',
            y=column,
            col='Data Aug.',
            style='Domain',
            hue='Optimizer',
            height=6,
        )
        add_async_task(
            hlp.relplot_and_save,
            outpath=os.path.join(output_dir, 'scores-optimizer-da_full_none-epoch-single-{}.png'.format(column)),
            data=scores[scores['Data Aug.'].isin(['full', 'none'])],
            kind='line',
            x='Epoch',
            y=column,
            col=None,
            style='Domain',
            hue='Optimizer',
            height=6,
            facet_kws=dict(
                ylim=[0.0,1.0]
            )
        )
        add_async_task(
            hlp.relplot_and_save,
            outpath=os.path.join(output_dir, 'scores-optimizer-da_full_none-testaug_noise-epoch-{}.png'.format(column)),
            data=scores[scores['DA'].isin(['full', 'none']) & (scores['test_augmentation'].str.contains('noise'))],
            kind='line',
            x='Epoch',
            y=column,
            col='Data Aug.',
            style='test_augmentation',
            hue='Optimizer',
            height=6,
        )
        add_async_task(
            hlp.relplot_and_save,
            outpath=os.path.join(output_dir, 'scores-bn-da_full_none-epoch-{}.png'.format(column)),
            data=scores[scores['DA'].isin(['full', 'none'])],
            kind='line',
            x='Epoch',
            y=column,
            col='Optimizer',
            row='Data Aug.',
            style='Domain',
            hue='Normalization',
            height=6,
        )
        add_async_task(
            hlp.relplot_and_save,
            outpath=os.path.join(output_dir, 'scores-bn-single-da_full_none-epoch-{}.png'.format(column)),
            data=scores[scores['DA'].isin(['full', 'none'])],
            kind='line',
            x='Epoch',
            y=column,
            #col='Optimizer',
            #row='Data Aug.',
            style='Domain',
            hue='Normalization',
            height=6,
        )
        add_async_task(
            hlp.relplot_and_save,
            outpath=os.path.join(output_dir, 'scores-bn-Optimizer-da_full_none-epoch-{}.png'.format(column)),
            data=scores[scores['DA'].isin(['full', 'none'])],
            kind='line',
            x='Epoch',
            y=column,
            col='Optimizer',
            style='Domain',
            hue='Normalization',
            height=6,
        )
        add_async_task(
            hlp.relplot_and_save,
            outpath=os.path.join(output_dir, 'scores-bn-Optimizer-Domain-Testaug-da_full_none-epoch-{}.png'.format(column)),
            data=scores[scores['DA'].isin(['full', 'none'])],
            kind='line',
            x='Epoch',
            y=column,
            col='Optimizer',
            row='test_augmentation',
            style='Domain',
            hue='Normalization',
            height=6,
        )
        add_async_task(
            hlp.relplot_and_save,
            outpath=os.path.join(output_dir, 'scores-bn-Data-Aug-da_full_none-epoch-{}.png'.format(column)),
            data=scores[scores['DA'].isin(['full', 'none'])],
            kind='line',
            x='Epoch',
            y=column,
            col='Data Aug.',
            style='Domain',
            hue='Normalization',
            height=6,
        )
    
    join_async_tasks()

    for name, scores_f in {
        'DA full none only': scores[scores['DA'].isin(['full', 'none'])],
        'DA all': scores[~scores['bn']]
    }.items():
        for groupby in [
            ['domain_val_testaug', 'DA'],
            ['domain_val_testaug', 'optimizer'],
            ['domain_val_testaug', 'optimizer', 'DA'],
            ['domain_val_testaug', 'bn'],
            ['domain_val_testaug', 'bn', 'DA'],
            ['domain_val_testaug', 'optimizer', 'bn'],
            ['domain_val_testaug', 'optimizer', 'bn', 'DA'],
            
        ]:
            print(name, groupby)
            print(
                scores_f.groupby(groupby)[columns].agg(['mean', 'std', 'count']).round(2).to_string()
            )


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
    'philips15',
    'philips3'
]
epochs = [10,20,30,40,80,120,200,280,360]

#plot_scores_full_small_compare(task, trainers, folds, epochs)
plot_scores(task, trainers, folds, epochs)