import pandas as pd
import itertools
import os
import json
import re
import io

import helpers as hlp



def load_json(path):
    with open(path) as f:
        return json.load(f)

def get_summary_scores(task, trainer, fold, epoch):
    try:
        data = load_json(
            os.path.join(
                hlp.get_testdata_dir(task, trainer, fold, epoch),
                'summary-small.json'
            )
        )
        def extract(data):
            # available_fields = [
            #     'Accuracy',
            #     'Dice',
            #     'False Discovery Rate',
            #     'False Negative Rate',
            #     'False Omission Rate',
            #     'False Positive Rate',
            #     'Jaccard',
            #     'Negative Predictive Value',
            #     'Precision',
            #     'Recall',
            #     'Total Positives Reference',
            #     'Total Positives Test',
            #     'True Negative Rate',
            # ]
            return {
                'id': re.search(r'/(CC\d{4})_', data['reference']).groups(1)[0],
                'trainer': trainer,
                'fold_train': fold,
                'epoch': epoch,
                **data['1']
            }
        return pd.read_json(
            io.StringIO(json.dumps(list(map(extract, data['results']['all']))))
        )
    except FileNotFoundError:
        return pd.DataFrame()

def compare_score(task, trainers, folds, epochs):
    scores_summary = pd.concat([ 
        get_summary_scores(
            task, trainer, fold, epoch
        ) for trainer, fold, epoch in itertools.product(trainers, folds, epochs)
    ]).reset_index(drop=True)
    scores = pd.concat([
        hlp.get_scores(
            task, trainer, fold, epoch
        ) for trainer, fold, epoch in itertools.product(trainers, folds, epochs)
    ]).reset_index(drop=True)

    join_on=['trainer', 'fold_train', 'epoch', 'id']
    joined = scores_summary.join(scores.set_index(join_on), on=join_on)[
        join_on + [
            'Dice',
            'dice_score',
            'sdice_score',
            'Jaccard',
            'iou_score'
        ]
    ]
    dif_dice = joined['Dice'] - joined['dice_score']
    dif_iou = joined['Jaccard'] - joined['iou_score']
    print(dif_dice.min())
    print(dif_dice.mean())
    print(dif_dice.abs().mean())
    print(dif_dice.max())
    print(dif_iou.min())
    print(dif_iou.mean())
    print(dif_iou.abs().mean())
    print(dif_iou.max())

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
        dice_score_mean=('dice_score', 'mean'),
        dice_score_std=('dice_score', 'std'),
        iou_score_mean=('iou_score', 'mean'),
        iou_score_std=('iou_score', 'std'),
        sdice_score_mean=('sdice_score', 'mean'),
        sdice_score_std=('sdice_score', 'std')
    ).reset_index()

    print(scores)
    #scores = scores[scores['is_validation']]

    
    scores['wd_bn'] = scores['wd'].apply(lambda x: 'wd=' + str(x)) + ' ' + scores['bn'].apply(lambda x: 'bn=' + str(x))
    scores['wd_DA'] = scores['wd'].apply(lambda x: 'wd=' + str(x)) + ' ' + scores['DA']
    scores['optimizer_wd_DA'] = scores['optimizer'] + ' ' + scores['wd_DA']
    scores['optimizer_DA'] = scores['optimizer'] + ' ' + scores['DA']
    scores['wd_bn_DA'] = scores['wd_bn'] + ' ' + scores['DA']
    scores['same_domain'] = scores['fold_train'] == scores['fold_test']
    scores['domain_val'] = scores['same_domain'].apply(lambda x: 'same' if x else 'other') + ' ' + scores['is_validation'].apply(lambda x: 'validation' if x else '')

    print(scores)

    output_dir = 'data/fig/scores'
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
            ci=None,
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
    'nnUNetTrainerV2_MA_noscheduler_depth5_SGD_ep120__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_SGD_ep120_noDA__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_ep120__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_ep120_noDA__nnUNetPlansv2.1',

    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120_noDA__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120_nogamma',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120_nomirror',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120_norotation',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120_noscaling',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120_noDA__nnUNetPlansv2.1',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120_nogamma',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120_nomirror',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120_norotation',
    'nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120_noscaling',

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
epochs = [10,20,30,40,80,120]

#compare_score(task, trainers, folds, epochs)
plot_scores(task, trainers, folds, epochs)