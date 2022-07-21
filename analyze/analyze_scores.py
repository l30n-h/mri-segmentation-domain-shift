import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re

import helpers as hlp

def get_params_from_path(path):
    dirname = os.path.dirname(path)
    name = os.path.basename(dirname)
    match = re.search("^(.*?)-ep(\d+)-(.*)$", name)

    task = os.path.basename(os.path.dirname(dirname))
    trainer = match.group(1)
    epoch = int(match.group(2))
    fold_train = match.group(3)
    return task, trainer, fold_train, epoch

scores = pd.concat([
    hlp.get_scores(
        *get_params_from_path(path)
    ) for path in glob.iglob('archive/old/nnUNet-container/data/testout/Task601_cc359_all_training/*/scores.csv', recursive=False)
])

print(scores)
scores['trainer_short'] = scores['trainer'].apply(hlp.get_trainer_short)
scores['psize'] = 2#scores['epoch'] / 10
scores['fold_train_id'] = pd.factorize(scores['fold_train'], sort=True)[0]
column_color = 'fold_train_id'
scores = scores.sort_values(['id'])
scores = scores[scores['epoch'] == 40]

columns = [
    'dice_score',
    'sdice_score'
]
for column in columns:
    fig, axes = hlp.create_plot(
        scores,
        column_x='id',
        column_y=column,
        column_subplots='trainer_short',
        column_size='psize',
        column_color=column_color,
        ncols=1,
        figsize=(32, 24),
        lim_same_x=False,
        lim_same_y=True,
        lim_same_c=True,
        colormap='cool'
    )

    fig.savefig('data/fig/scores-{}.png'.format(column))
    plt.close(fig)