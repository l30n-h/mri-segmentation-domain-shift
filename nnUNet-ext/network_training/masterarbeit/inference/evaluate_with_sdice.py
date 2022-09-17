import argparse
import glob
import re
import os

from nnunet.evaluation.evaluator import aggregate_scores

from nnunet.training.network_training.masterarbeit.inference.SDiceNiftiEvaluator import SDiceNiftiEvaluator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', help='Must contain all modalities for each patient in the correct'
                                                     ' order (same as training). Files must be named '
                                                     'CASENAME_XXXX.nii.gz where XXXX is the modality '
                                                     'identifier (0000, 0001, etc)', required=True)
    parser.add_argument('-l', '--label_folder', help='Must contain all labels for each patient in the correct'
                                                     ' order (same as training). Files must be named '
                                                     'CASENAME.nii.gz', required=True)
    parser.add_argument('-o', '--output_file', required=True, help='json file for saving evaluated scores')
    parser.add_argument('-c', '--num_classes', required=True, default=2, type=int, help='Determines number of predicted classes. Default: 2')
    parser.add_argument('--num_threads', required=False, default=6, type=int, help='Determines how many background processes will be used for evaluation. Default: 6')

    args = parser.parse_args()
    input_folder = args.input_folder
    label_folder = args.label_folder
    output_file = args.output_file
    num_classes = args.num_classes
    num_threads = args.num_threads

    regex = re.compile(r'_\d\d\d\d\.nii.gz$')
    replacement = '.nii.gz'

    pred_gt_tuples = list(map(
        lambda path: (
            path,
            os.path.join(
                label_folder,
                re.sub(regex, replacement, os.path.basename(path))
            )
        ),
        glob.iglob(os.path.join(input_folder, '*.nii.gz'))
    ))
    print('number of files to evaluate:', len(pred_gt_tuples))

    _ = aggregate_scores(
        pred_gt_tuples,
        evaluator=SDiceNiftiEvaluator,
        labels=list(range(num_classes)),
        json_output_file=output_file,
        json_name='',
        json_author='',
        json_task='',
        num_threads=num_threads
    )

if __name__ == '__main__':
    main()