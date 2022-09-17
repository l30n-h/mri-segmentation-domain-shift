import argparse
import torch
import numpy as np
from multiprocessing import Pool
import json

from batchgenerators.utilities.file_and_folder_operations import join, isdir, isfile, maybe_mkdir_p, save_json, load_pickle, subfiles
from nnunet.evaluation.evaluator import aggregate_scores
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from nnunet.paths import default_plans_identifier, network_training_output_dir, default_cascade_trainer, default_trainer
from nnunet.training.model_restore import load_model_and_checkpoint_files
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name

import nnunet.training.network_training.masterarbeit.inference.activations_extraction as act_ext
from nnunet.training.network_training.masterarbeit.inference.SDiceNiftiEvaluator import SDiceNiftiEvaluator

def predict_preprocessed_ram(
    trainer,
    activations_extractor=None,
    name_paths_dict: dict = {}, # { name: { 'preprocessed_file': '/abc/xyz.npz' } }
    do_mirroring: bool = True,
    use_sliding_window: bool = True,
    step_size: float = 0.5,
    use_gaussian: bool = True,
    all_in_gpu: bool = False,
):
    assert trainer.network.training == False, "must set mode to eval"
    assert trainer.was_initialized, "must initialize, ideally with checkpoint (or train first)"

    if do_mirroring:
        if not trainer.data_aug_params['do_mirror']:
            raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
        mirror_axes = trainer.data_aug_params['mirror_axes']
    else:
        mirror_axes = ()

    if activations_extractor is not None:
        activations_extractor.set_trainer(trainer)

    for name, paths in name_paths_dict.items():
        data = np.load(paths['preprocessed_file'])['data']

        data[-1][data[-1] == -1] = 0
        if activations_extractor is None or not activations_extractor.is_active():
            data = data[:-1]

        if activations_extractor is not None:
            activations_extractor.reset_activations_dict()

        softmax_pred = trainer.predict_preprocessed_data_return_seg_and_softmax(
            data,
            do_mirroring=do_mirroring,
            mirror_axes=mirror_axes,
            use_sliding_window=use_sliding_window,
            step_size=step_size,
            use_gaussian=use_gaussian,
            all_in_gpu=all_in_gpu,
            mixed_precision=trainer.fp16
        )[1]

        activations_dict = {}
        if activations_extractor is not None:
            activations_dict = activations_extractor.get_activations_dict()

        softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in trainer.transpose_backward])
        yield name, softmax_pred, activations_dict

def predict_preprocessed_fs(
    trainer,
    output_folder,
    name_paths_dict: dict = {}, # { name: { 'preprocessed_file': '/abc/xyz.npz', 'gt_file': '/abc/xyz.nii.gz', 'properties_file': '/abc/xyz.pkl' } }
    do_mirroring: bool = True,
    use_sliding_window: bool = True,
    step_size: float = 0.5,
    save_softmax: bool = True,
    use_gaussian: bool = True,
    overwrite: bool = True,
    folder_name_prediction: str = 'prediction_raw',
    debug: bool = False,
    all_in_gpu: bool = False,
    segmentation_export_kwargs: dict = None,
    num_threads: int = 8,
):
    """
    if debug=True then the temporary files generated for postprocessing determination will be kept
    """

    current_mode = trainer.network.training
    trainer.network.eval()

    assert trainer.was_initialized, "must initialize, ideally with checkpoint (or train first)"

    if segmentation_export_kwargs is None:
        if 'segmentation_export_params' in trainer.plans.keys():
            force_separate_z = trainer.plans['segmentation_export_params']['force_separate_z']
            interpolation_order = trainer.plans['segmentation_export_params']['interpolation_order']
            interpolation_order_z = trainer.plans['segmentation_export_params']['interpolation_order_z']
        else:
            force_separate_z = None
            interpolation_order = 1
            interpolation_order_z = 0
    else:
        force_separate_z = segmentation_export_kwargs['force_separate_z']
        interpolation_order = segmentation_export_kwargs['interpolation_order']
        interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

    # predictions as they come from the network go here
    output_folder_prediction = join(output_folder, folder_name_prediction)
    maybe_mkdir_p(output_folder_prediction)
    # this is for debug purposes
    my_input_args = {
        'do_mirroring': do_mirroring,
        'use_sliding_window': use_sliding_window,
        'step_size': step_size,
        'save_softmax': save_softmax,
        'use_gaussian': use_gaussian,
        'overwrite': overwrite,
        'folder_name_prediction': folder_name_prediction,
        'debug': debug,
        'all_in_gpu': all_in_gpu,
        'segmentation_export_kwargs': segmentation_export_kwargs,
    }
    save_json(my_input_args, join(output_folder_prediction, "prediction_args.json"))

    pred_gt_tuples = []

    export_pool = Pool(num_threads)
    results = []

    for name, softmax_pred, activations_dict in predict_preprocessed_ram(
        trainer=trainer,
        activations_extractor=act_ext.create_activations_extractor_from_env(),
        name_paths_dict=name_paths_dict,
        do_mirroring=do_mirroring,
        use_sliding_window=use_sliding_window,
        step_size=step_size,
        use_gaussian=use_gaussian,
        all_in_gpu=all_in_gpu
    ):
        properties = load_pickle(name_paths_dict[name]['properties_file'])
        if overwrite or (not isfile(join(output_folder_prediction, name + ".nii.gz"))) or (save_softmax and not isfile(join(output_folder_prediction, fname + ".npz"))):
            if save_softmax:
                softmax_fname = join(output_folder_prediction, name + ".npz")
            else:
                softmax_fname = None

            """There is a problem with python process communication that prevents us from communicating objects
            larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
            communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long
            enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
            patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will
            then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
            filename or np.ndarray and will handle this automatically"""
            if np.prod(softmax_pred.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
                np.save(join(output_folder_prediction, name + ".npy"), softmax_pred)
                softmax_pred = join(output_folder_prediction, name + ".npy")

            results.append(
                export_pool.starmap_async(
                    save_segmentation_nifti_from_softmax,
                    (
                        (softmax_pred, join(output_folder_prediction, name + ".nii.gz"),
                        properties, interpolation_order, trainer.regions_class_order,
                        None, None,
                        softmax_fname, None, force_separate_z,
                        interpolation_order_z),
                    )
                )
            )

            results.append(export_pool.apply_async(
                act_ext.save_activations_dict, (
                    activations_dict,
                    join(output_folder_prediction, name)
                )
            ))

        pred_gt_tuples.append([
            join(output_folder_prediction, name + ".nii.gz"),
            join(name_paths_dict[name]['gt_file'])
        ])

    _ = [i.get() for i in results]
    trainer.print_to_log_file("finished prediction")

    # evaluate raw predictions
    trainer.print_to_log_file("evaluation of raw predictions")
    task = trainer.dataset_directory.split("/")[-1]
    job_name = trainer.experiment_name
    _ = aggregate_scores(
        pred_gt_tuples,
        evaluator=SDiceNiftiEvaluator,
        labels=list(range(trainer.num_classes)),
        json_output_file=join(output_folder_prediction, "summary.json"),
        json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
        json_author="Fabian",
        json_task=task, num_threads=num_threads
    )

    trainer.network.train(current_mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", '--name_paths_dict', required=True, help="Path to a json object file. Must contain the full path to the preprocessed, gt, and properties file for each patient. e.g.: { key: { 'preprocessed_file': '/abc/xyz.npz', 'gt_file': '/abc/xyz.nii.gz', 'properties_file': '/abc/xyz.pkl' }, ... }")
    parser.add_argument('-o', "--output_folder", required=True, help="folder for saving predictions")
    parser.add_argument('-t', '--task_name', help='task name or task ID, required.',
                        default=default_plans_identifier, required=True)

    parser.add_argument('-tr', '--trainer_class_name',
                        help='Name of the nnUNetTrainer used for 2D U-Net, full resolution 3D U-Net and low resolution '
                             'U-Net. The default is %s. If you are running inference with the cascade and the folder '
                             'pointed to by --lowres_segmentations does not contain the segmentation maps generated by '
                             'the low resolution U-Net then the low resolution segmentation maps will be automatically '
                             'generated. For this case, make sure to set the trainer class here that matches your '
                             '--cascade_trainer_class_name (this part can be ignored if defaults are used).'
                             % default_trainer,
                        required=False,
                        default=default_trainer)
    parser.add_argument('-ctr', '--cascade_trainer_class_name',
                        help="Trainer class name used for predicting the 3D full resolution U-Net part of the cascade."
                             "Default is %s" % default_cascade_trainer, required=False,
                        default=default_cascade_trainer)

    parser.add_argument('-m', '--model', help="2d, 3d_lowres, 3d_fullres or 3d_cascade_fullres. Default: 3d_fullres",
                        default="3d_fullres", required=False)

    parser.add_argument('-p', '--plans_identifier', help='do not touch this unless you know what you are doing',
                        default=default_plans_identifier, required=False)

    parser.add_argument('-f', '--folds', nargs='+', default='None',
                        help="folds to use for prediction. Default is None which means that folds will be detected "
                             "automatically in the model output folder")

    parser.add_argument('-z', '--save_npz', required=False, action='store_true',
                        help="use this if you want to ensemble these predictions with those of other models. Softmax "
                             "probabilities will be saved as compressed numpy arrays in output_folder and can be "
                             "merged between output_folders with nnUNet_ensemble_predictions")

    parser.add_argument("--num_threads_nifti_save", required=False, default=2, type=int, help=
    "Determines many background processes will be used for segmentation export. Reduce this if you "
    "run into out of memory (RAM) problems. Default: 2")

    parser.add_argument("--disable_tta", required=False, default=False, action="store_true",
                        help="set this flag to disable test time data augmentation via mirroring. Speeds up inference "
                             "by roughly factor 4 (2D) or 8 (3D)")

    parser.add_argument("--overwrite_existing", required=False, default=False, action="store_true",
                        help="Set this flag if the target folder contains predictions that you would like to overwrite")

    parser.add_argument("--all_in_gpu", type=str, default="None", required=False, help="can be None, False or True. "
                                                                                       "Do not touch.")
    parser.add_argument("--step_size", type=float, default=0.5, required=False, help="don't touch")
    parser.add_argument('-chk',
                        help='checkpoint name, default: model_final_checkpoint',
                        required=False,
                        default='model_final_checkpoint')
    parser.add_argument('--disable_mixed_precision', default=False, action='store_true', required=False,
                        help='Predictions are done with mixed precision by default. This improves speed and reduces '
                             'the required vram. If you want to disable mixed precision you can set this flag. Note '
                             'that yhis is not recommended (mixed precision is ~2x faster!)')

    args = parser.parse_args()
    output_folder = args.output_folder
    folds = args.folds
    save_npz = args.save_npz
    num_threads_nifti_save = args.num_threads_nifti_save
    disable_tta = args.disable_tta
    step_size = args.step_size
    overwrite_existing = args.overwrite_existing
    all_in_gpu = args.all_in_gpu
    model = args.model
    trainer_class_name = args.trainer_class_name
    cascade_trainer_class_name = args.cascade_trainer_class_name

    task_name = args.task_name

    if not task_name.startswith("Task"):
        task_id = int(task_name)
        task_name = convert_id_to_task_name(task_id)

    assert model in ["2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"], "-m must be 2d, 3d_lowres, 3d_fullres or 3d_cascade_fullres"

    if isinstance(folds, list):
        if folds[0] == 'all' and len(folds) == 1:
            pass
        else:
            folds = [int(i) for i in folds]
    elif folds == "None":
        folds = None
    else:
        raise ValueError("Unexpected value for argument folds")

    assert all_in_gpu in ['None', 'False', 'True']
    if all_in_gpu == "None":
        #all_in_gpu = None
        all_in_gpu = False
    elif all_in_gpu == "True":
        all_in_gpu = True
    elif all_in_gpu == "False":
        all_in_gpu = False

    if model == "3d_cascade_fullres":
        trainer = cascade_trainer_class_name
    else:
        trainer = trainer_class_name

    model_folder_name = join(
        network_training_output_dir,
        model,
        task_name,
        trainer + "__" + args.plans_identifier
    )
    print("using model stored in ", model_folder_name)
    assert isdir(model_folder_name), "model output folder not found. Expected: %s" % model_folder_name


    print("emptying cuda cache")
    torch.cuda.empty_cache()

    print("loading parameters for folds,", folds)
    trainer, params = load_model_and_checkpoint_files(
        model_folder_name,
        folds,
        mixed_precision=not args.disable_mixed_precision,
        checkpoint_name=args.chk
    )
    trainer.load_checkpoint_ram(params[0], False)
    
    with open(args.name_paths_dict) as f:
        name_paths_dict = json.load(f)
        print(name_paths_dict.keys())

    predict_preprocessed_fs(
        trainer,
        output_folder,
        name_paths_dict=name_paths_dict,
        do_mirroring=not disable_tta,
        use_sliding_window=True,
        step_size=step_size,
        save_softmax=save_npz,
        use_gaussian=True,
        overwrite=overwrite_existing,
        folder_name_prediction='prediction_raw',
        debug=False,
        all_in_gpu=all_in_gpu,
        segmentation_export_kwargs=None,
        num_threads=num_threads_nifti_save,
    )


if __name__ == "__main__":
    main()