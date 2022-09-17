import nibabel as nb
import nibabel.processing as nbp
import glob
import os

import numpy as np
import pickle
import torch
import torchvision
import torchvision.transforms.functional as tfv


def augment_gaussian_noises(image, target, noise_mean=0.0, noise_std=0.0):
    return (
        torch.clamp(
            image + torch.randn(image.size()) * noise_std + noise_mean,
            0.0,
            1.0
        ),
        target
    )

def augment_sharpnesses(image, target, sharpness_factor=1.0):
    return (
        tfv.adjust_sharpness(image[:,None,:], sharpness_factor)[:,0,:],
        target
    )


def min_max_scaling(data, to_min, to_max):
    from_min = data.min()
    from_max = data.max()
    return (
        (data - from_min) / (from_max - from_min) * (to_max - to_min) + to_min,
        from_min,
        from_max
    )

def get_augmentation_name(params):
    return "_".join(map(
            lambda key: '{}={}'.format(key, params[key]),
            sorted(params.keys())
    ))


def create_augmentations(augmentations, input_base_path, output_base_path):
    EXT = '.nii.gz'
    for path in glob.iglob(os.path.join(BASE_PATH, 'labelsTr', '*{}'.format(EXT))):
        filename = os.path.basename(path).replace(EXT, '')
        filename_img = '{}_0000{}'.format(filename, EXT)
        filename_seg = '{}{}'.format(filename, EXT)
        print(filename)

        img = nb.load(os.path.join(BASE_PATH, 'imagesTr', filename_img))
        seg = nb.load(path)
        header_img = img.header.copy()
        header_seg = seg.header.copy()
        voxel_spacing = header_img['pixdim'][1:4].copy()
        
        img_resampled = img
        seg_resampled = seg
        # img_resampled = nbp.resample_from_to(img, [np.round(np.array(img.shape) * voxel_spacing).astype(int), np.eye(*img.affine.shape)])
        # seg_resampled = nbp.resample_from_to(seg, [np.round(np.array(seg.shape) * voxel_spacing).astype(int), np.eye(*seg.affine.shape)])
        data_img = torch.from_numpy(img_resampled.get_fdata().astype(np.float32)).permute(2,0,1)
        data_seg = torch.from_numpy(seg_resampled.get_fdata().astype(np.float32)).permute(2,0,1)
        
        data_img, old_min_img, old_max_img = min_max_scaling(data_img, 0.0, 1.0)
        data_seg, old_min_seg, old_max_seg = min_max_scaling(data_seg, 0.0, 1.0)

        for aug in augmentations:
            name = get_augmentation_name(aug['params'])
            data_img_aug, data_seg_aug = aug['fn'](
                data_img,
                data_seg,
                **aug['params']
            )
            data_img_aug = min_max_scaling(data_img_aug, old_min_img, old_max_img)[0]
            data_seg_aug = min_max_scaling(data_seg_aug, old_min_seg, old_max_seg)[0]
            
            # img_aug = nbp.resample_from_to(nb.Nifti1Image(data_img_aug.permute(1,2,0).numpy(), img_resampled.affine), [img.shape, img.affine])
            # seg_aug = nbp.resample_from_to(nb.Nifti1Image(data_seg_aug.permute(1,2,0).numpy(), seg_resampled.affine), [seg.shape, seg.affine])
            # data_img_aug = torch.from_numpy(img_aug.get_fdata()).permute(2,0,1)
            # data_seg_aug = torch.from_numpy(seg_aug.get_fdata()).permute(2,0,1)

            dtype_img = np.float32 #img.get_data_dtype()
            dtype_seg = np.float32 #seg.get_data_dtype()
            img_aug = nb.Nifti1Image(data_img_aug.permute(1,2,0).numpy().astype(dtype_img), img.affine, header=header_img)
            seg_aug = nb.Nifti1Image(data_seg_aug.permute(1,2,0).numpy().astype(dtype_seg), seg.affine, header=header_seg)
            

            output_dir = os.path.join(BASE_PATH, 'augmented', name)
            img_output_dir = os.path.join(output_dir, 'imagesTs')
            seg_output_dir = os.path.join(output_dir, 'labelsTs')
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(img_output_dir, exist_ok=True)
            os.makedirs(seg_output_dir, exist_ok=True)
            nb.save(img_aug, os.path.join(img_output_dir, filename_img))
            nb.save(seg_aug, os.path.join(seg_output_dir, filename_seg))



def get_preprocessor_by_plans(plans):
    stage = 0
    import nnunet
    from nnunet.training.model_restore import recursive_find_python_class
    preprocessor_name = plans['preprocessor_name']
    preprocessor_class = recursive_find_python_class(
        [os.path.join(nnunet.__path__[0], "preprocessing")],
        preprocessor_name,
        current_module="nnunet.preprocessing"
    )
    assert preprocessor_class is not None, "Could not find preprocessor %s in nnunet.preprocessing" % preprocessor_name
    preprocessor = preprocessor_class(
        plans['normalization_schemes'],
        plans['use_mask_for_norm'],
        plans['transpose_forward'],
        plans['dataset_properties']['intensityproperties']
    )
    return preprocessor

def preprocess(plans, preprocessor, img_file, seg_file, out_file_without_ext):
    data, seg, properties = preprocessor.preprocess_test_case(
        [img_file],
        target_spacing=plans['plans_per_stage'][0]['current_spacing'],
        seg_file=seg_file,
        force_separate_z=None
    )
    all_data = np.vstack((data, seg)).astype(np.float32)

    np.savez_compressed(
        os.path.join('{}.npz'.format(out_file_without_ext)),
        data=all_data.astype(np.float32)
    )
    with open('{}.pkl'.format(out_file_without_ext), 'wb') as f:
        pickle.dump(properties, f)

def preprocess_augmented_files(plans, input_base_path):
    EXT='.nii.gz'
    preprocessor = get_preprocessor_by_plans(plans)
    for augmentation_dir in os.listdir(input_base_path):
        if augmentation_dir == 'noise_std=0.15':
            print("continue")
            continue
        augmentation_path = os.path.join(input_base_path, augmentation_dir)
        for file_path in glob.iglob(os.path.join(augmentation_path, 'labelsTs', '*.nii.gz')):
            filename_without_ext = os.path.basename(file_path).replace(EXT, '')
            preprocess_dir = os.path.join(augmentation_path, 'preprocessed')
            os.makedirs(preprocess_dir, exist_ok=True)
            preprocess(
                plans,
                preprocessor,
                img_file=os.path.join(augmentation_path, 'imagesTs', '{}_0000{}'.format(filename_without_ext, EXT)),
                seg_file=file_path,
                out_file_without_ext=os.path.join(preprocess_dir, filename_without_ext),
            )



AUGMENTATIONS = [
    {
        "fn": augment_gaussian_noises,
        "params": {
            "noise_std": 0.05
        }
    }, {
        "fn": augment_gaussian_noises,
        "params": {
            "noise_std": 0.10
        }
    }, {
        "fn": augment_gaussian_noises,
        "params": {
            "noise_std": 0.15
        }
    }, {
        "fn": augment_sharpnesses,
        "params": {
            "sharpness_factor": 0.5
        }
    }, {
        "fn": augment_sharpnesses,
        "params": {
            "sharpness_factor": 2.0
        }
    }
]

#BASE_PATH = 'data/nnUNet_raw/nnUNet_raw_data/Task601_cc359_all_training'
INPUT_BASE_PATH = 'archive/old/nnUNet-container/data/nnUNet_raw/nnUNet_raw_data/Task601_cc359_all_training/'
OUTPUT_BASE_PATH = os.path.join(INPUT_BASE_PATH, 'augmented')
plans = np.load('data/nnUNet_preprocessed/Task601_cc359_all_training/nnUNetPlansv2.1_plans_2D.pkl', allow_pickle=True)

#create_augmentations(AUGMENTATIONS, INPUT_BASE_PATH, OUTPUT_BASE_PATH)
#preprocess_augmented_files(plans, OUTPUT_BASE_PATH)