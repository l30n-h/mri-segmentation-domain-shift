#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import numpy as np
import torch
from torch import nn

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, save_pickle
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.data_augmentation.data_augmentation_noDA import get_no_augmentation
from nnunet.training.dataloading.dataset_loading import DataLoader3D, DataLoader2D, unpack_dataset
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.nd_softmax import softmax_helper


class nnUNetTrainerV2_MA(nnUNetTrainerV2):

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.deep_supervision = True

    def wrap__loss_for_deep_supervision(self):
        # we need to know the number of outputs of the network
        net_numpool = len(self.net_num_pool_op_kernel_sizes)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
        weights[~mask] = 0
        weights = weights / weights.sum()
        self.ds_loss_weights = weights
        # now wrap the loss
        self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
    
    def get_tr_and_val_gen(self):
        print('with more augmentation generators')
        tr_gen, val_gen = get_moreDA_augmentation(
            self.dl_tr, self.dl_val,
            self.data_aug_params[
                'patch_size_for_spatialtransform'],
            self.data_aug_params,
            deep_supervision_scales=self.deep_supervision_scales,
            pin_memory=self.pin_memory,
            use_nondetMultiThreadedAugmenter=False
        )
        return tr_gen, val_gen

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            if self.deep_supervision:
                self.wrap__loss_for_deep_supervision()
            ################# END ###################

            self.folder_with_preprocessed_data = join(
                self.dataset_directory,
                self.plans['data_identifier'] + "_stage%d" % self.stage
            )

            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = self.get_tr_and_val_gen()

                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                        also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                        also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True



class nnUNetTrainerV2_MA_noDA(nnUNetTrainerV2_MA):
    def setup_DA_params(self):
        super().setup_DA_params()
        # important because we need to know in validation and inference that we did not mirror in training
        self.data_aug_params["do_mirror"] = False
        self.data_aug_params["mirror_axes"] = tuple()

    def get_basic_generators(self):
        print('no augmentation basic generators')
        self.load_dataset()
        self.do_split()

        if self.threeD:
            dl_tr = DataLoader3D(self.dataset_tr, self.patch_size, self.patch_size, self.batch_size,
                                 False, oversample_foreground_percent=self.oversample_foreground_percent
                                 , pad_mode="constant", pad_sides=self.pad_all_sides)
            dl_val = DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, False,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides)
        else:
            dl_tr = DataLoader2D(self.dataset_tr, self.patch_size, self.patch_size, self.batch_size,
            #                     transpose=self.plans.get('transpose_forward'),
                                 oversample_foreground_percent=self.oversample_foreground_percent
                                 , pad_mode="constant", pad_sides=self.pad_all_sides)
            dl_val = DataLoader2D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
            #                      transpose=self.plans.get('transpose_forward'),
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides)
        return dl_tr, dl_val

    def get_tr_and_val_gen(self):
        print('no augmentation generators')
        return get_no_augmentation(
            self.dl_tr,
            self.dl_val,
            params=self.data_aug_params,
            deep_supervision_scales=self.deep_supervision_scales,
            pin_memory=self.pin_memory
        )

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        if do_mirroring:
            print("WARNING! do_mirroring was True but we cannot do that because we trained without mirroring. "
                  "do_mirroring was set to False")
        do_mirroring = False

        return super().validate(
            do_mirroring=do_mirroring,
            use_sliding_window=use_sliding_window,
            step_size=step_size,
            save_softmax=save_softmax,
            use_gaussian=use_gaussian,
            overwrite=overwrite,
            validation_folder_name=validation_folder_name,
            debug=debug,
            all_in_gpu=all_in_gpu,
            segmentation_export_kwargs=segmentation_export_kwargs,
            run_postprocessing_on_folds=run_postprocessing_on_folds
        )

class nnUNetTrainerV2_MA_nogamma(nnUNetTrainerV2_MA):
    
    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params["do_gamma"] = False
        print(self.data_aug_params)


class nnUNetTrainerV2_MA_nomirror(nnUNetTrainerV2_MA):
    
    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params["do_mirror"] = False
        print(self.data_aug_params)
    
    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        if do_mirroring:
            print("WARNING! do_mirroring was True but we cannot do that because we trained without mirroring. "
                    "do_mirroring was set to False")
        do_mirroring = False
        self.network.do_ds = False
        ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                                save_softmax=save_softmax, use_gaussian=use_gaussian,
                                overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                                all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                                run_postprocessing_on_folds=run_postprocessing_on_folds)
        self.network.do_ds = ds
        return ret


class nnUNetTrainerV2_MA_norotation(nnUNetTrainerV2_MA):
    
    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params["do_rotation"] = False
        print(self.data_aug_params)


class nnUNetTrainerV2_MA_noscaling(nnUNetTrainerV2_MA):
    
    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params["do_scaling"] = False
        print(self.data_aug_params)



class nnUNetTrainerV2_MA_noscheduler_depth5_ep120(nnUNetTrainerV2_MA):

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)

        self.initial_lr = 1e-4
        self.weight_decay = 1e-2
        self.save_latest_only = False
        self.save_every = 10
        self.max_num_epochs = 120
        self.deep_supervision = True

    def load_plans_file(self):
        super().load_plans_file()
        self.plans['base_num_features'] = 16
        for plans_per_stage in self.plans['plans_per_stage'].values():
            plans_per_stage['batch_size'] = 64
            plans_per_stage['conv_kernel_sizes'] = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
            plans_per_stage['pool_op_kernel_sizes'] = [[2, 2], [2, 2], [2, 2], [2, 1]]
            plans_per_stage['num_pool_per_axis'] = [4, 3]
            print(plans_per_stage)
        return self.plans

    def initialize_optimizer_and_scheduler(self):
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay
        )
        self.lr_scheduler = None
        print(self.optimizer)
        print(self.lr_scheduler)


class nnUNetTrainerV2_MA_noscheduler_depth7_bf24_ep360(nnUNetTrainerV2_MA):

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)

        self.initial_lr = 1e-4
        self.weight_decay = 1e-2
        self.save_latest_only = False
        self.save_every = 10
        self.max_num_epochs = 360
        self.deep_supervision = True

    def load_plans_file(self):
        super().load_plans_file()
        self.plans['base_num_features'] = 24
        for plans_per_stage in self.plans['plans_per_stage'].values():
            plans_per_stage['batch_size'] = 64
            plans_per_stage['conv_kernel_sizes'] = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
            plans_per_stage['pool_op_kernel_sizes'] = [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 1]]
            plans_per_stage['num_pool_per_axis'] = [6, 5]
            print(plans_per_stage)
        return self.plans

    def initialize_optimizer_and_scheduler(self):
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay
        )
        self.lr_scheduler = None
        print(self.optimizer)
        print(self.lr_scheduler)


class nnUNetTrainerV2_MA_SGD(nnUNetTrainerV2_MA):

    def initialize_optimizer_and_scheduler(self):
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True
        )
        self.lr_scheduler = None
        print(self.optimizer)
        print(self.lr_scheduler)


class nnUNetTrainerV2_MA_wd0(nnUNetTrainerV2_MA):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.weight_decay = 0.0


class nnUNetTrainerV2_MA_bn(nnUNetTrainerV2_MA):
    def initialize_network(self):
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.BatchNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.BatchNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(
            self.num_input_channels,
            self.base_num_features,
            self.num_classes,
            len(self.net_num_pool_op_kernel_sizes),
            self.conv_per_stage,
            2,
            conv_op,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            net_nonlin,
            net_nonlin_kwargs,
            True,
            False,
            lambda x: x,
            InitWeights_He(1e-2),
            self.net_num_pool_op_kernel_sizes,
            self.net_conv_kernel_sizes,
            False,
            True,
            True
        )
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper




class nnUNetTrainerV2_MA_noscheduler_depth5_ep120_noDA(
    nnUNetTrainerV2_MA_noscheduler_depth5_ep120,
    nnUNetTrainerV2_MA_noDA
):
    pass

class nnUNetTrainerV2_MA_noscheduler_depth5_SGD_ep120(
    nnUNetTrainerV2_MA_SGD,
    nnUNetTrainerV2_MA_noscheduler_depth5_ep120
):
    pass

class nnUNetTrainerV2_MA_noscheduler_depth5_SGD_ep120_noDA(
    nnUNetTrainerV2_MA_noscheduler_depth5_SGD_ep120,
    nnUNetTrainerV2_MA_noDA
):
    pass




class nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120(
    nnUNetTrainerV2_MA_wd0,
    nnUNetTrainerV2_MA_noscheduler_depth5_ep120
):
    pass


class nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120_noDA(
    nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120,
    nnUNetTrainerV2_MA_noDA
):
    pass

class nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120_nogamma(
    nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120,
    nnUNetTrainerV2_MA_nogamma
):
    pass

class nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120_nomirror(
    nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120,
    nnUNetTrainerV2_MA_nomirror
):
    pass

class nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120_norotation(
    nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120,
    nnUNetTrainerV2_MA_norotation
):
    pass

class nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120_noscaling(
    nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120,
    nnUNetTrainerV2_MA_noscaling
):
    pass



class nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120(
    nnUNetTrainerV2_MA_SGD,
    nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120
):
    pass


class nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120_noDA(
    nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120,
    nnUNetTrainerV2_MA_noDA
):
    pass

class nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120_nogamma(
    nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120,
    nnUNetTrainerV2_MA_nogamma
):
    pass

class nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120_nomirror(
    nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120,
    nnUNetTrainerV2_MA_nomirror
):
    pass

class nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120_norotation(
    nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120,
    nnUNetTrainerV2_MA_norotation
):
    pass

class nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120_noscaling(
    nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120,
    nnUNetTrainerV2_MA_noscaling
):
    pass






class nnUNetTrainerV2_MA_noscheduler_depth5_wd0_bn_ep120(
    nnUNetTrainerV2_MA_bn,
    nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120,
):
    pass

class nnUNetTrainerV2_MA_noscheduler_depth5_wd0_bn_ep120_noDA(
    nnUNetTrainerV2_MA_noscheduler_depth5_wd0_bn_ep120,
    nnUNetTrainerV2_MA_noDA
):
    pass

class nnUNetTrainerV2_MA_noscheduler_depth5_wd0_bn_SGD_ep120(
    nnUNetTrainerV2_MA_bn,
    nnUNetTrainerV2_MA_noscheduler_depth5_wd0_SGD_ep120
):
    pass

class nnUNetTrainerV2_MA_noscheduler_depth5_wd0_bn_SGD_ep120_noDA(
    nnUNetTrainerV2_MA_noscheduler_depth5_wd0_bn_SGD_ep120,
    nnUNetTrainerV2_MA_noDA
):
    pass