import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, save_pickle
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from torch import nn

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