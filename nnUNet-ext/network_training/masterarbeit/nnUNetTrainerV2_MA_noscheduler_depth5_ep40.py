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


import torch
from .base.nnUNetTrainerV2_MA import nnUNetTrainerV2_MA
from .base.scheduler import LinearWarmupCosineAnnealingLR
from .base.lab_losses import LovaszFocal

class nnUNetTrainerV2_MA_noscheduler_depth5_ep40(nnUNetTrainerV2_MA):

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)

        self.initial_lr = 1e-4
        self.weight_decay = 1e-2
        self.save_latest_only = False
        self.save_every = 10
        self.max_num_epochs = 40
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