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
from .nnUNetTrainerV2_MA_noscheduler_depth5_ep120 import nnUNetTrainerV2_MA_noscheduler_depth5_ep120

class nnUNetTrainerV2_MA_noscheduler_depth5_SGD_ep120(nnUNetTrainerV2_MA_noscheduler_depth5_ep120):

    def initialize_optimizer_and_scheduler(self):
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True
        )
        self.lr_scheduler = None