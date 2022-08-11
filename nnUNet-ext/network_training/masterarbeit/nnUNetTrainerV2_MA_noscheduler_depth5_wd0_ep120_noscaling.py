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


from .nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120 import nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120

class nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120_noscaling(nnUNetTrainerV2_MA_noscheduler_depth5_wd0_ep120):
    
    def setup_DA_params(self):
        super().setup_DA_params()

        self.data_aug_params["do_scaling"] = False

        print(self.data_aug_params)