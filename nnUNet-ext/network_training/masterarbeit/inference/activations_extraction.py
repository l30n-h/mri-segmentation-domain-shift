import torch
import os

def is_module_parametrized(module):
    ps = list(module.named_parameters(recurse=False))
    return len(ps) > 0

def is_module_tracked(name, module):
    return is_module_parametrized(module)


def extract_activations_mean(name, activations, slices_and_tiles_count, activations_dict):
    if slices_and_tiles_count % 64 != 0:
        return activations_dict
    data = activations_dict.setdefault(name, [])
    data.append(
        activations[0].flatten(1).mean(dim=1)
    )
    return activations_dict

def extract_activations_full(name, activations, slices_and_tiles_count, activations_dict):
    if slices_and_tiles_count % 64 != 0:
        return activations_dict
    data = activations_dict.setdefault(name, [])
    data.append(
        activations[0].clone()
    )
    return activations_dict


def create_activations_extractor_from_env():
    if os.getenv('MA_USE_TEST_HOOKS', 'FALSE').lower() != 'true':
        return activations_extractor_dummy()
    return activations_extractor(
        extract_activations=extract_activations_full,
        is_module_tracked=is_module_tracked,
        merge_activations_dict=get_activations_dict_fs_friendly
    )

def reduce_batchsize_to_4(activations):
    #TODO heuristic to choose 4 slices/tiles that are mostly centered
    start = 1 if activations.shape[0] > 4 else 0
    step = activations.shape[0] // 4
    return activations[start::step].clone()

def get_activations_dict_fs_friendly(activations_dict):
    return dict(map(
        lambda item: (
            item[0],
            reduce_batchsize_to_4(
                torch.stack(item[1])
            ).cpu()
        ),
        activations_dict.items()
    ))

def save_activations_dict(activations_dict, output_filename):
    if os.getenv('MA_USE_TEST_HOOKS', 'FALSE').lower() != 'true':
        return

    output_dir = os.path.join(
        os.path.dirname(output_filename),
        'activations'
    )
    filename = os.path.basename(output_filename)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(
        activations_dict,
        os.path.join(output_dir, filename + "_activations.pkl")
    )


class activations_extractor():
    def __init__(
        self,
        extract_activations=extract_activations_full,
        is_module_tracked=is_module_tracked,
        merge_activations_dict=lambda x: x
    ):
        self.hooks = {}
        self.activations_dict = {}
        self.slices_and_tiles_count = -1
        self.extract_activations = extract_activations
        self.is_module_tracked = is_module_tracked
        self.merge_activations_dict = merge_activations_dict

    def __del__(self):
        self.reset_activations_dict()
        self.clear_hooks()

    def reset_activations_dict(self):
        self.activations_dict = {}
        self.slices_and_tiles_count = -1

    def get_activations_dict(self):
        return self.merge_activations_dict(self.activations_dict)
    
    def is_active(self):
        return len(self.hooks)
    
    def set_trainer(self, trainer):
        self.reinitialize_hooks(
            trainer=trainer,
            input_channels=trainer.num_input_channels,
            is_module_tracked=self.is_module_tracked,
            extract_activations=self.extract_activations
        )
        self.reset_activations_dict()

    def reinitialize_hooks(self, trainer, input_channels, extract_activations=extract_activations_full, is_module_tracked=is_module_tracked):
        self.clear_hooks()
        act_ext = self

        def forward_pre_hook_input_layer(module, input):
            act_ext.slices_and_tiles_count += 1

            act_ext.activations_dict = extract_activations(
                name='input',
                activations=input[0][:,:input_channels,:,:],
                slices_and_tiles_count=act_ext.slices_and_tiles_count,
                activations_dict=act_ext.activations_dict
            )
            act_ext.activations_dict = extract_activations(
                name='gt',
                activations=input[0][:,input_channels:,:,:],
                slices_and_tiles_count=act_ext.slices_and_tiles_count,
                activations_dict=act_ext.activations_dict
            )
            
            return tuple(map(lambda i: i[:,:input_channels,:,:], input))

        def get_forward_hook_fn(name, module):
            def forward_hook(module, input, output):
                act_ext.activations_dict = extract_activations(
                    name=name,
                    activations=output,
                    slices_and_tiles_count=act_ext.slices_and_tiles_count,
                    activations_dict=act_ext.activations_dict
                )
                return None
            return forward_hook

        
        self.hooks['forward_pre_hook_input_layer'] = trainer.network.register_forward_pre_hook(
            forward_pre_hook_input_layer
        )
        for name, module in trainer.network.named_modules():
            if is_module_tracked(name, module):
                self.hooks['forward_hook_' + name] = module.register_forward_hook(
                    get_forward_hook_fn(name, module)
                )

    def clear_hooks(self):
        for hook in self.hooks.values():
            hook.remove()
        self.hooks = {}

class activations_extractor_dummy(activations_extractor):
    def set_trainer(self, trainer):
        pass
    def reinitialize_hooks(self, trainer, input_channels, extract_activations=extract_activations_full, is_module_tracked=is_module_tracked):
        pass
