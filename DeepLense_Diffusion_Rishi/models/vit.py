import torch
from torch import nn

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from transformers import ViTConfig, ViTModel

class vit(nn.Module):
    
    def __init__(self, config):
        super(vit, self).__init__()

        config = ViTConfig(image_size=config.vit.image_size, patch_size=config.vit.patch_size, num_channels=config.vit.num_channels, hidden_size=config.vit.hidden_size, num_hidden_layers=config.vit.num_hid_layers, 
                           num_attention_heads=config.vit.num_attention_heads, intermediate_size=config.vit.intermediate_size, hidden_act=config.vit.hidden_act,
                           layer_norm_eps=config.vit.layer_norm_eps, qkv_bias=config.vit.qkv_bias)
        
        self.model = ViTModel(config)

    def forward(self, x):
        out = self.model(x)
        last_hidden_state = out.last_hidden_state[:,0]
        return last_hidden_state
    
if __name__ == '__main__':
    # Read the configuration
    config_name = "default"
    pipe = ConfigPipeline(
        [
            YamlConfig(
                "./clip_config.yaml", config_name='default', config_folder='cfg/'
            ),
            ArgparseConfig(infer_types=True, config_name=None, config_file=None),
            YamlConfig(config_folder='cfg/')
        ]
    )
    config = pipe.read_conf()
    sample = torch.rand(1,1,64,64)
    model = vit(config)
    output = model(sample)
    print(output.shape)
