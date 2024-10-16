import torch
from torch import nn

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig

class variableencoder(nn.Module):
    def __init__(self, config):
        super(variableencoder, self).__init__()
        self.fc1 = nn.Linear(config.vencoder.input_size, config.vencoder.hidden_size)
        self.fc2 = nn.Linear(config.vencoder.hidden_size, 2*config.vencoder.hidden_size)
        self.fc3 = nn.Linear(2*config.vencoder.hidden_size, config.vencoder.output_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
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
    sample = torch.rand(1,4)
    model = variableencoder(config)
    output = model(sample)
    print(output.shape)