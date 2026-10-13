from torch import nn


# Encoder for image
class VisionTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ):
        super().__init__()
        self.input_resolution = input_resolution


# Encoder for astro physical variables
class PhysicalVariableEncoder(nn.Module):
    pass
