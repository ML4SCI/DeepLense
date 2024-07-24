import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class patch_embedding(nn.Module):
    def __init__(
            self,
            image_size: Tuple[int, int],
            patch_size: Tuple[int, int],
            input_channels: int,
            patch_embedding_dim: int,
            norm_layer: nn.Module = None, 
            padding: bool = False,
            flatten: bool = True, # converts BCHW to B(C*H)W
            bias: bool = True
        ):
        super().__init__()
        assert isinstance(image_size, tuple),\
            'image_size must be a tuple describing image dimension'
        assert isinstance(patch_size, tuple),\
            'patch_size must be a tuple describing patch dimension'
        self.padding = padding
        self.flatten = flatten
        self.patch_size = patch_size
        if not padding:
            assert image_size[0]%patch_size[0] == 0, \
                f'image height {image_size[0]} is not divisible by patch height {patch_size[0]}'
            assert image_size[1]%patch_size[1] == 0, \
                f'image width {image_size[1]} is not divisible by patch width {patch_size[1]}'
            self.resolution = [image_size[0]//patch_size[0], image_size[1]//patch_size[1]]
        else:
            self.pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            self.pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            self.resolution = [(image_size[0]+self.pad_h)//patch_size[0], (image_size[1]+self.pad_w)//patch_size[1]]
        self.num_patches = (self.resolution[0])*(self.resolution[1])
        self.proj = nn.Conv2d(input_channels, patch_embedding_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm_layer = norm_layer(patch_embedding_dim) \
                                if norm_layer is not None else nn.Identity()
    def forward(self, x):
        if self.padding:
            x = F.pad(x, (0, self.pad_w, 0, self.pad_h))
        x = self.proj(x)
        # Hp, Wp = x.shape[2], x.shape[3]
        if self.flatten: # converts BCHW to B(C*H)W
            x = x.flatten(2).transpose(1, 2)
        if isinstance(self.norm_layer, nn.BatchNorm1d):
            x = self.norm_layer(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            x = self.norm_layer(x)
        return x

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution + batch norm"""
    return torch.nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True),
        nn.BatchNorm2d(out_planes)
    )
    
class ConvPatchEmbed(nn.Module):
    """Image to Patch Embedding using multiple convolutional layers"""

    def __init__(
                self,
                image_size: Tuple[int, int],
                patch_size: Tuple[int, int],
                input_channels: int,
                patch_embedding_dim: int,
                activation: nn.Module = nn.GELU,
                padding: bool = False,
                flatten: bool = True, # converts BCHW to B(C*H)W
                bias: bool = False
            ):
        super().__init__()
        if not isinstance(image_size, tuple):
            image_size = (image_size, image_size)
        if not isinstance(patch_size, tuple):
            patch_size = (patch_size, patch_size)
        self.padding = padding
        self.flatten = flatten
        self.patch_size = patch_size
        if not padding:
            assert image_size[0]%patch_size[0] == 0, \
                f'image height {image_size[0]} is not divisible by patch height {patch_size[0]}'
            assert image_size[1]%patch_size[1] == 0, \
                f'image width {image_size[1]} is not divisible by patch width {patch_size[1]}'
            self.resolution = [image_size[0]//patch_size[0], image_size[1]//patch_size[1]]
        else:
            self.pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            self.pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            self.resolution = [(image_size[0]+self.pad_h)//patch_size[0], (image_size[1]+self.pad_w)//patch_size[1]]
        self.num_patches = (self.resolution[0])*(self.resolution[1])

        assert patch_size[0] in [16, 8, 4], f'For convolutional projection, patch size has to be in [8, 16], given patch size {patch_size}'
        if patch_size[0] == 16:
            # self.proj = torch.nn.Sequential(
            #     conv3x3(input_channels, patch_embedding_dim // 8, 2),
            #     nn.ReLU(),
            #     conv3x3(patch_embedding_dim // 8, patch_embedding_dim // 4, 2),
            #     nn.ReLU(),
            #     conv3x3(patch_embedding_dim // 4, patch_embedding_dim // 2, 2),
            #     nn.ReLU(),
            #     conv3x3(patch_embedding_dim // 2, patch_embedding_dim, 2),
            # )
            self.proj = nn.Conv2d(input_channels, patch_embedding_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        elif patch_size[0] == 8:
            self.proj = torch.nn.Sequential(
                conv3x3(input_channels, patch_embedding_dim // 4, 2),
                activation(),
                conv3x3(patch_embedding_dim // 4, patch_embedding_dim // 2, 2),
                activation(),
                conv3x3(patch_embedding_dim // 2, patch_embedding_dim, 2),
            )
        elif patch_size[0] == 4:
            self.proj = torch.nn.Sequential(
                conv3x3(input_channels, patch_embedding_dim // 4, 2),
                activation(),
                conv3x3(patch_embedding_dim // 4, patch_embedding_dim // 1, 2),
            )


    def forward(self, x):
        if self.padding:
            x = F.pad(x, (0, self.pad_w, 0, self.pad_h))
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        if self.flatten: # converts BCHW to B(C*H)W
            x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)

class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(
                backbone, 
                image_size: Tuple[int, int],
                patch_size: Tuple[int, int],
                input_channels: int,
                patch_embedding_dim: int,
                activation: nn.Module = nn.GELU,
                padding: bool = False,
                flatten: bool = True, # converts BCHW to B(C*H)W
                bias: bool = True
            ):
        super().__init__()
        if not isinstance(image_size, tuple):
            image_size = (image_size, image_size)
        if not isinstance(patch_size, tuple):
            patch_size = (patch_size, patch_size)
        self.padding = padding
        self.flatten = flatten
        self.patch_size = patch_size
        self.backbone = backbone
        if input_channels is None:
            with torch.no_grad():
                # NOTE Most reliable way of determining output dims is to run forward pass
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone.forward_features(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[1] == 0
        self.grid_size = (feature_size[0] // patch_size[0], feature_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        B, C, H, W = x.shape
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x , (H//self.patch_size[0], W//self.patch_size[1])

class channel_vit_patch_embedding(nn.Module):
    def __init__(
            self,
            image_size: Tuple[int, int],
            patch_size: Tuple[int, int],
            input_channels: int,
            patch_embedding_dim: int,
            padding: bool = False,
            flatten: bool = True, # converts BCHW to B(C*H)W
            bias: bool = True
        ):
        super().__init__()
        assert isinstance(image_size, tuple),\
            'image_size must be a tuple describing image dimension'
        assert isinstance(patch_size, tuple),\
            'patch_size must be a tuple describing patch dimension'
        self.padding = padding
        self.flatten = flatten
        self.patch_size = patch_size
        if not padding:
            assert image_size[0]%patch_size[0] == 0, \
                f'image height {image_size[0]} is not divisible by patch height {patch_size[0]}'
            assert image_size[1]%patch_size[1] == 0, \
                f'image width {image_size[1]} is not divisible by patch width {patch_size[1]}'
            self.num_patches = (image_size[0]//patch_size[0])*(image_size[1]//patch_size[1])*input_channels
        else:
            self.pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            self.pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            self.num_patches = ((image_size[0]+self.pad_h)//patch_size[0])*((image_size[1]+self.pad_w)//patch_size[1])*input_channels
        self.proj = nn.Conv3d(1, patch_embedding_dim, kernel_size=(1, patch_size[0], patch_size[1]), stride=(1, patch_size[0], patch_size[1]), bias=bias)
        # self.channel_embed = nn.parameter.Parameter(
        #     torch.zeros(1, patch_embedding_dim, input_channels, 1, 1)
        # )
        # nn.init.trunc_normal_(self.proj.bias, std=0.02)
        
    def forward(self, x):
        if self.padding:
            x = F.pad(x, (0, self.pad_w, 0, self.pad_h))
        x = self.proj(x.unsqueeze(1))
        # x += self.channel_embed[:, :, 3, :, :]
        if self.flatten: # converts BCHW to B(C*H)W
            x = x.flatten(2).transpose(1, 2)
        return x
