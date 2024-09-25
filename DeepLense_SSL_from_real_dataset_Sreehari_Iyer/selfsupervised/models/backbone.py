# adapted from
#     https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
#     https://github.com/facebookresearch/dino/blob/main/vision_transformer.py

import torch
import torch.nn as nn
from .vit import VisionTransformer
from timm.models.resnet import resnet50, resnet18
from functools import partial
from typing import Union, Tuple, List, Optional

def vit_tiny(
        image_size: Union[int, Tuple[int, int]],
        input_channels: int,
        patch_size: Union[int, Tuple[int, int]] = 16,
        **kwargs):
    return VisionTransformer(
                image_size = image_size,
                input_channels = input_channels,
                patch_size=patch_size, 
                embed_dim=192, 
                depth=12, 
                num_heads=3, 
                mlp_ratio=4,
                qkv_bias=True, 
                norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                **kwargs
            )

def vit_small(
        image_size: Union[int, Tuple[int, int]],
        input_channels: int,
        patch_size: Union[int, Tuple[int, int]] = 16,
        use_dense_prediction: bool = False,
        return_all_tokens: bool = True,
        masked_im_modeling: bool = True,
        **kwargs):
    return VisionTransformer(
                image_size = image_size,
                input_channels = input_channels,
                patch_size=patch_size, 
                embed_dim=384, 
                depth=12, 
                num_heads=8, 
                mlp_ratio=4,
                qkv_bias=True, 
                qk_norm=False,
                norm_layer=partial(nn.LayerNorm, eps=1e-12), 
                attn_drop = 0.2,
                # attn_drop = 0.,
                drop_path_rate = 0.1,
                pos_drop_rate = 0.,
                proj_drop = 0.,
                use_dense_prediction = use_dense_prediction,
                return_all_tokens = return_all_tokens,
                masked_im_modeling = masked_im_modeling,
                **kwargs
            )

def vit_mlp_small(
        image_size: Union[int, Tuple[int, int]],
        input_channels: int,
        patch_size: Union[int, Tuple[int, int]] = 16,
        **kwargs):
    return VisionTransformer(
                image_size = image_size,
                input_channels = input_channels,
                patch_size=patch_size, 
                embed_dim=384, 
                depth=12, 
                num_heads=8, 
                mlp_ratio=4,
                qkv_bias=True, 
                qk_norm=False,
                norm_layer=partial(nn.LayerNorm, eps=1e-12), 
                attn_drop = 0.2,
                drop_path_rate = 0.1,
                pos_drop_rate = 0.,
                proj_drop = 0.,
                use_fc_norm = True,
                head_drop_rate = 0.2,
                head = nn.Sequential(nn.LayerNorm(384), nn.Dropout(p=0.2), nn.ReLU(),\
                                     nn.Linear(384, 8*384), nn.LayerNorm(8*384), nn.Dropout(p=0.2),\
                                     nn.ReLU(),\
                                     # nn.Linear(8*384, 8*384), nn.LayerNorm(8*384), nn.Dropout(p=0.2),\
                                     # nn.ReLU(),\
                                     nn.Linear(8*384, 384)),
                **kwargs
            )
    
def vit_base(
        image_size: Union[int, Tuple[int, int]],
        input_channels: int,
        patch_size: Union[int, Tuple[int, int]] = 16,
        use_dense_prediction: bool = False,
        return_all_tokens: bool = True,
        masked_im_modeling: bool = True,
        **kwargs):
    return VisionTransformer(
                image_size = image_size,
                input_channels = input_channels,
                patch_size=patch_size, 
                embed_dim=768, 
                depth=12, 
                num_heads=12, 
                mlp_ratio=4,
                qkv_bias=True, 
                qk_norm=False,
                norm_layer=partial(nn.LayerNorm, eps=1e-12), 
                attn_drop = 0.2,
                # attn_drop = 0.,
                drop_path_rate = 0.1,
                pos_drop_rate = 0.,
                proj_drop = 0.,
                use_dense_prediction = use_dense_prediction,
                return_all_tokens = return_all_tokens,
                masked_im_modeling = masked_im_modeling,
                **kwargs
            )

def vit_mlp_base(
        image_size: Union[int, Tuple[int, int]],
        input_channels: int,
        patch_size: Union[int, Tuple[int, int]] = 16,
        **kwargs):
    return VisionTransformer(
                image_size = image_size,
                input_channels = input_channels,
                patch_size=patch_size, 
                embed_dim=768, 
                depth=12, 
                num_heads=12, 
                mlp_ratio=4,
                qkv_bias=True, 
                norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                head = nn.Sequential(nn.Linear(768, 192), nn.GELU(),\
                                     nn.Linear(192, 768)),
                use_fc_norm = True,
                head_drop_rate = 0.,
                **kwargs
            )

    
class _resnet18(nn.Module):
    def __init__(self, input_channels, drop_rate=0., drop_path_rate=.2, drop_block_rate=0.):
        super().__init__()
        self.resnet18 = resnet18(in_chans=input_channels, drop_rate=0., drop_path_rate=.2, drop_block_rate=0.)
        self.resnet18.fc = nn.Identity()
        self.embed_dim = self.resnet18.num_features
        self.return_all_tokens = False
        self.masked_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.masked_embed, std=0.02)
    def mask_model(self, x, mask):
        B, L, C = x.shape
        mask_token = self.masked_embed.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w
        return x
    def forward(self, x: torch.Tensor):
        mask = None
        if isinstance(x, tuple):
            x, mask = x
        B, C, W, H = x.shape
        if mask is not None:
            x = self.mask_model(x, mask)
        return self.resnet18(x)

def Backbone(
        arch: str,
        image_size: Union[int, Tuple[int, int]],
        input_channels: int,
        patch_size: Union[int, Tuple[int, int]] = 16,
        use_dense_prediction: Optional[bool] = False,
        return_all_tokens: bool = False,
        masked_im_modeling: bool = False,
        window_size: Optional[int] = None,
        ape: Optional[bool] = False,
    ):
    if arch.lower() == "resnet18":
        return _resnet18(input_channels, drop_rate=0., drop_path_rate=.2, drop_block_rate=0.)
    if arch.lower() == "vit_tiny":
        return vit_tiny(image_size, input_channels, patch_size)
    elif arch.lower() == "vit_small":
        return vit_small(image_size, input_channels, patch_size, use_dense_prediction, return_all_tokens, masked_im_modeling)
    elif arch.lower() == "vit_mlp_small":
        return vit_mlp_small(image_size, input_channels, patch_size)
    elif arch.lower() == "vit_base":
        return vit_base(image_size, input_channels, patch_size, use_dense_prediction, return_all_tokens, masked_im_modeling)
    elif arch.lower() == "vit_mlp_base":
        return vit_mlp_base(image_size, input_channels, patch_size)
    else:
        print(f"Backbone architecture specified as {arch} is not implemented. Exiting.")
        sys.exit(1)
