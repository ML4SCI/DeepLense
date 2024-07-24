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

def channel_vit_tiny(
        image_size: Union[int, Tuple[int, int]],
        input_channels: int,
        patch_size: Union[int, Tuple[int, int]] = 16,
        **kwargs):
    return VisionTransformer(
                image_size = image_size,
                input_channels = input_channels,
                patch_size=patch_size, 
                patch_embedding_type="channel_vit",
                embed_dim=192, 
                depth=12, 
                num_heads=3, 
                mlp_ratio=4,
                qkv_bias=True, 
                norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                use_dense_prediction = use_dense_prediction,
                return_all_tokens = False,
                masked_im_modeling = False,
                **kwargs
            )

def channel_vit_small(
        image_size: Union[int, Tuple[int, int]],
        input_channels: int,
        patch_size: Union[int, Tuple[int, int]] = 16,
        use_dense_prediction: bool = False,
        **kwargs):
    return VisionTransformer(
                image_size = image_size,
                input_channels = input_channels,
                patch_size=patch_size, 
                patch_embedding_type="channel_vit", 
                embed_dim=384, 
                depth=12, 
                num_heads=8, 
                mlp_ratio=4,
                qkv_bias=True, 
                qk_norm=False,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                attn_drop = 0.,
                # attn_drop = 0.,
                drop_path_rate = 0.,
                pos_drop_rate = 0.,
                proj_drop = 0.,
                use_dense_prediction = use_dense_prediction,
                return_all_tokens = False,
                masked_im_modeling = False,
                **kwargs
            )

def channel_vit_base(
        image_size: Union[int, Tuple[int, int]],
        input_channels: int,
        patch_size: Union[int, Tuple[int, int]] = 16,
        **kwargs):
    return VisionTransformer(
                image_size = image_size,
                input_channels = input_channels,
                patch_size=patch_size, 
                patch_embedding_type="channel_vit", 
                embed_dim=768, 
                depth=12, 
                num_heads=12, 
                mlp_ratio=4,
                qkv_bias=True, 
                norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                **kwargs
            )

    
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
    if arch.lower() == "resnet50":
        net = resnet50(in_chans=input_channels, drop_rate=0., drop_path_rate=.2, drop_block_rate=0.)
        net.fc = nn.Identity()
        net.embed_dim = net.num_features
        net.return_all_tokens = return_all_tokens
        return net
    if arch.lower() == "resnet18":
        net = resnet18(in_chans=input_channels, drop_rate=0., drop_path_rate=.2, drop_block_rate=0.)
        net.fc = nn.Identity()
        net.embed_dim = net.num_features
        net.return_all_tokens = return_all_tokens
        return net
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
    if arch.lower() == "channel_vit_tiny":
        return channel_vit_tiny(image_size, input_channels, patch_size)
    elif arch.lower() == "channel_vit_small":
        return channel_vit_small(image_size, input_channels, patch_size, use_dense_prediction)
    elif arch.lower() == "channel_vit_base":
        return channel_vit_base(image_size, input_channels, patch_size)
    else:
        print(f"Backbone architecture specified as {arch} is not implemented. Exiting.")
        sys.exit(1)
