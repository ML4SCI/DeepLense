"""
Code referenced from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
"""

import torch
from torch import nn
from torch.nn import Conv2d

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchvision import models


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        """Feed forward neural network. MLP architecture.

        Args:
            dim: input dim of this module
            hidden_dim: hidden dim of this module
            dropout: dropout rate
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        """Attention module

        Args:
            dim: input dimension of this module
            heads: head number
            dim_head: dimension of head
            dropout: dropout rate
        """
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # can tune
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            Rearrange('b c h w -> b (h w) c')
            # Rearrange('b c h w -> b c (h w)')
        )

    def forward(self, x):
        return self.cnn(x)


# class SimCNN(nn.Module):
#     def __init__(self):
#         super(SimCNN, self).__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=2),
#             nn.BatchNorm2d(8),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=2),
#             nn.BatchNorm2d(8),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=2),
#             nn.BatchNorm2d(8),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(8*8*8, 512),
#             nn.ReLU(),
#             nn.Linear(512, 10)
#         )
#
#     def forward(self, x):
#         return self.cnn(x)
#
#
# class ResNet(nn.Module):
#
#     def __init__(self, pre_trained=False):
#         """
#         Args:
#             pre_trained: True if want to use pretrained weight else false
#         """
#         super(ResNet, self).__init__()
#         self.backbone = models.resnet34(pretrained=pre_trained)
#         self.reg = nn.Sequential(
#             # nn.Linear(2048, 1)
#             nn.Linear(512, 10)
#         )
#         self.backbone.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         self.backbone.fc = self.reg
#
#     def forward(self, x):
#         return self.backbone(x)


class CNNT(nn.Module):
    def __init__(self, num_classes, depth, heads, mlp_dim, pool='mean', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.to_patch_embedding = CNN()

        # hyper-params
        num_patches = 16 * 16
        dim = 32

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)

        # b: batch size  n: patch number  _: dim of patch
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
