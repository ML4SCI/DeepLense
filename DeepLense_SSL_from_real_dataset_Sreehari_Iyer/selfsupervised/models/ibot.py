# code from https://github.com/bytedance/ibot/blob/main/models/head.py

import torch
import torch.nn as nn

class CSyncBatchNorm(nn.SyncBatchNorm):
    def __init__(self,
                 *args,
                 with_var=False,
                 **kwargs):
        super(CSyncBatchNorm, self).__init__(*args, **kwargs)
        self.with_var = with_var

    def forward(self, x):
        # center norm
        self.training = False
        if not self.with_var:
            self.running_var = torch.ones_like(self.running_var)
        normed_x = super(CSyncBatchNorm, self).forward(x)
        # udpate center
        self.training = True
        _ = super(CSyncBatchNorm, self).forward(x)
        return normed_x

class CustomSequential(nn.Sequential):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)

    def forward(self, input):
        for module in self:
            dim = len(input.shape)
            if isinstance(module, self.bn_types) and dim > 2:
                perm = list(range(dim - 1)); perm.insert(1, dim - 1)
                inv_perm = list(range(dim)) + [1]; inv_perm.pop(1)
                input = module(input.permute(*perm)).permute(*inv_perm)
            else:
                input = module(input)
        return input

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, norm=None, act='gelu', last_norm=None, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256, norm_last_layer=True, **kwargs):
        super().__init__()
        norm = self._build_norm(norm, hidden_dim)
        last_norm = self._build_norm(last_norm, out_dim, affine=False, **kwargs)
        act = self._build_act(act)

        nlayers = max(nlayers, 1)
        if nlayers == 1:
            if bottleneck_dim > 0:
                self.mlp = nn.Linear(in_dim, bottleneck_dim)
            else:
                self.mlp = nn.Linear(in_dim, out_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if norm is not None:
                layers.append(norm)
            layers.append(act)
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if norm is not None:
                    layers.append(norm)
                layers.append(act)
            if bottleneck_dim > 0:
                layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            else:
                layers.append(nn.Linear(hidden_dim, out_dim))
            self.mlp = CustomSequential(*layers)
        self.apply(self._init_weights)
        
        if bottleneck_dim > 0:
            self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
            self.last_layer.weight_g.data.fill_(1)
            if norm_last_layer:
                self.last_layer.weight_g.requires_grad = False
        else:
            self.last_layer = None

        self.last_norm = last_norm

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        if self.last_layer is not None:
            x = nn.functional.normalize(x, dim=-1, p=2)
            x = self.last_layer(x)
        if self.last_norm is not None:
            x = self.last_norm(x)
        return x

    def _build_norm(self, norm, hidden_dim, **kwargs):
        if norm == 'bn':
            norm = nn.BatchNorm1d(hidden_dim, **kwargs)
        elif norm == 'syncbn':
            norm = nn.SyncBatchNorm(hidden_dim, **kwargs)
        elif norm == 'csyncbn':
            norm = CSyncBatchNorm(hidden_dim, **kwargs)
        elif norm == 'ln':
            norm = nn.LayerNorm(hidden_dim, **kwargs)
        else:
            assert norm is None, "unknown norm type {}".format(norm)
        return norm

    def _build_act(self, act):
        if act == 'relu':
            act = nn.ReLU()
        elif act == 'gelu':
            act = nn.GELU()
        else:
            assert False, "unknown act type {}".format(act)
        return act

class iBOTHead(DINOHead):

    def __init__(self, *args, out_dim=8192, norm=None, act='gelu', last_norm=None, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256, norm_last_layer=True, 
                 shared_head=False, use_bn=True, **kwargs):

        super(iBOTHead, self).__init__(*args,
                                        norm=None if not use_bn else "bn",
                                        act=act,
                                        last_norm=last_norm,
                                        nlayers=nlayers,
                                        out_dim=out_dim,
                                        hidden_dim=hidden_dim,
                                        bottleneck_dim=bottleneck_dim,
                                        norm_last_layer=norm_last_layer, 
                                        **kwargs)

        if not shared_head:
            if bottleneck_dim > 0:
                self.last_layer2 = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
                self.last_layer2.weight_g.data.fill_(1)
                if norm_last_layer:
                    self.last_layer2.weight_g.requires_grad = False
            else:
                self.mlp2 = nn.Linear(hidden_dim, out_dim)
                self.last_layer2 = None

            self.last_norm2 = self._build_norm(last_norm, out_dim, affine=False, **kwargs)
        else:
            if bottleneck_dim > 0:
                self.last_layer2 = self.last_layer
            else:
                self.mlp2 = self.mlp[-1]
                self.last_layer2 = None

            self.last_norm2 = self.last_norm

    def forward(self, x):
        if len(x.shape) == 2:
            return super(iBOTHead, self).forward(x)

        if self.last_layer is not None:
            x = self.mlp(x)
            x = nn.functional.normalize(x, dim=-1, p=2)
            x1 = self.last_layer(x[:, 0])
            x2 = self.last_layer2(x[:, 1:])
        else:
            x = self.mlp[:-1](x)
            x1 = self.mlp[-1](x[:, 0])
            x2 = self.mlp2(x[:, 1:])
        
        if self.last_norm is not None:
            x1 = self.last_norm(x1)
            x2 = self.last_norm2(x2)
        
        return x1, x2
