# adapted from
#    https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
import torch
import torch.nn as nn
from typing import Union, Tuple, List, Optional

class MLPHead(nn.Module):
    def __init__(
            self, 
            input_dim: int, 
            output_dim: int, 
            hidden_dim: int = 2048,
            nlayers: int = 3,
            bottleneck_dim: Optional[int] = None, #256 in dino
            use_bn: bool = False, 
            norm_last_layer: bool = False,   
            normalize_outputs: bool = True,
            init: str = "trunc_normal",
            norm_layer: nn.Module = nn.BatchNorm1d,
            activation: nn.Module = nn.GELU,
        ):
        super().__init__()
        if init == "kaiming_normal":
            activation = nn.ReLU
        self.init = init
        nlayers = max(nlayers, 1)
        layer_dim = []
        layer_dim.append(input_dim)
        for i in range(nlayers-1):
            layer_dim.append(hidden_dim)
        bottleneck = bottleneck_dim is not None
        bottleneck_dim = bottleneck_dim if bottleneck_dim is not None else output_dim
        layer_dim.append(bottleneck_dim)

        mlp = []
        for i in range(len(layer_dim)-1):
            mlp.append(nn.Linear(layer_dim[i], layer_dim[i+1]))
            if i < len(layer_dim) - 2:
                if use_bn:
                    mlp.append(norm_layer(layer_dim[i+1]))
                mlp.append(activation())
        if (not bottleneck) and norm_last_layer:
            mlp.append(norm_layer(layer_dim[-1]))
        self.mlp = nn.Sequential(*mlp)
            
        self.apply(self._init_weights)
        self.last_layer = None
        if bottleneck:
            self.last_layer = nn.utils.weight_norm(
                nn.Linear(bottleneck_dim, output_dim, bias=False)
            )
            self.last_layer.weight_g.data.fill_(1)
            if norm_last_layer:
                self.last_layer.weight_g.requires_grad = False

        self.normalize_outputs = normalize_outputs
        self.output_dim = output_dim
        self.init = init

    def _init_weights(self, m):
        if isinstance(m, nn.BatchNorm1d):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            if self.init == "trunc_normal":  
                if m.weight is not None:
                    nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif self.init == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        

    def forward(self, x):
        x = self.mlp(x)
        if self.normalize_outputs:
            x = nn.functional.normalize(x, dim=-1, p=2)
        if self.last_layer is not None:
            x = self.last_layer(x)
        return x

class PrototypesWrapper(nn.Module):
    def __init__(
            self, 
            output_dim: int, 
            num_prototypes: Union[int, List],
        ):
        super().__init__()
        if not isinstance(num_prototypes, List):
            num_prototypes = [num_prototypes]
        self.num_heads = len(num_prototypes)
        for i, k in enumerate(num_prototypes):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.num_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out
        
class MultiCropWrapper(nn.Module):
    def __init__(
            self, 
            backbone: nn.Module, 
            head: nn.Module,
            num_prototypes: Optional[Union[int, List]] = None,
            head_dense: Optional[nn.Module] = None,
            use_dense_prediction: bool = False,
        ):
        super().__init__()
        # backbone.head = nn.Identity()  # deactivate original head
        self.backbone = backbone
        self.head = head
        self.head_dense = head_dense
        self.use_dense_prediction = use_dense_prediction
        self.num_prototypes = num_prototypes
        self.prototypes = None
        if num_prototypes is not None:
            self.prototypes = nn.Linear(head.output_dim, num_prototypes, bias=False)

    def forward(self, x, mask=None, return_backbone_feat=False, ):
        # convert to list
        if not isinstance(x, list):
            x = [x]
            mask = [mask] if mask is not None else None
        
        crop_dims = torch.tensor([inp.shape[-1] for inp in x])
        crop_dim_counts = torch.unique_consecutive(crop_dims, return_counts=True)[1]
        crops_idx = torch.cumsum(crop_dim_counts, 0)

        start_idx = 0
        output, output_fea, npatch = None, None, None

        if self.use_dense_prediction:
            for end_idx in crops_idx:
                x_cat = torch.cat(x[start_idx: end_idx])
                # print(self.backbone.use_dense_prediction)
                _out, _out_fea  = self.backbone(x_cat)
                B, N, C = _out_fea.shape

                if start_idx == 0:
                    output = _out
                    output_fea = _out_fea.reshape(B * N, C)
                    npatch = [N]
                else:
                    output = torch.cat((output, _out))
                    output_fea = torch.cat((output_fea, _out_fea.reshape(B * N, C)))
                    npatch.append(N)
                start_idx = end_idx

            return self.head(output), self.head_dense(output_fea), output_fea, npatch 
        else:
            for end_idx in crops_idx:
                # concatenate similar shaped crops along the batch dim
                x_cat = torch.cat(x[start_idx: end_idx])
                inp_m = None
                if mask is not None:
                    inp_m = torch.cat(mask[start_idx: end_idx])
                _x = (x_cat, inp_m) if mask is not None else x_cat
                _out = self.backbone(x=_x)

                if start_idx == 0:
                    output = _out
                else:
                    output = torch.cat((output, _out))
                start_idx = end_idx
            output_ = self.head(output)
            if self.prototypes is not None:
                output_ = (output_, self.prototypes(output_))
            if return_backbone_feat:
                return output, output_
            return output_