import numpy as np
import math
import torch
from torch import Tensor

from utils.nn.model.ParticleNet import ParticleNetTagger


def get_model(data_config, **kwargs):
    conv_params = [
        (16, (64, 64, 64)),
        (16, (128, 128, 128)),
        (16, (256, 256, 256)),
        ]
    fc_params = [(256, 0.1)]
    use_fusion = True

    pf_features_dims = len(data_config.input_dicts['pf_features'])
    sv_features_dims = len(data_config.input_dicts['sv_features'])
    num_classes = 1
    model = ParticleNetTagger(pf_features_dims, sv_features_dims, num_classes,
                              conv_params, fc_params,
                              use_fusion=use_fusion,
                              use_fts_bn=kwargs.get('use_fts_bn', False),
                              use_counts=kwargs.get('use_counts', True),
                              pf_input_dropout=kwargs.get('pf_input_dropout', None),
                              sv_input_dropout=kwargs.get('sv_input_dropout', None),
                              for_inference=False,
                              )

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['output'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'output': {0: 'N'}}},
        }

    return model, model_info


class RatioSmoothL1Loss(torch.nn.SmoothL1Loss):
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean', beta: float = 1.0, cutoff: float = 1.0, sine_weight_max: float = None) -> None:
        super(RatioSmoothL1Loss, self).__init__(None, None, reduction)
        self.beta = beta
        self.cutoff = cutoff
        self.sine_weight_max = sine_weight_max

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        label = torch.maximum(target, self.cutoff * torch.ones_like(target))
        if self.sine_weight_max is None:
            return torch.nn.functional.smooth_l1_loss(
                input / label, torch.ones_like(target), reduction=self.reduction, beta=self.beta)
        else:
            wgt = torch.sin(
                np.pi / (2 * self.sine_weight_max) * torch.clip(label, self.cutoff, self.sine_weight_max))
            loss = torch.nn.functional.smooth_l1_loss(
                input / label, torch.ones_like(target), reduction='none', beta=self.beta) * wgt
            # print(list(zip(label, wgt, loss)))
            if self.reduction == 'none':
                return loss
            elif self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()


class SymmetricRatioSmoothL1Loss(torch.nn.SmoothL1Loss):
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean', beta: float = 1.0, cutoff: float = 1.0, sine_weight_max: float = None) -> None:
        super(SymmetricRatioSmoothL1Loss, self).__init__(None, None, reduction)
        self.beta = beta
        self.cutoff = cutoff
        self.sine_weight_max = sine_weight_max

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        label = torch.maximum(target, self.cutoff * torch.ones_like(target))
        pred = torch.maximum(input, self.cutoff * torch.ones_like(input))
        if self.sine_weight_max is None:
            return torch.nn.functional.smooth_l1_loss(
                pred / label, torch.ones_like(target), reduction=self.reduction, beta=self.beta) + torch.nn.functional.smooth_l1_loss(
                label / pred, torch.ones_like(target), reduction=self.reduction, beta=self.beta)
        else:
            wgt = torch.sin(
                np.pi / (2 * self.sine_weight_max) * torch.clip(label, self.cutoff, self.sine_weight_max))
            loss = torch.nn.functional.smooth_l1_loss(
                pred / label, torch.ones_like(target), reduction='none', beta=self.beta) + torch.nn.functional.smooth_l1_loss(
                label / pred, torch.ones_like(target), reduction='none', beta=self.beta)
            loss *= wgt
            # print(list(zip(label, wgt, loss)))
            if self.reduction == 'none':
                return loss
            elif self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()


class LogCoshLoss(torch.nn.L1Loss):
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean') -> None:
        super(LogCoshLoss, self).__init__(None, None, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        x = input - target
        loss = x + torch.nn.functional.softplus(-2. * x) - math.log(2)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()


def get_loss(data_config, **kwargs):
    if kwargs.get('loss_mode', 3) == 1:
        return RatioSmoothL1Loss(beta=kwargs.get('loss_beta', 0.3), cutoff=kwargs.get('loss_cutoff', 1), sine_weight_max=kwargs.get('loss_sine_weight_max', None))
    elif kwargs.get('loss_mode', 3) == 2:
        return SymmetricRatioSmoothL1Loss(beta=kwargs.get('loss_beta', 0.3), cutoff=kwargs.get('loss_cutoff', 1), sine_weight_max=kwargs.get('loss_sine_weight_max', None))
    elif kwargs.get('loss_mode', 3) == 3:
        return LogCoshLoss()
    else:
        return torch.nn.SmoothL1Loss(beta=kwargs.get('loss_beta', 10))
