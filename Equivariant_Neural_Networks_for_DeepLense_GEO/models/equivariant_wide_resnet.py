from typing import Tuple

import torch
import torch.nn.functional as F

import math

import e2cnn.nn as enn
from e2cnn.nn import init
from e2cnn import gspaces


__all__ = [
    "wrn16_8_stl_d8d4d1",
    "wrn16_8_stl_d8d4d4",
    "wrn16_8_stl_d1d1d1",
    "wrn28_10_d8d4d1",
    "wrn28_7_d8d4d1",
    "wrn28_10_c8c4c1",
    "wrn28_10_d1d1d1",
]


def conv7x7(in_type: enn.FieldType, out_type: enn.FieldType, stride=1, padding=3,
            dilation=1, bias=False):
    """7x7 convolution with padding"""
    return enn.R2Conv(in_type, out_type, 7,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3*r,
                      )


def conv5x5(in_type: enn.FieldType, out_type: enn.FieldType, stride=1, padding=2,
            dilation=1, bias=False):
    """5x5 convolution with padding"""
    return enn.R2Conv(in_type, out_type, 5,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3*r,
                      )


def conv3x3(in_type: enn.FieldType, out_type: enn.FieldType, stride=1, padding=1,
            dilation=1, bias=False):
    """3x3 convolution with padding"""
    return enn.R2Conv(in_type, out_type, 3,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3*r,
                      )


def conv1x1(in_type: enn.FieldType, out_type: enn.FieldType, stride=1, padding=0,
            dilation=1, bias=False):
    """1x1 convolution with padding"""
    return enn.R2Conv(in_type, out_type, 1,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3*r,
                      )


def regular_feature_type(gspace: gspaces.GSpace, planes: int, fixparams: bool = True):
    """ build a regular feature map with the specified number of channels"""
    assert gspace.fibergroup.order() > 0

    N = gspace.fibergroup.order()

    if fixparams:
        planes *= math.sqrt(N)

    planes = planes / N
    planes = int(planes)

    return enn.FieldType(gspace, [gspace.regular_repr] * planes)


def trivial_feature_type(gspace: gspaces.GSpace, planes: int, fixparams: bool = True):
    """ build a trivial feature map with the specified number of channels"""

    if fixparams:
        planes *= math.sqrt(gspace.fibergroup.order())

    planes = int(planes)
    return enn.FieldType(gspace, [gspace.trivial_repr] * planes)


FIELD_TYPE = {
    "trivial": trivial_feature_type,
    "regular": regular_feature_type,
}


class WideBasic(enn.EquivariantModule):

    def __init__(self,
                 in_type: enn.FieldType,
                 inner_type: enn.FieldType,
                 dropout_rate: float,
                 stride: int = 1,
                 out_type: enn.FieldType = None,
                 ):
        super(WideBasic, self).__init__()

        if out_type is None:
            out_type = in_type

        self.in_type = in_type
        inner_type = inner_type
        self.out_type = out_type

        if isinstance(in_type.gspace, gspaces.FlipRot2dOnR2):
            rotations = in_type.gspace.fibergroup.rotation_order
        elif isinstance(in_type.gspace, gspaces.Rot2dOnR2):
            rotations = in_type.gspace.fibergroup.order()
        else:
            rotations = 0

        if rotations in [0, 2, 4]:
            conv = conv3x3
        else:
            conv = conv5x5

        self.bn1 = enn.InnerBatchNorm(self.in_type)
        self.relu1 = enn.ReLU(self.in_type, inplace=True)
        self.conv1 = conv(self.in_type, inner_type)

        self.bn2 = enn.InnerBatchNorm(inner_type)
        self.relu2 = enn.ReLU(inner_type, inplace=True)

        self.dropout = enn.PointwiseDropout(inner_type, p=dropout_rate)

        self.conv2 = conv(inner_type, self.out_type, stride=stride)

        self.shortcut = None
        if stride != 1 or self.in_type != self.out_type:
            self.shortcut = conv1x1(
                self.in_type, self.out_type, stride=stride, bias=False)

    def forward(self, x):
        x_n = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(x_n)))
        out = self.dropout(out)
        out = self.conv2(out)

        if self.shortcut is not None:
            out += self.shortcut(x_n)
        else:
            out += x

        return out

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        if self.shortcut is not None:
            return self.shortcut.evaluate_output_shape(input_shape)
        else:
            return input_shape


class Wide_ResNet(torch.nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes=100,
                 N: int = 8,
                 r: int = 1,
                 f: bool = True,
                 deltaorth: bool = False,
                 fixparams: bool = True,
                 initial_stride: int = 1,
                 ):

        super(Wide_ResNet, self).__init__()

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = int((depth - 4) / 6)
        k = widen_factor

        print(f'| Wide-Resnet {depth}x{k}')

        nStages = [16, 16 * k, 32 * k, 64 * k]

        self._fixparams = fixparams

        self._layer = 0

        # number of discrete rotations to be equivariant to
        self._N = N

        # if the model is [F]lip equivariant
        self._f = f
        if self._f:
            if N != 1:
                self.gspace = gspaces.FlipRot2dOnR2(N)
            else:
                self.gspace = gspaces.Flip2dOnR2()
        else:
            if N != 1:
                self.gspace = gspaces.Rot2dOnR2(N)
            else:
                self.gspace = gspaces.TrivialOnR2()

        assert r in [0, 1, 2, 3]
        self._r = r

        # the input has 3 color channels (RGB).
        # Color channels are trivial fields and don't transform when the input is rotated or flipped
        r1 = enn.FieldType(self.gspace, [self.gspace.trivial_repr])

        # input field type of the model
        self.in_type = r1

        # in the first layer we always scale up the output channels to allow for enough independent filters
        r2 = FIELD_TYPE["regular"](self.gspace, nStages[0], fixparams=True)

        # dummy attribute keeping track of the output field type of the last submodule built, i.e. the input field type of
        # the next submodule to build
        self._in_type = r2

        self.conv1 = conv5x5(r1, r2)
        self.layer1 = self._wide_layer(
            WideBasic, nStages[1], n, dropout_rate, stride=initial_stride)
        if self._r >= 2:
            N_new = N//2
            id = (0, N_new) if self._f else N_new
            self.restrict1 = self._restrict_layer(id)
        else:
            self.restrict1 = lambda x: x

        self.layer2 = self._wide_layer(
            WideBasic, nStages[2], n, dropout_rate, stride=2)
        if self._r == 3:
            id = (0, 1) if self._f else 1
            self.restrict2 = self._restrict_layer(id)
        elif self._r == 1:
            N_new = N // 2
            id = (0, N_new) if self._f else N_new
            self.restrict2 = self._restrict_layer(id)
        else:
            self.restrict2 = lambda x: x

        # last layer maps to a trivial (invariant) feature map
        self.layer3 = self._wide_layer(
            WideBasic, nStages[3], n, dropout_rate, stride=2, totrivial=True)

        self.bn = enn.InnerBatchNorm(self.layer3.out_type, momentum=0.9)
        self.relu = enn.ReLU(self.bn.out_type, inplace=True)
        self.linear = torch.nn.Linear(self.bn.out_type.size, num_classes)

        for name, module in self.named_modules():
            if isinstance(module, enn.R2Conv):
                if deltaorth:
                    init.deltaorthonormal_init(
                        module.weights, module.basisexpansion)
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, torch.nn.Linear):
                module.bias.data.zero_()

        print("MODEL TOPOLOGY:")
        for i, (name, mod) in enumerate(self.named_modules()):
            print(f"\t{i} - {name}")

    def _restrict_layer(self, subgroup_id) -> enn.SequentialModule:
        layers = list()
        layers.append(enn.RestrictionModule(self._in_type, subgroup_id))
        layers.append(enn.DisentangleModule(layers[-1].out_type))
        self._in_type = layers[-1].out_type
        self.gspace = self._in_type.gspace

        restrict_layer = enn.SequentialModule(*layers)
        return restrict_layer

    def _wide_layer(self, block, planes: int, num_blocks: int, dropout_rate: float, stride: int,
                    totrivial: bool = False
                    ) -> enn.SequentialModule:

        self._layer += 1
        print("start building", self._layer)
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        main_type = FIELD_TYPE["regular"](
            self.gspace, planes, fixparams=self._fixparams)
        inner_type = FIELD_TYPE["regular"](
            self.gspace, planes, fixparams=self._fixparams)

        if totrivial:
            out_type = FIELD_TYPE["trivial"](
                self.gspace, planes, fixparams=self._fixparams)
        else:
            out_type = FIELD_TYPE["regular"](
                self.gspace, planes, fixparams=self._fixparams)

        for b, stride in enumerate(strides):
            if b == num_blocks - 1:
                out_f = out_type
            else:
                out_f = main_type
            layers.append(block(self._in_type, inner_type,
                          dropout_rate, stride, out_type=out_f))
            self._in_type = out_f

        print("layer", self._layer, "built")
        return enn.SequentialModule(*layers)

    def features(self, x):

        x = enn.GeometricTensor(x, self.in_type)

        out = self.conv1(x)

        x1 = self.layer1(out)

        x2 = self.layer2(self.restrict1(x1))

        x3 = self.layer3(self.restrict2(x2))

        return x1, x2, x3

    def forward(self, x):

        # wrap the input tensor in a GeometricTensor
        x = enn.GeometricTensor(x, self.in_type)

        out = self.conv1(x)
        out = self.layer1(out)

        out = self.layer2(self.restrict1(out))

        out = self.layer3(self.restrict2(out))

        out = self.bn(out)
        out = self.relu(out)

        # extract the tensor from the GeometricTensor to use the common Pytorch operations
        out = out.tensor

        b, c, w, h = out.shape
        out = F.avg_pool2d(out, (w, h))

        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out
