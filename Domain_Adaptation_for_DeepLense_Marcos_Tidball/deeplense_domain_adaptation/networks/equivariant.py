from torch import nn
from e2cnn import gspaces
import e2cnn.nn as e2nn

class Encoder(nn.Module):
    def __init__(self, features_size=256, sym_group='Dihedral', N=2):
        """
        Equivariant neural network receives that images and encodes them into an array of size `features_size`.
        NEED TO DOWNGRADE VERSIONS: !pip install torch==1.4 torchvision==0.5

        Arguments:
        ----------
        features_size: int
            Size of encoded features array.

        sym_group: str ['Dihedral', 'Circular']
            Kind of symmetry group to consider.

        N: int
            Number of discrete rotations (integer greater than 1) or -1 for continuous rotations.
        """

        super(Encoder, self).__init__()

        if sym_group == 'Dihedral':
            self.r2_act = gspaces.FlipRot2dOnR2(N=N)

        elif sym_group == 'Circular':
            self.r2_act = gspaces.Rot2dOnR2(N=N)

        in_type = e2nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.input_type = in_type

        out_type = e2nn.FieldType(self.r2_act, 24*[self.r2_act.regular_repr])
        self.block1 = e2nn.SequentialModule(
            e2nn.MaskModule(in_type, 150, margin=1),
            e2nn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True)
        )

        in_type = self.block1.out_type
        out_type = e2nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.block2 = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True)
        )
        self.pool1 = e2nn.SequentialModule(
            e2nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        in_type = self.block2.out_type
        out_type = e2nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.block3 = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True)
        )

        in_type = self.block3.out_type
        out_type = e2nn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.block4 = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True)
        )
        self.pool2 = e2nn.SequentialModule(
            e2nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        in_type = self.block4.out_type
        out_type = e2nn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.block5 = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True)
        )

        in_type = self.block5.out_type
        out_type = e2nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])
        self.block6 = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True)
        )
        self.pool3 = e2nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)

        self.gpool = e2nn.GroupPooling(out_type)
        c = self.gpool.out_type.size

        self.fully_net = nn.Sequential(
            nn.Linear(61504, features_size),
            nn.BatchNorm1d(features_size),
            nn.ELU(inplace=True),
            nn.Linear(features_size, features_size),
        )

    def forward(self, x):
        x = e2nn.GeometricTensor(x, self.input_type)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.pool3(x)
        x = self.gpool(x)
        x = x.tensor
        x = self.fully_net(x.reshape(x.shape[0], -1))
        return x

class Classifier(nn.Module):
    def __init__(self, features_size=256, num_classes=3):        
        """
        ResNet based neural network that receives an array of size `features_size` and classifies it into `num_classes` classes.

        Arguments:
        ----------
        features_size: int
            Size of encoded features array.

        num_classes: int
            Number of classes to classify the encoded array into.
        """

        super(Classifier, self).__init__()
        self.fc = nn.Linear(features_size, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x