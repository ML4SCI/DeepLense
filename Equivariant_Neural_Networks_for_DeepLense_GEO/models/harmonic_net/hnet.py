from .hnet_modules import Conv2d, HNonlin, BatchNorm, mean_pool, sum_magnitudes
import torch.nn.functional as F
from torch import nn
import torch

# import sys
# [TODO] sys.path.append('../')


class HNet(nn.Module):
    """Pytorch model class for Rot-Equivariant CNN"""

    def __init__(self, args):
        super().__init__()
        """
        Initialization function
        Args:
            args: argument variable to be used to initialize values for all the hyperparameters
        """

        self.args = args
        self.order = 1
        # NUmber of filters
        self.nf = self.args.n_filters
        self.nf2 = int(self.nf * self.args.filter_gain)
        self.nf3 = int(self.nf * (self.args.filter_gain**2.0))
        self.bs = self.args.batch_size
        self.fs = self.args.filter_size
        self.ncl = self.args.n_classes
        self.sm = self.args.std_mult
        self.nr = self.args.n_rings

        self.bias = torch.ones(self.ncl) * 1e-2
        self.bias = torch.nn.Parameter(
            self.bias.type(
                torch.cuda.FloatTensor
                if torch.cuda.is_available()
                else torch.FloatTensor
            )
        )

        # defining convolutional layer objects
        self.conv2d_1_nf = Conv2d(
            1, self.nf, self.fs, padding=(self.fs - 1) // 2, n_rings=self.nr
        )
        self.conv2d_nf_nf = Conv2d(
            self.nf, self.nf, self.fs, padding=(self.fs - 1) // 2, n_rings=self.nr
        )
        self.conv2d_nf_nf2 = Conv2d(
            self.nf, self.nf2, self.fs, padding=(self.fs - 1) // 2, n_rings=self.nr
        )
        self.conv2d_nf2_nf2 = Conv2d(
            self.nf2, self.nf2, self.fs, padding=(self.fs - 1) // 2, n_rings=self.nr
        )
        self.conv2d_nf2_nf3 = Conv2d(
            self.nf2, self.nf3, self.fs, padding=(self.fs - 1) // 2, n_rings=self.nr
        )
        self.conv2d_nf3_nf3 = Conv2d(
            self.nf3, self.nf3, self.fs, padding=(self.fs - 1) // 2, n_rings=self.nr
        )
        self.conv2d_nf3_ncl = Conv2d(
            self.nf3,
            self.ncl,
            self.fs,
            padding=(self.fs - 1) // 2,
            n_rings=self.nr,
            phase=False,
        )

        # defining the nonliearity objects
        self.nonlin1 = HNonlin(F.relu, self.order, self.nf, eps=1e-12)
        self.nonlin3 = HNonlin(F.relu, self.order, self.nf2, eps=1e-12)
        self.nonlin5 = HNonlin(F.relu, self.order, self.nf3, eps=1e-12)

        # defining the batchnorm objects
        self.bn2 = BatchNorm(2, 2, 8)
        self.bn4 = BatchNorm(2, 2, 16)
        self.bn6 = BatchNorm(2, 2, 32)

    def forward(self, x: torch.FloatTensor):
        """
        Defines the forward propagation for the model

        Args:
            x (torch FloatTensor): contains of images for the current
            training step.

        Returns output activations of the model
        """
        x = x.view(x.shape[0], x.shape[2], x.shape[3], 1, 1, 1)
        # defining block 1
        cv1 = self.conv2d_1_nf(x)
        cv1_bcc = cv1.detach().clone()
        cv1 = self.nonlin1(cv1)

        cv2 = self.conv2d_nf_nf(cv1)
        # cv2_bcc = cv2.detach().clone()
        cv2 = self.bn2(cv2)

        # defining block 2
        cv2 = mean_pool(cv2, kernel_size=(2, 2), strides=(2, 2))
        cv3 = self.conv2d_nf_nf2(cv2)
        cv3 = self.nonlin3(cv3)

        cv4 = self.conv2d_nf2_nf2(cv3)
        # cv4_bcc = cv4.detach().clone()
        cv4 = self.bn4(cv4)

        # defining block 3
        cv4 = mean_pool(cv4, kernel_size=(2, 2), strides=(2, 2))
        cv5 = self.conv2d_nf2_nf3(cv4)
        cv5 = self.nonlin5(cv5)

        cv6 = self.conv2d_nf3_nf3(cv5)
        # cv6_bcc = cv6.detach().clone()
        cv6 = self.bn6(cv6)

        # defining the final  block
        cv7 = self.conv2d_nf3_ncl(cv6)
        real = sum_magnitudes(cv7)
        cv7 = torch.mean(real, dim=(1, 2, 3, 4))
        return cv7 + self.bias.view(1, -1)
