import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .hnet_ops import *
from .hnet_ops import h_conv


class Conv2d(nn.Module):
    '''Defining custom convolutional layer'''

    @staticmethod
    def get_weights(fs, W_init=None, std_scale=0.4):
        """
        Initializes the weights using He initialization method

        Args:
            fs (list of ints): filter shape expressed as a list of dimensions
            W_init (torch tensor): Contains initial values for the weights (default None)
            std_scale (float): multiplier for weight standard deviation (default 0.4)

        Returns: 
            W_init (torch tensor): intialized weights converted into nn parameters
        """

        if W_init == None:
            stddev = std_scale*np.sqrt(2.0 / np.prod(fs[:3]))
            W_init = torch.normal(torch.zeros(*fs), stddev)

        W_init = nn.Parameter(W_init)
        return W_init

    @staticmethod
    def init_weights_dict(layer_shape, max_order, std_scale=0.4, ring_count=None):
        """
        Initializes the dict of weights

        Args: 
            layer_shape (list of ints): contains dimensions of the layer as a list
            max_order: maximum rotation order e.g. max_order=2 uses 0,1,2 (default 1)
            std_scale: multiplier for weight standard deviation (default 0.4)
            ring_count (int): Number of rings for calculating the basis
        Returns:
            weights_dict: initialized dict of weights
        """

        if isinstance(max_order, int):
            rotation_orders = range(-max_order, max_order+1)
        else:
            diff_order = max_order[1]-max_order[0]
            rotation_orders = range(-diff_order, diff_order+1)
        weights_dict = {}
        for order in rotation_orders:
            if ring_count is None:
                ring_count = np.maximum(layer_shape[0]/2, 2)
            sh = [ring_count,] + list(layer_shape[2:])
            weights_dict[order] = Conv2d.get_weights(sh, std_scale=std_scale)
        return weights_dict

    @staticmethod
    def init_phase_dict(inp_channel_count, out_channel_count, max_order):
        """
        Initializes phase dict with phase offsets

        Args:
            inp_channel_count (int): number of input channels
            out_channel_count (int): number of output channels
            max_order (int): maximum rotation order e.g. max_order=2 uses 0,1,2 (default 1)

        Returns:
            phase_dict: initialized dict of phase offsets
        """

        if isinstance(max_order, int):
            rotation_orders = range(-max_order, max_order+1)
        else:
            diff_order = max_order[1]-max_order[0]
            rotation_orders = range(-diff, diff+1)
        phase_dict = {}
        for order in rotation_orders:
            init = np.random.rand(1, 1, inp_channel_count,
                                  out_channel_count) * 2. * np.pi
            init = torch.from_numpy(init)
            phase = nn.Parameter(init)
            phase_dict[order] = phase
        return phase_dict

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, phase=True,
                 max_order=1, stddev=0.4, n_rings=None):
        super().__init__()
        '''
        initializer function for Harmonic convolution lite wrapper class

        Args:
            in_channels (int): Number of input channels 
            out_channels (int ): Number of output channels (int)
            kernel_size (int): dimensions of the (square) filter
            stride (tuple of ints): tuple denoting strides for h and w directions. Similar
            to the original tf code as well as the pytorch tuple standard format for stride,
            we provide a 4-size tuple here. The dimensions N and c as per convention are 
            also set to 1.(default (1,1,1,1))
            padding (int): amount of implicit zero paddings on both sides
            phase (boolean): to decide whether the phase-offset term is used (default True)
            max_order (int): max. order of roation to be modeled(default: 1)
            stddev (float): scaling term for He weight initialization
        '''

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.phase = phase
        self.max_order = max_order
        self.stddev = stddev
        self.n_rings = n_rings
        self.shape = (kernel_size, kernel_size,
                      self.in_channels, self.out_channels)
        self.Q = Conv2d.init_weights_dict(
            self.shape, self.max_order, self.stddev, self.n_rings)
        for k, v in self.Q.items():
            self.register_parameter('weights_dict_' + str(k), v)
        if self.phase:
            self.P = Conv2d.init_phase_dict(
                self.in_channels, self.out_channels, self.max_order)
            for k, v in self.P.items():
                self.register_parameter('phase_dict_' + str(k), v)
        else:
            self.P = None

    def forward(self, X):
        '''
        Forward propagation function for the harmonic convolution operation

        Args:
            X (torch tensor): input feature tensor

        Returns:
            R (torch tensor): output feature tensor obtained from harmonic convolution 
        '''

        W = get_filter_weights(self.Q, fs=self.kernel_size,
                               P=self.P, n_rings=self.n_rings)
        R = h_conv(X, W, strides=self.stride,
                   padding=self.padding, max_order=self.max_order)
        return R


class HNonlin(nn.Module):
    '''
    Nonlinear activation module class based on func handle

    Args:
        X: dict of channels {rotation order: (real, imaginary)}
        fnc: function handle for a nonlinearity. MUST map to non-negative reals R+
        eps: regularization since grad |Z| is infinite at zero (default 1e-8)
    '''

    def __init__(self, fnc, rotation_order, channels, eps=1e-12):
        super().__init__()
        '''
        Intializer function for getting iput details and nonlinear fn handle

        Args:
            fnc (handle): function handle for nonlinear activation
            rotation_order (int): defines the order of rotation nbeing modeled 
            channels (int): number of channels
        '''

        self.fnc = fnc
        self.eps = eps

        # creating bias parameter to add and initializing using xavier normal method
        self.b = nn.Parameter(torch.FloatTensor(
            1, 1, 1, rotation_order, 1, channels))
        nn.init.xavier_normal_(self.b)

    def forward(self, X):
        '''
        forward propagation function for the nonlinearity defined for complex feature output

        Args:
            X (dict of channels): features maps for the complex mapping (real and imaginary)

        Returns: 
            Xnonlin: output feature map after applying nonlinearity
        '''
        magnitude = concat_feature_magnitudes(X, self.eps)
        Rb = magnitude + self.b
        c = self.fnc(Rb) / magnitude
        Xnonlin = c*X
        return Xnonlin


class BatchNorm(nn.Module):
    '''Batch norm module for complex-valued feature maps'''

    def __init__(self, rotation_order, cmplx, channels, fnc=F.relu, decay=0.99, eps=1e-4):
        '''
        Initialization function

        Args:
            rotation_order (int): defines the order for complex rotations
            cmplx (int): 
            channels (int): number of the channels in the feature map
            fnc: choice of activation function
            decay (float): defines the momentum term to compute the running_mean and 
            running_var terms
            eps (float): small term added to the denominator for numerical stability
        '''

        super().__init__()
        self.fnc = fnc
        self.eps = eps
        self.n_out = rotation_order, cmplx, channels
        self.tn_out = rotation_order*cmplx*channels
        self.bn = nn.BatchNorm1d(self.tn_out, eps=self.eps, momentum=1-decay)

    def forward(self, X: torch.Tensor):
        '''
        Model forward propogation function for batchnorm

        Args:
            X (torch tensor): Input image tensor of shape (bs,h,w,order,complex,channels)

        Returns:        
            Xnormed (torch tensor): Batch normalized output feature maps
        '''

        magnitude = concat_feature_magnitudes(X, self.eps)
        Xsh = tuple(X.size())
        assert Xsh[-3:] == self.n_out, (Xsh, self.n_out)
        X = X.reshape(-1, self.tn_out)
        Rb = self.bn(X)
        X = X.view(Xsh)
        Rb = Rb.view(Xsh)
        c = self.fnc(Rb) / magnitude
        Xnormed = c*X
        return Xnormed


def mean_pool(X, kernel_size=(1, 1), strides=(1, 1)):
    '''
    Performs avg pooling over the spatial dimensions

    Args:
        X (torch tensor): Input image tensor of shape (bs,h,w,order,complex,channels)
        kernel_size (int tuple): defines pooling kernel size
        strides (tuple of ints): tuple denoting strides for h and w directions. Similar
        to the original tf code as well as the pytorch tuple standard format for stride,
        we provide a 4-size tuple here. The dimensions N and c as per convention are 
        also set to 1.(default (1,1,1,1))
    Returns:
        Y (torch tensor): Output features after applying average pooling
    '''

    Y = avg_pool(X, kernel_size=kernel_size, strides=strides)
    return Y


def sum_magnitudes(X, eps=1e-12, keep_dims=True):
    '''
    Performs summation along the order dimension
    Args:
        X (torch tensor): contains concantenated feature maps for the real 
        and imaginary components across different rotation orders.
        eps (float): regularization term for min clamping
        keep_dim (boolean): to decide whether the dimension is to be retained or not

    Returns:
        R (torch tensor): output feature maps 
    '''

    R = torch.sum(torch.mul(X, X), dim=(4,), keepdim=keep_dims)
    R = torch.sum(torch.sqrt(torch.clamp(R, eps)), dim=(3,), keepdim=keep_dims)
    return R
