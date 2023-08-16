import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.linalg import dft


def h_conv(X, W, strides=(1, 1, 1, 1), padding=0, max_order=1):
    """
    This functions performs harmonic convolution operation between the input X
    and weights W. Harmonic convolutions between different orders, also referred
    as cross-stream convolutions, can be converted into a single convolution. 
    See Worrall et al, CVPR 2017 for details.

    Args:
        X (pytorch tensor): Input image tensor of shape (bs,h,w,order,complex,channels)
        W (dict of pytorch tensors): Contains tensors for m orders for filters as well
        as the respective phases. 
        strides (tuple of ints): tuple denoting strides for h and w directions. Similar
        to the original tf code as well as the pytorch tuple standard format for stride,
        we provide a 4-size tuple here. The dimensions N and c as per convention are 
        also set to 1.(default (1,1,1,1))
        padding: as per the Pytorch conv2d convention (default: 0)
        max_order: max. order of roation to be modeled(default: 1)

    Returns:
        Y: 
    """

    Xsh = list(X.size())
    X_ = X.view(Xsh[:3]+[-1])  # flatten out the last 3 dimensions

    # To convert the stream convolutions into a stacked single filter, we
    # combine the components of real and imaginary parts
    W_ = []
    for out_order in range(max_order+1):
        # For each output order build input
        Wr = []
        Wi = []
        for inp_order in range(Xsh[3]):
            # Difference in orders is the convolution order
            weight_order = out_order - inp_order
            weights = W[np.abs(weight_order)]
            sign = np.sign(weight_order)

            if Xsh[4] == 2:
                Wr += [weights[0], -sign*weights[1]]
                Wi += [sign*weights[1], weights[0]]
            else:
                Wr += [weights[0]]
                Wi += [weights[1]]

        W_ += [torch.cat(Wr, 2), torch.cat(Wi, 2)]
    W_ = torch.cat(W_, 3)

    # Convolving the constructed weights and feature map
    W_ = W_.permute(3, 2, 0, 1)
    W_ = W_.type(torch.cuda.FloatTensor) if torch.cuda.is_available() \
        else W_.type(torch.FloatTensor)

    X_ = X_.permute(0, 3, 1, 2)
    Y = torch.nn.functional.conv2d(X_, W_, stride=strides, padding=padding)
    Y = Y.permute(0, 2, 3, 1)
    # Reshae results into appropriate format
    Ysh = list(Y.size())
    new_shape = Ysh[:3] + [max_order+1, 2] + [Ysh[3]//(2*(max_order+1))]
    Y = Y.view(*new_shape)
    return Y


def avg_pool(X, kernel_size=(1, 1), strides=(1, 1)):
    '''
    Performs average pooling across the real as well as imaginary-valued feature maps.

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

    Xsh = list(X.size())
    # Collapse output the order, complex, and channel dimensions
    X_ = X.view(*(Xsh[:3]+[-1]))
    X_ = X_.permute(0, 3, 1, 2)
    Y = F.avg_pool2d(X_, kernel_size=kernel_size, stride=strides, padding=0)
    Y = Y.permute(0, 2, 3, 1)
    Ysh = list(Y.size())
    new_shape = Ysh[:3] + Xsh[3:]
    Y = Y.view(*new_shape)
    return Y


def concat_feature_magnitudes(X, eps=1e-12, keep_dims=True):
    '''
    Concat the complex feature magnitudes in X.

    Args:
        X (dict): contains feature maps for the real and imaginary components 
        for different rotation orders.
        eps (float): regularization term for min clamping

    Returns:
        R (torch tensor): concatenated feature maps
    '''

    R = torch.sum(torch.mul(X, X), dim=(4,), keepdim=keep_dims)
    R = torch.sqrt(torch.clamp(R, min=eps))
    return R


def get_interpolation_weights(fs, m, n_rings=None):
    '''
    Used to construct the steerable filters using Radial basis functions.
    The filters are constructed on the patches of n_rings using Gaussian
    interpolation. (Code adapted from the tf code of Worrall et al, CVPR, 2017)

    Args:
        fs (int): filter size for the H-net convoutional layer
        m (int): max. rotation order for the steerbable filters
        n_rings (int): No. of rings for the steerbale filters

    Returns:
        norm_weights (numpy): contains normalized weights for interpolation
        using the steerable filters
    '''

    if n_rings is None:
        n_rings = np.maximum(fs/2, 2)

    # We define below radii up to n_rings-0.5 (as in Worrall et al, CVPR 2017)
    radii = np.linspace(m != 0, n_rings-0.5, n_rings)

    # We define pixel centers to be at positions 0.5
    center_pt = np.asarray([fs, fs])/2.

    # Extracting the set of angles to be sampled
    N = get_sample_count(fs)

    # Choosing the sampling locations for the rings
    lin = (2*np.pi*np.arange(N))/N
    ring_locations = np.vstack([-np.sin(lin), np.cos(lin)])

    # Create interpolation coefficient coordinates
    coords = get_l2_neighbors(center_pt, fs)

    # getting samples based on the choisen center_pt and the coords
    radii = radii[:, np.newaxis, np.newaxis, np.newaxis]
    ring_locations = ring_locations[np.newaxis, :, :, np.newaxis]
    diff = radii*ring_locations - coords[np.newaxis, :, np.newaxis, :]
    dist2 = np.sum(diff**2, axis=1)

    # Convert distances to weightings
    weights = np.exp(-0.5*dist2/(0.5**2))  # For bandwidth of 0.5

    # Normalizing the weights to calibrate the different steerable filters
    norm_weights = weights/np.sum(weights, axis=2, keepdims=True)
    return norm_weights


def get_filter_weights(R_dict, fs, P=None, n_rings=None):
    '''
    Calculates filters in the form of weight matrices through performing
    single-frequency DFT on every ring obtained from sampling in the polar 
    domain. 

    Args:
        R_dict (dict): contains initialization weights
        fs (int): filter size for the h-net convolutional layer

    Returns:
        W (dict): contains the filter matrices
    '''

    k = fs
    W = {}  # dict to store the filter matrices
    N = get_sample_count(k)

    for m, r in R_dict.items():
        rsh = list(r.size())

        # Get the basis matrices built from the steerable filters
        weights = get_interpolation_weights(k, m, n_rings=n_rings)
        DFT = dft(N)[m, :]
        low_pass_filter = np.dot(DFT, weights).T

        cos_comp = np.real(low_pass_filter).astype(np.float32)
        sin_comp = np.imag(low_pass_filter).astype(np.float32)

        # Arranging the two components in a manner that they can be directly
        #  multiplied with the steerable weights
        cos_comp = torch.from_numpy(cos_comp)
        cos_comp = cos_comp.to(
            device="cuda" if torch.cuda.is_available() else "cpu")
        sin_comp = torch.from_numpy(sin_comp)
        sin_comp = sin_comp.to(
            device="cuda" if torch.cuda.is_available() else "cpu")

        # Computng the projetions on the rotational basis
        r = r.view(rsh[0], rsh[1]*rsh[2])
        ucos = torch.matmul(cos_comp, r).view(k, k, rsh[1], rsh[2]).double()
        usin = torch.matmul(sin_comp, r).view(k, k, rsh[1], rsh[2]).double()

        if P is not None:
            # Rotating the basis matrices
            ucos_ = torch.cos(P[m])*ucos + torch.sin(P[m])*usin
            usin = -torch.sin(P[m])*ucos + torch.cos(P[m])*usin
            ucos = ucos_
        W[m] = (ucos, usin)

    return W


def get_sample_count(fs):
    '''
    Calculates the number of points to be samples.

    Args:
        fs: filter size

    Returns: 
        n_samples: numeber of points to be sampled on the grid
    '''

    n_samples = np.maximum(np.ceil(np.pi*fs), 101)
    return n_samples


def get_l2_neighbors(center, shape):
    '''
    Creates a grid of indices for the neighbors

    Args:
        center (int, int): denotes indeices for the center point 
        shape (int): number of points along each dimesion

    Returns:
        l2_grid (numpy): contains indices of the neighbors in grid
    '''

    # Get the neighborhood indices
    lin = np.arange(shape)+0.5
    jj, ii = np.meshgrid(lin, lin)
    ii = ii - center[1]
    jj = jj - center[0]
    l2_grid = np.vstack((np.reshape(ii, -1), np.reshape(jj, -1)))
    return l2_grid
