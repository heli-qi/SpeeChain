"""
    Author: Sashi Novitasari
    Affiliation: NAIST (-2022)
    Date: 2022.08

    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.09
"""
from typing import List
import torch
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from speechain.module.abs import Module
from speechain.module.prenet.linear import LinearPrenet


class Conv1dEv(torch.nn.Module):
    """
        A 1D convolutional layer with support for different padding modes.

        Attributes:
            cutoff (bool):
                Indicates whether the output should be cut off for the 'same' padding mode.
            causal_padding (int):
                Additional padding required for the 'causal' padding mode.
            dilation (int):
                The dilation rate of the convolutional layer.
            conv_lyr (torch.nn.Conv1d):
                The 1D convolutional layer.

    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, dilation: int = 1, padding_mode: str = 'same',
                 bias: bool = True, use_weight_norm: bool = False, groups: int = 1):
        """
            Initializes the Conv1dEv module with the specified parameters.

            Args:
                in_channels (int):
                    Number of channels in the input feature.
                out_channels (int):
                    Number of channels produced by the convolution.
                kernel_size (int):
                    Size of the convolutional kernel.
                stride (int, optional):
                    Stride of the convolution. Defaults to 1.
                dilation (int, optional):
                    The dilation rate of the kernel. Defaults to 1.
                padding_mode (str, optional):
                    Padding mode. Supported values are 'valid', 'full', 'same' and 'causal'. Defaults to 'same'.
                bias (bool, optional):
                    If True, adds a learnable bias to the output. Defaults to True.

            Raises:
                ValueError: If an unsupported padding mode is specified.
        """
        super().__init__()

        self.cutoff = False
        self.causal_padding = 0
        self.dilation = dilation

        # no padding is used
        if padding_mode == 'valid':
            padding = 0
        # full padding
        elif padding_mode == 'full':
            padding = dilation * (kernel_size - 1)
        # same padding, the output is the same in dimension with input
        elif padding_mode == 'same':
            assert stride == 1, "Stride should be 1 for 'same' padding mode"
            if kernel_size % 2 == 0:
                padding = dilation * kernel_size // 2
                self.cutoff = True
            else:
                padding = dilation * (kernel_size - 1) // 2
        # causal padding
        elif padding_mode == 'causal':
            padding = 0
            self.causal_padding = dilation * (kernel_size - 1)
        else:
            raise ValueError("Unsupported padding mode. Supported modes are 'valid', 'full', 'same' and 'causal'.")

        self.conv_lyr = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias, groups=groups
        )
        if use_weight_norm:
            self.conv_lyr = weight_norm(self.conv_lyr)

    def forward(self, feat: torch.Tensor):
        """
        Performs a forward pass through the convolutional layer.

        Args:
            feat (torch.Tensor):
                The input feature tensor. Shape: (batch, feat_dim, feat_maxlen).

        Returns:
            torch.Tensor:
                The output tensor. Shape: (batch, out_channels, output_len).

        """
        # attach additional paddings at the end for the 'causal' padding mode
        if self.causal_padding > 0:
            feat = F.pad(feat, (self.causal_padding, 0))
        output = self.conv_lyr(feat)
        # cut off the redundant tails for the 'same' padding mode
        if self.cutoff:
            output = output[:, :, :-self.dilation]
        return output


class Conv1dPrenet(Module):
    """
        The Conv1d prenet. Usually used before the TTS encoder.
        This prenet is made up of two parts:
            1. (mandatory) The Conv1d part contains one or more Conv1d blocks which are composed of the components below
                1. (mandatory) a Conv1d layer
                2. (optional) a BatchNorm1d layer
                3. (optional) an activation function
                4. (optional) a Dropout layer.
            2. (optional) The Linear part contains one or more Linear blocks which are composed of the components below
                1. (mandatory) a Linear layer
                2. (optional) an activation function
                3. (optional) a Dropout layer.

        Reference:
            Neural Speech Synthesis with Transformer Network
            https://ojs.aaai.org/index.php/AAAI/article/view/4642/4520
    """

    def module_init(self,
                    feat_dim: int = None,
                    conv_dims: int or List[int] = [512, 512, 512],
                    conv_kernel: int = 5,
                    conv_stride: int = 1,
                    conv_batchnorm: bool = True,
                    conv_activation: str = 'ReLU',
                    conv_dropout: float or List[float] = None,
                    lnr_dims: int or List[int] = -1,
                    lnr_activation: str = None,
                    lnr_dropout: int or List[int] = None,
                    zero_centered: bool = False):
        """

        Args:
            feat_dim: int
                The dimension of input acoustic feature tensors.
                Used for calculating the in_features of the first Linear layer.
            conv_dims: List[int] or int
                The values of out_channels of each Conv1d layer.
                If a list of integers is given, multiple Conv1d layers will be initialized.
                If an integer is given, there will be only one Conv1d layer
            conv_kernel: int
                The value of kernel_size of all Conv1d layers.
            conv_stride: int
                The value of stride of all Conv1d layers.
            conv_batchnorm: bool
                Whether a BatchNorm1d layer is added right after a Conv1d layer
            conv_activation: str
                The type of the activation function after all Conv1d layers.
                None means no activation function is needed.
            conv_dropout: float or List[float]
                The values of p rate of the Dropout layer after each Linear layer.
            lnr_dims: int or List[int]
                The values of out_features of each Linear layer.
                The first value in the List represents the out_features of the first Linear layer.
                -1: same size as the last convolutional layer's dim
            lnr_activation: str
                The type of the activation function after all Linear layers.
                None means no activation function is needed.
            lnr_dropout: float or List[float]
                The values of p rate of the Dropout layer after each Linear layer.
            zero_centered: bool
                Whether the output of this module is centered at 0.
                If the specified activation function changes the centroid of the output distribution, e.g. ReLU and
                LeakyReLU, the activation function won't be attached to the final Linear layer if zer_centered is set
                to True.

        """
        # --- 0. Argument Checking --- #
        # Convolution arguments checking
        assert isinstance(conv_dims, (List, int)), \
            "The dimensions of convolutional layers must be given as a list of integers or an integer!"
        assert isinstance(conv_kernel, int), \
            "The sizes of convolutional kernels must be given as an integer!"
        assert isinstance(conv_stride, int), \
            "The lengths of convolutional strides must be given as an integer!"
        if conv_dropout is not None:
            assert isinstance(conv_dropout, (List, float)), \
                "The dropout rates of convolutional layers must be given as a list of integers or an integer!"

        # Linear arguments checking
        if lnr_dropout is not None:
            assert isinstance(lnr_dropout, (List, float)), \
                "The dropout rates of linear layers must be given as a list of integers or an integer!"
        if lnr_dims is not None:
            assert isinstance(lnr_dims, (List, int)), \
                "The dimensions of linear layers must be given as a list of integers or an integer!"

        # input_size initialization
        if self.input_size is not None:
            feat_dim = self.input_size
        else:
            assert feat_dim is not None

        # --- 1. Convolutional Part Initialization --- #
        # register convolution arguments
        self.conv_dims = conv_dims if isinstance(conv_dims, List) else [conv_dims]
        self.conv_kernel = conv_kernel
        self.conv_stride = conv_stride
        self.conv_dropout = conv_dropout

        # Conv1d blocks construction
        _prev_dim = feat_dim
        _tmp_conv = []
        for i in range(len(self.conv_dims)):
            _tmp_conv.append(
                # don't include bias in the convolutional layer if it is followed by a batchnorm layer
                # reference: https://stackoverflow.com/questions/46256747/can-not-use-both-bias-and-batch-normalization-in-convolution-layers
                Conv1dEv(in_channels=_prev_dim,
                         out_channels=self.conv_dims[i],
                         kernel_size=self.conv_kernel,
                         stride=self.conv_stride,
                         padding_mode='same',
                         bias=not conv_batchnorm)
            )
            # BatchNorm is better to be placed before activation
            # reference: https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
            if conv_batchnorm:
                _tmp_conv.append(torch.nn.BatchNorm1d(self.conv_dims[i]))
            if conv_activation is not None:
                # no 'ReLU'-series activation is added for the last layer if zero_centered is specified
                if not (i == len(self.conv_dims) - 1 and lnr_dims is None) or \
                        not (zero_centered and 'ReLU' in conv_activation):
                    _tmp_conv.append(getattr(torch.nn, conv_activation)())
            if conv_dropout is not None:
                _tmp_conv.append(torch.nn.Dropout(
                    p=self.conv_dropout if not isinstance(self.conv_dropout, List) else self.conv_dropout[i]
                ))
            _prev_dim = conv_dims[i]
        self.conv = torch.nn.Sequential(*_tmp_conv)
        self.output_size = _prev_dim

        # --- 2. Linear Part Initialization --- #
        if lnr_dims is not None:
            lnr_dims = lnr_dims if isinstance(lnr_dims, List) else [lnr_dims]
            for i in range(len(lnr_dims)):
                _prev_dim = self.conv_dims[-1] if i == 0 else lnr_dims[i - 1]
                if lnr_dims[i] == -1:
                    lnr_dims[i] = _prev_dim

            self.linear = LinearPrenet(feat_dim=self.output_size,
                                       lnr_dims=lnr_dims,
                                       lnr_activation=lnr_activation,
                                       lnr_dropout=lnr_dropout,
                                       zero_centered=zero_centered)
            self.output_size = self.linear.output_size

    def forward(self, feat: torch.Tensor, feat_len: torch.Tensor):
        """

        Args:
            feat: (batch, feat_maxlen, feat_dim)
                The input feature tensors.
            feat_len: (batch,)
                The length of each feature tensor.

        Returns: feat, feat_len
            The embedded feature vectors with their lengths.

        """
        # forward the convolutional layers
        # (batch, feat_maxlen, feat_dim) -> (batch, feat_dim, feat_maxlen)
        feat = feat.transpose(1, 2)
        # (batch, feat_dim, feat_maxlen) -> (batch, conv_dim, feat_maxlen)
        feat = self.conv(feat)
        # (batch, conv_dim, feat_maxlen) -> (batch, feat_maxlen, conv_dim)
        feat = feat.transpose(1, 2)

        # forward the linear layers
        if hasattr(self, 'linear'):
            # (batch, feat_maxlen, conv_dim) -> (batch, feat_maxlen, lnr_dim)
            feat, feat_len = self.linear(feat, feat_len)

        # return both feat & feat_len for the compatibility with other prenet
        return feat, feat_len
