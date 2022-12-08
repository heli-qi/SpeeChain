"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
from typing import List
import torch

from speechain.module.abs import Module
from speechain.module.prenet.linear import LinearPrenet


class Conv2dPrenet(Module):
    """
        The Conv2d prenet. Usually used before the ASR encoder.
        This prenet is made up of two parts:
            1. (mandatory) The Conv2d part contains one or more Conv2d blocks which are composed of the components below
                1. (mandatory) a Conv2d layer
                2. (optional) a BatchNorm2d layer
                3. (optional) an activation function
                4. (optional) a Dropout layer
            2. (optional) The Linear part contains one or more Linear blocks which are composed of the components below
                1. (mandatory) a Linear layer
                2. (optional) an activation function
                3. (optional) a Dropout layer.

        Reference:
            Speech-transformer: a no-recurrence sequence-to-sequence model for speech recognition
            https://ieeexplore.ieee.org/abstract/document/8462506/
    """

    def module_init(self,
                    feat_dim: int = None,
                    conv_dims: int or List[int] = [64, 64],
                    conv_kernel: int or List[int] = 3,
                    conv_stride: int or List[int] = 2,
                    conv_padding: str or int or List[int] = 0,
                    conv_batchnorm: bool = True,
                    conv_activation: str = 'ReLU',
                    conv_dropout: float or List[float] = None,
                    lnr_dims: int or List[int] = 512,
                    lnr_activation: str = None,
                    lnr_dropout: float or List[float] = None,
                    zero_centered: bool = False):
        """

        Args:
            feat_dim: int
                The dimension of input acoustic feature tensors.
                Used for calculating the in_features of the first Linear layer.
            conv_dims: List[int] or int
                The values of out_channels of each Conv2d layer.
                If a list of integers is given, multiple Conv2d layers will be initialized.
                If an integer is given, there will be only one Conv2d layer
            conv_kernel: int or List[int]
                The value of kernel_size of all Conv2d layers.
                An integer means the same kernel size for time and frequency dimension.
                List[int] is needed if you would like to make different dimensions have different kernel sizes.
            conv_stride: int or List[int]
                The value of stride of all Conv2d layers.
                An integer means the same stride for time and frequency dimension.
                List[int] is needed if you would like to make different dimensions have different strides.
            conv_padding: str or int or List[int]
                The padding added to all four sides of the input. It can be either a string {‘valid’, ‘same’} or a
                list of integers giving the amount of implicit padding applied on both sides.
            conv_batchnorm: bool
                Whether a BatchNorm2d layer is added after each Conv2d layer
            conv_activation: str
                The type of the activation function after all Conv2d layers.
                None means no activation function is needed.
            conv_dropout: float or List[float]
                The values of p rate of the Dropout layer after each Linear layer.
            lnr_dims: int or List[int]
                The values of out_features of each Linear layer.
                The first value in the List represents the out_features of the first Linear layer.
            lnr_activation: str
                The type of the activation function after all Linear layers. None means no activation function is needed.
                For transformer training, it's better not to add a non-negative ReLU activation function to the last
                linear layer because the ReLU activation will make the range of the output (>= 0) different from the
                sinusoidal positional encoding [-1, 1]. For more details, please refer to Section 3.3 of the paper below:
                    'Neural Speech Synthesis with Transformer Network'
                    https://ojs.aaai.org/index.php/AAAI/article/view/4642/4520
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
        assert isinstance(conv_kernel, (List, int)), \
            "The sizes of convolutional kernels must be given as a list of integers or an integer!"
        assert isinstance(conv_stride, (List, int)), \
            "The lengths of convolutional strides must be given as a list of integers or an integer!"
        assert isinstance(conv_padding, (List, int, str)), \
            "The lengths of convolutional paddings must be given as a list of integers, an integer, or a string!"
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
        elif feat_dim is None:
            raise RuntimeError

        # --- 1. Convolutional Part Initialization --- #
        # register convolution arguments
        self.conv_dims = conv_dims if isinstance(conv_dims, List) else [conv_dims]
        self.conv_kernel = tuple(conv_kernel) if isinstance(conv_kernel, List) else (conv_kernel, conv_kernel)
        self.conv_stride = tuple(conv_stride) if isinstance(conv_stride, List) else (conv_stride, conv_stride)
        if not isinstance(conv_padding, str):
            self.conv_padding = tuple(conv_padding) if isinstance(conv_padding, List) else (conv_padding, conv_padding)
        self.conv_dropout = conv_dropout

        # Conv2d blocks construction
        _prev_dim = 1
        _tmp_conv = []
        for i in range(len(self.conv_dims)):
            _tmp_conv.append(
                # don't include bias in the convolutional layer if it is followed by a batchnorm layer
                # reference: https://stackoverflow.com/questions/46256747/can-not-use-both-bias-and-batch-normalization-in-convolution-layers
                torch.nn.Conv2d(in_channels=_prev_dim,
                                out_channels=self.conv_dims[i],
                                kernel_size=self.conv_kernel,
                                stride=self.conv_stride,
                                padding=self.conv_padding,
                                bias=not conv_batchnorm)
            )
            # BatchNorm is better to be placed before activation
            # reference: https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
            if conv_batchnorm:
                _tmp_conv.append(torch.nn.BatchNorm2d(self.conv_dims[i]))
            if conv_activation is not None:
                # no 'ReLU'-series activation is added for the last layer if zero_centered is specified
                if not (i == len(self.conv_dims) - 1 and lnr_dims is None) or \
                        not (zero_centered and 'ReLU' in conv_activation):
                    _tmp_conv.append(getattr(torch.nn, conv_activation)())
            if conv_dropout is not None:
                _tmp_conv.append(torch.nn.Dropout(
                    p=self.conv_dropout if not isinstance(self.conv_dropout, List) else self.conv_dropout[i]
                ))
            _prev_dim = self.conv_dims[i]
        self.conv = torch.nn.Sequential(*_tmp_conv)

        # feature dimension recalculation after convolutional layers
        for _ in self.conv_dims:
            feat_dim = (feat_dim - self.conv_kernel[-1]) // self.conv_stride[-1] + 1
        _prev_dim *= feat_dim
        self.output_size = _prev_dim

        # --- 2. Linear Part Initialization --- #
        if lnr_dims is not None:
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
                The input acoustic feature tensors.
            feat_len: (batch,)
                The length of each acoustic feature tensor.

        Returns:
            The embedded feature vectors with their lengths.

        """
        # forward the convolutional layers
        # (batch, feat_maxlen, feat_dim) -> (batch, 1, feat_maxlen, feat_dim)
        feat = feat.unsqueeze(1)
        # (batch, 1, feat_maxlen, feat_dim) -> (batch, conv_dim, feat_maxlen_after, feat_dim_after)
        feat = self.conv(feat)
        batch, channels, feat_maxlen, feat_dim = feat.size()
        # (batch, conv_dim, feat_maxlen_after, feat_dim_after) -> (batch, feat_maxlen_after, feat_dim_after × conv_dim)
        feat = feat.transpose(1, 2).contiguous().view(batch, feat_maxlen, -1)

        # modify the feature length
        for _ in self.conv_dims:
            # torch version of 'feat_len = (feat_len - self.conv_kernel[0]) // self.conv_stride[0] + 1'
            feat_len = torch.div(feat_len - self.conv_kernel[0], self.conv_stride[0], rounding_mode='floor') + 1

        # check the input feat length
        if max(feat_len) != feat.size(1):
            raise RuntimeError(f"There is a bug in the {self.__class__.__name__}."
                               f"The calculation of the feature lengths has something wrong.")

        # forward the linear layers
        # (batch, feat_maxlen_after, feat_dim_after × conv_dim) -> (batch, feat_maxlen_after, lnr_dim)
        if hasattr(self, 'linear'):
            feat, feat_len = self.linear(feat, feat_len)

        return feat, feat_len
