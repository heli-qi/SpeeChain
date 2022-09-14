"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
from typing import List
import torch
import math

from speechain.module.abs import Module
from speechain.utilbox.train_util import generator_act_module


class Conv2dPrenet(Module):
    """
        The Conv2d prenet. Usually used before the Transformer ASR encoder.
        There are two parts in this prenet:
            1. Conv2d layers. Each Conv2d layer is followed by a BatchNorm2d layer and an activation function.
                We don't include a Dropout layer after each Conv2d layer.
            2. Linear layers. Each Linear layer is followed by an activation function and a Dropout layer.

        Reference:
            Speech-transformer: a no-recurrence sequence-to-sequence model for speech recognition
            https://ieeexplore.ieee.org/abstract/document/8462506/
    """

    def module_init(self,
                    conv_dims: int or List[int],
                    feat_dim: int = None,
                    conv_kernel: int or List[int] = 3,
                    conv_stride: int or List[int] = 2,
                    conv_activation: str = 'ReLU',
                    conv_batchnorm: bool = True,
                    lnr_dims: int or List[int] = None,
                    lnr_activation: str = 'ReLU',
                    lnr_dropout: float or List[float] = None):
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
            conv_activation: str
                The type of the activation function after all Conv2d layers.
                None means no activation function is needed.
            conv_batchnorm: bool
                Whether a BatchNorm2d layer is added between a Conv2d layer and a Dropout layer
            lnr_dims: int or List[int]
                The values of out_features of each Linear layer.
                The first value in the List represents the out_features of the first Linear layer.
            lnr_activation: str
                The type of the activation function after all Linear layers.
                None means no activation function is needed.
            lnr_dropout: float or List[float]
                The values of p rate of the Dropout layer after each Linear layer.

        """
        # Convolution arguments checking
        assert isinstance(conv_dims, (List, int)), \
            "The dimensions of convolution layers must be given as a list of integers or an integer!"
        assert isinstance(conv_kernel, (List, int)), \
            "The sizes of convolution kernels must be given as a list of integers or an integer!"
        assert isinstance(conv_stride, (List, int)), \
            "The lengths of convolution strides must be given as a list of integers or an integer!"
        # Linear arguments checking
        if lnr_dims is not None:
            assert isinstance(lnr_dims, (List, int)), \
                "The dimensions of linear layers must be given as a list of integers or an integer!"
            if lnr_dropout is not None:
                assert isinstance(lnr_dropout, (List, float)), \
                    "The dropout rates of linear layers must be given as a list of integers or an integer!"

        # input_size initialization
        if self.input_size is not None:
            feat_dim = self.input_size
        else:
            assert feat_dim is not None

        # register convolution arguments
        self.conv_dims = conv_dims if isinstance(conv_dims, List) else [conv_dims]
        self.conv_kernel = tuple(conv_kernel) if isinstance(conv_kernel, List) else (conv_kernel, conv_kernel)
        self.conv_stride = tuple(conv_stride) if isinstance(conv_stride, List) else (conv_stride, conv_stride)
        # register linear arguments
        if lnr_dims is not None:
            self.lnr_dims = lnr_dims if isinstance(lnr_dims, List) else [lnr_dims]
        self.lnr_dropout = lnr_dropout if lnr_dropout is not None else 0.0

        # Conv2d layers initialization
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
                                bias=not conv_batchnorm))
            # BatchNorm is better to be placed before activation
            # reference: https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
            if conv_batchnorm:
                _tmp_conv.append(torch.nn.BatchNorm2d(self.conv_dims[i]))
            _tmp_conv.append(generator_act_module(conv_activation))
            # we do not apply a dropout layer after each conv layer

            _prev_dim = self.conv_dims[i]
        self.conv = torch.nn.Sequential(*_tmp_conv)

        # feature dimension recalculation after convolutional layers
        for _ in self.conv_dims:
            feat_dim = (feat_dim - self.conv_kernel[-1]) // self.conv_stride[-1] + 1
        _prev_dim *= feat_dim

        # Linear layers initialization
        if not hasattr(self, 'lnr_dims'):
            self.output_size = _prev_dim
        else:
            _tmp_lnr = []
            for i in range(len(self.lnr_dims)):
                _tmp_lnr.append(torch.nn.Linear(in_features=_prev_dim, out_features=self.lnr_dims[i]))
                # The order of activation function and dropout layer is somewhat not a big deal
                # a useful blog: https://sebastianraschka.com/faq/docs/dropout-activation.html
                _tmp_lnr.append(generator_act_module(lnr_activation))
                _tmp_lnr.append(torch.nn.Dropout(
                    p=self.lnr_dropout if not isinstance(self.lnr_dropout, List) else self.lnr_dropout[i]
                ))
                _prev_dim = self.lnr_dims[i]

            self.linear = torch.nn.Sequential(*_tmp_lnr)
            self.output_size = self.lnr_dims[-1]


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
        feat = feat.unsqueeze(1)
        feat = self.conv(feat)
        batch, channels, feat_maxlen, feat_dim = feat.size()
        feat = feat.transpose(1, 2).contiguous().view(batch, feat_maxlen, -1)

        # modify the feature length
        for _ in self.conv_dims:
            # torch version of 'feat_len = (feat_len - self.conv_kernel[0]) // self.conv_stride[0] + 1'
            feat_len = torch.div(feat_len - self.conv_kernel[0], self.conv_stride[0], rounding_mode='floor') + 1

        # check the input feat length
        if max(feat_len) != feat.size(1):
            raise RuntimeError(f"There is a bug in the {self.__class__.__name__}."
                               f"The calculation of the feature lengths has something wrong.")

        # forward the linear layers if have
        if hasattr(self, 'linear'):
            feat = self.linear(feat)

        return feat, feat_len
