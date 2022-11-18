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

from speechain.module.abs import Module


class Conv1dEv(Module):
    def module_init(self,
                    in_channels: int, out_channels: int,
                    kernel_size, stride, padding: str, bias=True):
        """
        A guide to convolution arithmetic for deep learning
        https://arxiv.org/pdf/1603.07285v1.pdf

        """
        self.cutoff = False
        self.causal_padding = 0

        # no padding is used
        if padding == 'valid':
            padding = 0
        # full padding
        elif padding == 'full':
            padding = kernel_size - 1
        # same padding, the output is the same in dimension with input
        elif padding == 'same':
            padding = kernel_size // 2
            if kernel_size % 2 == 0:
                self.cutoff = True
        elif padding == 'causal':
            padding = 0
            self.causal_padding = kernel_size - 1

        self.conv_lyr = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
        )


    def forward(self, feat):
        """

        Args:
            feat: (batch, feat_dim, feat_maxlen)

        Returns:

        """
        if self.causal_padding > 0:
            feat = F.pad(feat, (self.causal_padding, 0))
        output = self.conv_lyr(feat)
        if self.cutoff:
            output = output[:, :, :-1]
        return output


class Conv1dPrenet(Module):
    """
        The Conv1d prenet for TTS. Usually used before the Transformer TTS encoder.
        There are two parts in this prenet:
            1. Conv1d layers.
                Each Conv1d layer is followed by a BatchNorm1d layer, an activation function, and a Dropout layer.
            2. Linear layers.
                Each Linear layer is followed by an activation function and a Dropout layer.

        Reference:
            Neural Speech Synthesis with Transformer Network
            https://ojs.aaai.org/index.php/AAAI/article/view/4642/4520
    """
    def module_init(self,
                    feat_dim: int = None,
                    conv_dims: int or List[int] = [512, 512, 512],
                    conv_kernel: int = 5,
                    conv_stride: int = 1,
                    conv_padding: str = 'same',
                    conv_batchnorm: bool = True,
                    conv_activation: str = 'ReLU',
                    conv_dropout: float or List[float] = None,
                    lnr_dims: int or List[int] = -1,
                    lnr_activation: str = None,
                    lnr_dropout: int or List[int] = None):
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

        """
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

        # register convolution arguments
        self.conv_dims = conv_dims if isinstance(conv_dims, List) else [conv_dims]
        self.conv_kernel = conv_kernel
        self.conv_stride = conv_stride
        self.conv_padding = conv_padding
        self.conv_dropout = conv_dropout
        # register linear arguments
        if lnr_dims is not None:
            self.lnr_dims = lnr_dims if isinstance(lnr_dims, List) else [lnr_dims]
            for i in range(len(self.lnr_dims)):
                _prev_dim = self.conv_dims[-1] if i == 0 else self.lnr_dims[i - 1]
                if self.lnr_dims[i] == -1:
                    self.lnr_dims[i] = _prev_dim
        self.lnr_dropout = lnr_dropout

        # Conv1d layers initialization
        _prev_dim = feat_dim
        _tmp_conv = []
        for i in range(len(self.conv_dims)):
            _tmp_conv.append(Conv1dEv(in_channels=_prev_dim, 
                                      out_channels=self.conv_dims[i],
                                      kernel_size=self.conv_kernel,
                                      stride=self.conv_stride,
                                      padding=self.conv_padding,
                                      bias=not conv_batchnorm))
            # order: BatchNorm1d -> Activation (ReLU) -> Dropout
            if conv_batchnorm:
                _tmp_conv.append(torch.nn.BatchNorm1d(self.conv_dims[i]))
            if conv_activation is not None:
                _tmp_conv.append(getattr(torch.nn, conv_activation)())
            if conv_dropout is not None:
                _tmp_conv.append(torch.nn.Dropout(
                    p=self.conv_dropout if not isinstance(self.conv_dropout, List) else self.conv_dropout[i]
                ))
            _prev_dim = conv_dims[i]

        self.conv = torch.nn.Sequential(*_tmp_conv)

        # Linear layers initialization
        if not hasattr(self, 'lnr_dims'):
            self.output_size = _prev_dim
        else:
            _tmp_lnr = []
            for i in range(len(self.lnr_dims)):
                _tmp_lnr.append(torch.nn.Linear(in_features=_prev_dim, out_features=self.lnr_dims[i]))
                # order: Activation (ReLU) -> Dropout
                if lnr_activation is not None:
                    _tmp_lnr.append(getattr(torch.nn, lnr_activation)())
                if lnr_dropout is not None:
                    _tmp_lnr.append(torch.nn.Dropout(
                        p=self.lnr_dropout if not isinstance(self.lnr_dropout, List) else self.lnr_dropout[i]
                    ))
                _prev_dim=self.lnr_dims[i]

            self.linear = torch.nn.Sequential(*_tmp_lnr)
            self.output_size = self.lnr_dims[-1]


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

        # forward the linear layers if have
        if hasattr(self, 'linear'):
            # (batch, feat_maxlen, conv_dim) -> (batch, feat_maxlen, lnr_dim)
            feat = self.linear(feat)

        return feat, feat_len
