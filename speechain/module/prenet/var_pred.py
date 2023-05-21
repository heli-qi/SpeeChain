import torch

from typing import List

from speechain.module.abs import Module
from speechain.module.prenet.conv1d import Conv1dEv


class LayerNorm(torch.nn.LayerNorm):
    """
        Layer normalization module.
        Borrowed from
            https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/layer_norm.py

        Args:
            nout (int): Output dim size.
            dim (int): Dimension to be normalized.
    """

    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Normalized tensor.
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return (
            super(LayerNorm, self)
            .forward(x.transpose(self.dim, -1))
            .transpose(self.dim, -1)
        )


class Conv1dVarPredictor(Module):
    """
        The Conv1d variance predictor for FastSpeech2.
        This module is made up of:
            1. (mandatory) The Conv1d part contains two or more Conv1d blocks which are composed of the components below
                1. (mandatory) a Conv1d layer
                2. (mandatory) a ReLU function
                3. (mandatory) a LayerNorm layer
                4. (mandatory) a Dropout layer.
            2. (mandatory) The Linear part contains one Linear block which is composed of the component below
                1. (mandatory) a Linear layer

        Reference:
            Fastspeech 2: Fast and high-quality end-to-end text to speech
            https://arxiv.org/pdf/2006.04558
    """
    def module_init(self,
                    feat_dim: int = None,
                    conv_dims: int or List[int] = [256, 256],
                    conv_kernel: int = 3,
                    conv_stride: int = 1,
                    conv_dropout: float or List[float] = 0.5,
                    use_conv_emb: bool = True,
                    conv_emb_kernel: int = 1,
                    conv_emb_dropout: float = 0.0):
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
            conv_dropout: float or List[float]
                The values of p rate of the Dropout layer after each Linear layer.
            use_conv_emb: bool
                Whether to embed the predicted scalar back to an embedding vector.
                This argument needs to be False for duration predictor.
            conv_emb_kernel: int
                The value of kernel_size for the conv1d embedding layer.
                Only effective when use_conv_emb is True.
            conv_emb_dropout: float
                The value of p reate of the Dropout layer after the conv1d embedding layer.
                Only effective when use_conv_emb is True.
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

        # input_size initialization
        if self.input_size is not None:
            feat_dim = self.input_size
        else:
            assert feat_dim is not None
            self.input_size = feat_dim

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
            # 0 means go back to the input feat_dim
            if self.conv_dims[i] == 0:
                self.conv_dims[i] = feat_dim
            # -1 means equal to the previous layer
            elif self.conv_dims[i] == -1:
                self.conv_dims[i] = self.conv_dims[i - 1]
            # Conv1d layer
            _tmp_conv.append(
                Conv1dEv(in_channels=_prev_dim,
                         out_channels=self.conv_dims[i],
                         kernel_size=self.conv_kernel,
                         stride=self.conv_stride,
                         padding_mode='same')
            )
            # ReLU function
            _tmp_conv.append(torch.nn.ReLU())
            # LayerNorm layer
            _tmp_conv.append(LayerNorm(self.conv_dims[i], dim=1))
            # Dropout layer
            if conv_dropout is not None:
                _tmp_conv.append(torch.nn.Dropout(
                    p=self.conv_dropout if not isinstance(self.conv_dropout, List) else self.conv_dropout[i]
                ))
            _prev_dim = conv_dims[i]
        self.conv = torch.nn.Sequential(*_tmp_conv)

        # --- 2. Linear Part Initialization --- #
        self.linear = torch.nn.Linear(_prev_dim, 1)

        # --- 3. Scalar Embedding Part Initialization --- #
        if use_conv_emb:
            _tmp_conv_emb = [
                Conv1dEv(in_channels=1,
                         out_channels=self.input_size,
                         kernel_size=conv_emb_kernel,
                         padding_mode='same')
            ]
            if conv_emb_dropout > 0:
                _tmp_conv_emb.append(torch.nn.Dropout(p=conv_emb_dropout))
            self.conv_emb = torch.nn.Sequential(*_tmp_conv_emb)
        self.output_size = self.input_size

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

        # forward the linear layer
        # (batch, feat_maxlen, conv_dim) -> (batch, feat_maxlen, 1) -> (batch, feat_maxlen)
        feat = self.linear(feat).squeeze(-1)

        # return feat_len for the compatibility with other prenets
        return feat, feat_len

    def emb_pred_scalar(self, pred_scalar: torch.Tensor):
        """

        Args:
            pred_scalar: (batch, feat_maxlen, 1) or (batch, feat_maxlen)
                The predicted scalar vectors calculated in the forward().

        """
        assert hasattr(self, 'conv_emb'), \
            "Please set the argument 'use_conv_emb' to True if you want to embed the predicted scalar!"
        if len(pred_scalar.shape) == 2:
            pred_scalar = pred_scalar.unsqueeze(-1)
        else:
            assert len(pred_scalar.shape) == 3 and pred_scalar.size(-1) == 1

        return self.conv_emb(pred_scalar.transpose(1, 2)).transpose(1, 2)
