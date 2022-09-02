"""
    Author: Sashi Novitasari
    Affiliation: NAIST (-2022)
    Date: 2022.08
"""
from typing import List
import torch
import math

from speechain.module.abs import Module
from speechain.utilbox.train_util import generator_act_module



class LinearPrenet(Module):
    """
        The Conv2d prenet for ASR. Usually used before the Transformer ASR encoder.
        There are two parts in this prenet:
            1. Conv2d layers. Each Conv2d layer is followed by an activation function and a BatchNorm2d layer(optional).
            We don't include a Dropout layer after each Conv2d layer.
            2. Linear layers. Each Linear layer is followed by an activation function and a Dropout layer.

        Reference:
            Speech-transformer: a no-recurrence sequence-to-sequence model for speech recognition
            https://ieeexplore.ieee.org/abstract/document/8462506/
    """

    def module_init(self,
                    feat_dim: int,
                    lnr_dims: int or List[int],
                    lnr_activation: str = None,
                    lnr_dropout: int or List[int] = 0.0,
                    pre_dropout: int = 0):
        """

        Args:
            feat_dim: int
                The dimension of input acoustic feature tensors.
                Used for calculating the in_features of the first Linear layer.
            conv_dims: List[int] or int
                The values of out_channels of each Conv2d layer.
                If a list of integers is given, multiple Conv2d layers will be initialized.
                If an integer is given, there will be only one Conv2d layer
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
                -1: same size as the last convolutional layer's dim
            lnr_activation: str
                The type of the activation function after all Linear layers.
                None means no activation function is needed.
            lnr_dropout: float or List[float]
                The values of p rate of the Dropout layer after each Linear layer.

        """
        assert isinstance(lnr_dims, (List, int)) and isinstance(lnr_dropout, (List, int)), \
            "The dimensions and dropout rates of linear layers must be given as a list of integers or an integer!"
        assert len(lnr_dims) == len(lnr_dropout), "The length of lnr_dims and lnr_dropout must be equal!"
        
        # input_size initialization
        if self.input_size is not None:
            feat_dim = self.input_size
        else:
            assert feat_dim is not None

        # para recording
        self.lnr_dims = lnr_dims if isinstance(lnr_dims, List) else [lnr_dims]
        self.lnr_dims = [feat_dim if self.lnr_dims[i]==-1 else self.lnr_dims[i] for i in range(len(self.lnr_dims))]        
        self.lnr_dropout = lnr_dropout if isinstance(lnr_dropout, List) else [lnr_dropout]

        self.pre_dropout = pre_dropout
        self.output_size = self.lnr_dims[-1]

        _prev_dim=feat_dim

        # Linear layers initialization
        _tmp_lnr = []
        for i in range(len(self.lnr_dims)):
            _tmp_lnr.append(torch.nn.Linear(in_features=_prev_dim, out_features=self.lnr_dims[i]))
            if lnr_activation is not None:
                _tmp_lnr.append(generator_act_module(lnr_activation))
            _tmp_lnr.append(torch.nn.Dropout(p=lnr_dropout[i]))
            _prev_dim=self.lnr_dims[i]
            
        self.linear = torch.nn.Sequential(*_tmp_lnr)
        
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
        # pass the convolutional layers
        if self.pre_dropout>0:
            feat = F.dropout(feat, self.pre_dropout, training=self.training)

        feat = self.linear(feat)
        
        return feat, feat_len



