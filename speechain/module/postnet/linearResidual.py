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



class LinearResidualPostnet(Module):
    """
        Stacked fully connected network (linears) with residual connection.
    """

    def module_init(self,
                    feat_dim: int,
                    lnr_dims: int or List[int],
                    lnr_activation: str = None,
                    lnr_dropout: int or List[int] = 0.1,
                    pre_dropout: int = 0,
                    residual: bool = False,
                    is_output_lyr: bool=False
                    ):
        """

        Args:
            feat_dim: int
                The dimension of input feature tensors.
                Used for calculating the in_features of the first Linear layer.
            lnr_dims: int or List[int]
                The values of out_features of each Linear layer.
                The first value in the List represents the out_features of the first Linear layer.
                -1: same size as the last convolutional layer's dim
            lnr_activation: str
                The type of the activation function after all Linear layers.
                None means no activation function is needed.
            lnr_dropout: float or List[float]
                The values of p rate of the Dropout layer after each Linear layer.
            pre_dropout: float
                Apply dropout to the input feature (use this only if the linearResidual network is an intermediate network inside a bigger model)
            residual: bool
                Apply residual connection to connect last layer and input layer
            is_output_lyr: bool
                True if this whole network is placed as the last layer inside a bigger network (True: no activation in the last layer)

        """
        assert isinstance(lnr_dims, (List, int)) and isinstance(lnr_dropout, (List, int)), \
            "The dimensions and dropout rates of linear layers must be given as a list of integers or an integer!"
        assert len(lnr_dims) == len(lnr_dropout), "The length of lnr_dims and lnr_dropout must be equal!"
        assert (not residual) or (residual and (feat_dim==lnr_dims if  isinstance(lnr_dims, (int)) else feat_dim==lnr_dims[-1])), "Residual connection: the dimension of feat_dim and the last lnr_dim is different!"
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
        self.residual = residual
        _prev_dim=feat_dim

        # Linear layers initialization
        _tmp_lnr = []
        for i in range(len(self.lnr_dims)):
            _tmp_lnr.append(torch.nn.Linear(in_features=_prev_dim, out_features=self.lnr_dims[i]))
            if not is_output_lyr or (is_output_lyr and i<(len(self.lnr_dims)-1)):
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

        Returns: feat

        """
        # pass the convolutional layers
        if self.pre_dropout>0:
            feat = F.dropout(feat, self.pre_dropout, training=self.training)
        #feat_pre = feat.clone()
        feat_post = self.linear(feat)

        if self.residual: #merge feature before/after linear
            feat_post += feat 

        return feat_post



