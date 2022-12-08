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

from speechain.module.abs import Module


class LinearPrenet(Module):
    """
        The Linear prenet. Usually used before the Transformer TTS decoder.
        This prenet is made up of one or more Linear blocks which is composed of the components below:
            1. (mandatory) a Linear layer
            2. (optional) an activation function
            3. (optional) a Dropout layer

        Reference:
            Neural Speech Synthesis with Transformer Network
            https://ojs.aaai.org/index.php/AAAI/article/view/4642/4520
    """

    def module_init(self,
                    feat_dim: int = None,
                    lnr_dims: int or List[int] = [256, 256],
                    lnr_activation: str = 'ReLU',
                    lnr_dropout: float or List[float] = None,
                    zero_centered: bool = False):
        """

        Args:
            feat_dim: int
                The dimension of input acoustic feature tensors.
                Used for calculating the in_features of the first Linear layer.
            lnr_dims: int or List[int]
                The values of out_features of each Linear layer.
                The first value in the List represents the out_features of the first Linear layer.
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
        # arguments checking
        if lnr_dropout is not None:
            assert isinstance(lnr_dropout, (List, float)), \
                "The dropout rates of linear layers must be given as a list of integers or an integer!"
        assert isinstance(lnr_dims, (List, int)), \
            "The dimensions of linear layers must be given as a list of integers or an integer!"

        # input_size initialization
        if self.input_size is not None:
            feat_dim = self.input_size
        else:
            assert feat_dim is not None

        # para recording
        if lnr_dims is not None:
            self.lnr_dims = lnr_dims if isinstance(lnr_dims, List) else [lnr_dims]
        self.lnr_dropout = lnr_dropout

        # --- 1. Linear Part Initialization --- #
        # Linear layers construction
        _tmp_lnr = []
        _prev_dim = feat_dim
        # The order of activation function and dropout layer is somewhat not a big deal
        # a useful blog: https://sebastianraschka.com/faq/docs/dropout-activation.html
        for i in range(len(self.lnr_dims)):
            _tmp_lnr.append(torch.nn.Linear(in_features=_prev_dim, out_features=self.lnr_dims[i]))
            if lnr_activation is not None:
                # no 'ReLU'-series activation is added for the last layer if zero_centered is specified
                if i != len(self.lnr_dims) - 1 or not (zero_centered and 'ReLU' in lnr_activation):
                    _tmp_lnr.append(getattr(torch.nn, lnr_activation)())
            if lnr_dropout is not None:
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
                The input feature tensors.
            feat_len: (batch,)
                The length of each feature tensor.
                feat_len is not used in this forward function, but it's better to include this argument here for
                compatibility with other prenet classes.

        Returns: feat, feat_len
            The embedded feature vectors with their lengths.

        """
        feat = self.linear(feat)
        return feat, feat_len
