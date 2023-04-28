"""
    Origin: Sashi Novitasari
    Modification: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import torch
import torch.nn as nn

from speechain.module.abs import Module
from speechain.module.prenet.conv1d import Conv1dEv

class PositionwiseFeedForward(Module):
    """
    Position-wise Feed-forward layer
    Projects the output vectors of multi-head attention layer to fdfwd_dim and then back to d_model.
    """

    def module_init(self, d_model: int = 512, fdfwd_dim: int = 2048, fdfwd_type: str = 'linear',
                    fdfwd_activation: str = 'ReLU', fdfwd_kernel: int = 9, dropout=0.1):
        """
        Initializes position-wise feed-forward layer.

        Args:
            d_model: int
                The dimension of the hidden feature vector in each Transformer layer
            fdfwd_dim: int
                The value of the out_features of the first linear feedforward layer and the in_features of the second
                linear feedforward layer
            fdfwd_type: str
                The type of the feed-forward layer. 'linear' means the Linear layer while 'conv' means the Conv1d layer.
            fdfwd_activation: str
                The name of the activation function of feedforward layers. Should be the name of functions in 'torch.nn'.
            fdfwd_kernel: int
                The kernal size of the Conv1d feed-forward layer. This argument is not effective if fdfwd_type == 'linear'.
            dropout: float
                The dropout rate for the Dropout layer after the first linear feedforward layer
        """
        # In-layer at the beginning
        if fdfwd_type == 'linear':
            self.in_layer = nn.Linear(d_model, fdfwd_dim)
        elif fdfwd_type == 'conv':
            self.in_layer = Conv1dEv(d_model, fdfwd_dim, fdfwd_kernel)
        else:
            raise NotImplementedError(f"Currently, fdfwd_type can only be one of 'linear' and 'conv'. "
                                      f"But got {fdfwd_type}!")

        # ReLU and DropOut layers in the middle
        self.activation = getattr(torch.nn, fdfwd_activation)()
        self.dropout = nn.Dropout(dropout)

        # Out-layer at the end
        if fdfwd_type == 'linear':
            self.out_layer = nn.Linear(fdfwd_dim, d_model)
        elif fdfwd_type == 'conv':
            self.out_layer = Conv1dEv(fdfwd_dim, d_model, fdfwd_kernel)
        else:
            raise NotImplementedError(f"Currently, fdfwd_type can only be one of 'linear' and 'conv'. "
                                      f"But got {fdfwd_type}!")

    def forward(self, x: torch.Tensor):
        """

        Args:
            x: (batch, seq_maxlen, d_model)

        Returns:

        """
        # forward the convolutional layers
        if isinstance(self.in_layer, Conv1dEv):
            # (batch, seq_maxlen, d_model) -> (batch, d_model, seq_maxlen)
            x = x.transpose(1, 2)
        # pass the in-layer at the beginning
        # (batch, d_model, seq_maxlen) -> (batch, fdfwd_dim, seq_maxlen) or
        # (batch, seq_maxlen, d_model) -> (batch, seq_maxlen, fdfwd_dim)
        x = self.in_layer(x)

        # pass the middle layers
        x = self.dropout(self.activation(x))

        # pass the out-layer at the end
        # (batch, fdfwd_dim, seq_maxlen) -> (batch, d_model, seq_maxlen) or
        # (batch, seq_maxlen, fdfwd_dim) -> (batch, seq_maxlen, d_model)
        x = self.out_layer(x)
        # forward the convolutional layers
        if isinstance(self.out_layer, Conv1dEv):
            # (batch, d_model, seq_maxlen) -> (batch, seq_maxlen, d_model)
            x = x.transpose(1, 2)
        return x
