"""
    Origin: Sashi Novitasari
    Modification: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import torch
import torch.nn as nn

from speechain.module.abs import Module


class PositionwiseFeedForward(Module):
    """
    Position-wise Feed-forward layer
    Projects the output vectors of multi-head attention layer to fdfwd_dim and then back to d_model.
    """

    def module_init(self, d_model: int = 512, fdfwd_dim: int = 2048, dropout=0.1):
        """
        Initializes position-wise feed-forward layer.

        Args:
            d_model: int
                The dimension of the hidden feature vector in each Transformer layer
            fdfwd_dim: int
                The value of the out_features of the first linear feedforward layer and the in_features of the second
                linear feedforward layer
            dropout: float
                The dropout rate for the Dropout layer after the first linear feedforward layer
        """

        self.fdfwd_layers = nn.Sequential(
            nn.Linear(d_model, fdfwd_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fdfwd_dim, d_model)
        )

    def forward(self, x: torch.Tensor):
        """

        Args:
            x:

        Returns:

        """
        return self.fdfwd_layers(x)
