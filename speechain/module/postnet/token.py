"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import torch

from speechain.module.abs import Module


class TokenPostnet(Module):
    """
        The decoder postnet that projects the model output vectors into token predictions.

    """

    def module_init(self, vocab_size: int, input_dim: int = None):
        """

        Args:
            input_dim: int
                The dimension of the output vectors from the decoder
            vocab_size: int
                The number of tokens in the dictionary.

        """
        # input_size and output_size initialization
        if self.input_size is not None:
            input_dim = self.input_size
        else:
            assert input_dim is not None
        self.output_size = vocab_size

        # para recording
        self.input_dim = input_dim
        self.vocab_size = vocab_size

        # initialize the output layer
        self.linear = torch.nn.Linear(in_features=input_dim, out_features=vocab_size)

    def forward(self, input: torch.Tensor):
        """

        Args:
            input:

        Returns:

        """
        return self.linear(input)
