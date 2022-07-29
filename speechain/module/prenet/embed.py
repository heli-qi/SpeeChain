"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import torch
import math

from speechain.module.abs import Module


class EmbedPrenet(Module):
    """
        Create new embeddings for the vocabulary. Use scaling for the Transformer.
    """
    def module_init(self,
                    embedding_dim,
                    vocab_size,
                    scale: bool = False,
                    padding_idx: int = 0):
        """

        Args:
            embedding_dim: int
                The dimension of token embedding vectors
            scale: bool
                Controls whether the values of embedding vectors are scaled according to the embedding dimension.
                Useful for the Transformer model.
            vocab_size: int
                The number of tokens in the dictionary.
            padding_idx: int
                The token index used for padding the tail areas of the short sentences.

        """
        # output_size initialization
        self.output_size = embedding_dim

        # para recording
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.scale = scale
        self.padding_idx = padding_idx

        # initialize Embedding layer
        self.embed = torch.nn.Embedding(num_embeddings=vocab_size,
                                        embedding_dim=embedding_dim,
                                        padding_idx=padding_idx)

    def forward(self, text: torch.Tensor):
        """
        Perform lookup for input `x` in the embedding table.

        Args:
            text: (batch, seq_len)
            index in the vocabulary

        Returns:
            embedded representation for `x`
        """
        if self.scale:
            return self.embed(text) * math.sqrt(self.embedding_dim)
        else:
            return self.embed(text)