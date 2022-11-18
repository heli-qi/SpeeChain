"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
from abc import ABC, abstractmethod

import torch
import numpy as np


class Tokenizer(ABC, object):
    """

    """

    def __init__(self, token_vocab: str, **kwargs):
        """

        Args:
            token_vocab:
            **kwargs:
        """
        token_vocab = np.loadtxt(token_vocab, dtype=str, delimiter="\n").tolist()
        self.idx2token = dict(enumerate(token_vocab))
        self.token2idx = dict(map(reversed, self.idx2token.items()))
        self.vocab_size = len(self.token2idx)

        self.sos_eos_idx = self.token2idx['<sos/eos>']
        self.ignore_idx = self.token2idx['<blank>']
        self.unk_idx = self.token2idx['<unk>']

        self.tokenizer_init(**kwargs)

    def tokenizer_init(self, **kwargs):
        pass

    def tensor2text(self, tensor: torch.Tensor) -> str:
        """
        Default sentence tensor decoding function by the built-in index-to-token Dict self.idx2token.
        Can be overridden if you would like to use some external tokenizer models to decode the input tensor.

        Args:
            tensor:
                1D integer torch.Tensor that contains the token indices of the sentence to be decoded.

        Returns:
            The string of the decoded sentence.

        """
        # the unknown tokens will be replaced by a star symbol '*'
        return "".join([self.idx2token[idx] if idx != self.unk_idx else '*'
                        for idx in tensor.tolist() if idx != self.sos_eos_idx])

    @abstractmethod
    def text2tensor(self, text: str) -> torch.LongTensor:
        raise NotImplementedError
