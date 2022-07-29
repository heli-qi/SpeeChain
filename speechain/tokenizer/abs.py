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
    def __init__(self, token_dict: str, **kwargs):
        """

        Args:
            token_dict:
            **kwargs:
        """
        token_dict = np.loadtxt(token_dict, dtype=str, delimiter="\n")
        self.token2idx = dict(map(reversed, enumerate(token_dict)))
        self.idx2token = dict(enumerate(token_dict))
        self.vocab_size = len(self.idx2token)

        self.sos_eos_idx = self.token2idx['<sos/eos>']
        self.ignore_idx = self.token2idx['<blank>']
        self.unk_idx = self.token2idx['<unk>']

        self.tokenizer_init(**kwargs)


    def tokenizer_init(self, **kwargs):
        pass


    def tensor2text(self, tensor: torch.Tensor):
        return "".join([self.idx2token[idx.item()] for idx in tensor])


    @abstractmethod
    def text2tensor(self, text: str):
        raise NotImplementedError