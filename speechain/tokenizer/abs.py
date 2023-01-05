"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
from abc import ABC, abstractmethod

import torch
import numpy as np

from speechain.utilbox.import_util import parse_path_args


class Tokenizer(ABC):
    """
    Tokenizer is the base class of all the _Tokenizer_ objects in this toolkit.
    It on-the-fly transforms text data between strings and tensors.

    For data storage and visualization, the text data should be in the form of strings which is not friendly for model
    forward calculation. For model forward calculation, the text data is better to be in the form of vectors
    (`torch.tensor` or `numpy.ndarray`).

    """

    def __init__(self, token_vocab: str, **tokenizer_conf):
        """
        This function registers some shared member variables for all _Tokenizer_ subclasses:
        1. `self.idx2token`: the mapping Dict from the token index to token string.
        2. `self.token2idx`: the mapping Dict from the token string to token index.
        3. `self.vocab_size`: the number of tokens in the given vocabulary.
        4. `self.sos_eos_idx`: the index of the joint <sos/eos> token used as the beginning and end of a sentence.
        5. `self.ignore_idx`: the index of the blank token used for either CTC blank modeling or ignored token for
            encoder-decoder ASR&TTS models.
        6. `self.unk_idx`: the index of the unknown token.

        Args:
            token_vocab: str
                The path where the token vocabulary is placed.
            **tokenizer_conf:
                The arguments used by tokenizer_init_fn() for your customized Tokenizer initialization.
        """
        self.token_vocab = parse_path_args(token_vocab)
        self.idx2token = dict(enumerate(np.loadtxt(self.token_vocab, dtype=str, delimiter="\n")))
        self.token2idx = dict(map(reversed, self.idx2token.items()))
        self.vocab_size = len(self.token2idx)

        self.sos_eos_idx = self.token2idx['<sos/eos>']
        self.ignore_idx = self.token2idx['<blank>']
        self.unk_idx = self.token2idx['<unk>']

        self.tokenizer_init_fn(**tokenizer_conf)

    def tokenizer_init_fn(self, **tokenizer_conf):
        """
        This hook interface function initializes the customized part of a _Tokenizer_ subclass if had.
        This interface is not mandatory to be overridden.

        Args:
            **tokenizer_conf:
                The arguments used by tokenizer_init_fn() for your customized Tokenizer initialization.
                For more details, please refer to the docstring of your target Tokenizer subclass.

        """
        pass

    def tensor2text(self, tensor: torch.LongTensor) -> str:
        """
        This functions decodes a text tensor into a human-friendly string.

        The default implementation transforms each token index in the input tensor to the token string by `
        self.idx2token`. If the token index is `self.unk_idx`, an asterisk (*) will be used to represent an unknown
        token in the string.

        This interface is not mandatory to be overridden. If your _Tokenizer_ subclass uses some third-party packages
        to decode the input tensor rather than the built-in `self.idx2token`, please override this function.

        Args:
            tensor: torch.LongTensor
                1D integer torch.Tensor that contains the token indices of the sentence to be decoded.

        Returns:
            The string of the decoded sentence.

        """
        # the unknown tokens will be replaced by a star symbol '*'
        return "".join([self.idx2token[idx] if idx != self.unk_idx else '*'
                        for idx in tensor.tolist() if idx != self.sos_eos_idx])

    @abstractmethod
    def text2tensor(self, text: str, no_sos: bool = False, no_eos: bool = False) -> torch.LongTensor:
        """
        This functions encodes a text string into a model-friendly tensor.
        This interface is mandatory to be overridden.
        By default, this function will attach two <sos/eos> at the beginning and end of the returned token id sequence.

        Args:
            text: str
                the input text string to be encoded
            no_sos: bool = False
                Whether to remove the <sos/eos> at the beginning of the token id sequence.
            no_eos: bool = False
                Whether to remove the <sos/eos> at the end of the token id sequence.

        Returns: torch.LongTensor
            The tensor of the encoded sentence

        """
        raise NotImplementedError
