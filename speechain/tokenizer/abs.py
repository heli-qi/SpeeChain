"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
from abc import ABC, abstractmethod

import os
import torch
import numpy as np

from speechain.utilbox.import_util import parse_path_args
from speechain.utilbox.data_loading_util import load_idx2data_file


class Tokenizer(ABC):
    """
    Tokenizer is the base class of all the _Tokenizer_ objects in this toolkit.
    It on-the-fly transforms text data between strings and tensors.

    For data storage and visualization, the text data should be in the form of strings which is not friendly for model
    forward calculation. For model forward calculation, the text data is better to be in the form of vectors
    (`torch.tensor` or `numpy.ndarray`).

    """

    def __init__(self, token_path: str = None, copy_path: str = None, **tokenizer_conf):
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
            token_path: str
                The path where the token vocabulary is placed.
            copy_path: str = None
                The path where you want to paste the given token vocabulary as a backup.
                If not given, no backup will be saved.
            **tokenizer_conf:
                The arguments used by tokenizer_init_fn() for your customized Tokenizer initialization.
        """
        # The vocab in token_path has the higher priority than the backup one in copy_path for vocabulary initialization
        if token_path is not None:
            token_vocab = os.path.join(parse_path_args(token_path), 'vocab')

        # if token_path is not given or vocab does not exist, use the backup one in copy_path
        if token_path is None or not os.path.exists(token_vocab):
            assert copy_path is not None, "Please give copy_path for vocabulary backup!"
            token_vocab = os.path.join(parse_path_args(copy_path), 'token_vocab')

        # register token-related variables
        self.idx2token = load_idx2data_file(token_vocab, do_separate=False)
        self.token2idx = dict(map(reversed, self.idx2token.items()))
        self.vocab_size = len(self.token2idx)
        self.sos_eos_idx = self.token2idx['<sos/eos>']
        self.ignore_idx = self.token2idx['<blank>']
        self.unk_idx = self.token2idx['<unk>']
        if '<space>' in self.token2idx.keys():
            self.space_idx = self.token2idx['<space>']
        else:
            self.space_idx = None

        # save the backup if copy_path is given
        if copy_path is not None:
            np.savetxt(os.path.join(copy_path, 'token_vocab'), list(self.token2idx.keys()), fmt="%s")

        # call the hook function for customized initialization
        self.tokenizer_init_fn(token_path=token_path, copy_path=copy_path, **tokenizer_conf)

    def tokenizer_init_fn(self, token_path: str, copy_path: str = None, **tokenizer_conf):
        """
        This hook interface function initializes the customized part of a _Tokenizer_ subclass if had.
        This interface is not mandatory to be overridden.

        Args:
            copy_path: str = None
                The path where you want to paste the given tokenizer model as a backup.
                If not given, no backup will be saved.
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
        token_list = []
        for idx in tensor.tolist():
            if idx in [self.sos_eos_idx, self.ignore_idx]:
                continue
            # the space tokens will be replaced by a blank
            elif self.space_idx is not None and idx == self.space_idx:
                token_list.append(' ')
            # the unknown tokens will be replaced by a star symbol '*'
            elif idx == self.unk_idx:
                token_list.append('*')
            else:
                token_list.append(self.idx2token[idx])
        return "".join(token_list)

    @abstractmethod
    def text2tensor(self, text: str, no_sos: bool = False, no_eos: bool = False, return_tensor: bool = True) -> torch.LongTensor or List:
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
            return_tensor: bool = True
                Whether to return the tokenization results as a tensor. If False, a List will be returned.

        Returns: torch.LongTensor
            The tensor of the encoded sentence

        """
        raise NotImplementedError
