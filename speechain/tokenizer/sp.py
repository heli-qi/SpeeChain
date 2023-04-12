"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import os
import shutil

import torch
import sentencepiece as spm

from speechain.tokenizer.abs import Tokenizer
from speechain.utilbox.import_util import parse_path_args


class SentencePieceTokenizer(Tokenizer):
    """
    Tokenizer implementation that converts the input sentence string into subword tokens, i.e., combinations of
    graphemes, by the sentencepiece package.

    References: https://github.com/google/sentencepiece

    """

    def tokenizer_init_fn(self, token_path: str, copy_path: str = None, **kwargs):
        """
        Initialize the sentencepiece tokenizer model.

        Args:
            copy_path: str = None
                The path where you want to paste the given tokenizer model as a backup.
                If not given, no backup will be saved.
            token_path: str
                The path of your specified sentencepiece tokenizer model file.
                If not given, the model will automatically selected in the same folder as the given token_vocab

        """
        # The model in token_path token_model has the highest priority for token_model initialization
        if token_path is not None:
            token_model = os.path.join(parse_path_args(token_path), 'model')

        # if token_path is not given or model does not exist, use the backup on in copy_path
        if token_path is None or not os.path.exists(token_model):
            assert copy_path is not None, "Please give copy_path for SentencePiece model backup!"
            token_model = os.path.join(parse_path_args(copy_path), 'token_model')

        # initialize the tokenizer model by the sentencepiece package
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(token_model)

        # save the backup if copy_path is given
        if copy_path is not None:
            try:
                shutil.copy(src=token_model, dst=os.path.join(copy_path, 'token_model'))
            except shutil.SameFileError:
                pass

    def tensor2text(self, tensor: torch.LongTensor):
        """

        Args:
            tensor:

        Returns:

        """
        text = self.sp_model.decode_ids(tensor.tolist())
        return text

    def text2tensor(self, text: str, no_sos: bool = False, no_eos: bool = False, return_tensor: bool = True):
        """

        Args:
            text:
            no_sos:
            no_eos:
            return_tensor:

        Returns:

        """
        # initialize the tensor as an empty list
        tokens = []
        # whether to attach sos at the beginning of the tokens
        if not no_sos:
            tokens.append(self.sos_eos_idx)
        # attach the main body of the text
        tokens.extend(self.sp_model.encode_as_ids(text))
        # whether to attach eos at the end of the tokens
        if not no_eos:
            tokens.append(self.sos_eos_idx)
        # turn the token list into a long-type tensor
        if return_tensor:
            return torch.LongTensor(tokens)
        else:
            return tokens
