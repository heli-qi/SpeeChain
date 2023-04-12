"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
from speechain.tokenizer.abs import Tokenizer

import torch


class CharTokenizer(Tokenizer):
    """
    Tokenizer implementation that converts the input sentence string into a list of graphemes (characters).

    """

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
        tokens.extend([self.token2idx[char] if char in self.token2idx.keys() else self.unk_idx for char in text])
        # whether to attach eos at the end of the tokens
        if not no_eos:
            tokens.append(self.sos_eos_idx)
        # turn the token list into a long-type tensor
        if return_tensor:
            return torch.LongTensor(tokens)
        else:
            return tokens
