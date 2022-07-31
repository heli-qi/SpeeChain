"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
from speechain.tokenizer.abs import Tokenizer

import torch

class CharTokenizer(Tokenizer):
    """

    """
    def text2tensor(self, text: str):
        """

        Args:
            text:

        Returns:

        """
        tokens = [self.sos_eos_idx]

        # loop each character in the text data
        for char in text:
            if char not in self.token2idx.keys():
                tokens.append(self.unk_idx)
            else:
                tokens.append(self.token2idx[char])

        tokens.append(self.sos_eos_idx)

        return torch.LongTensor(tokens)
