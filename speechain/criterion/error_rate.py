"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import time
from typing import Dict

import editdistance
import torch

from speechain.tokenizer.abs import Tokenizer
from speechain.criterion.abs import Criterion

class ErrorRate(Criterion):
    """

    """
    def forward(self, hypo_text: torch.Tensor, real_text: torch.Tensor, tokenizer: Tokenizer):
        """
        There are two libraries that provides the functions of calculating the edit distance:
        1. editdistance.eval(hypo, real)
        2. edit_distance.SequenceMatcher(a=hypo, b=real).distance()

        2 is way slower than 1 (about 1000 times, lol~~~~~), so 1 is used here.

        Args:
            hypo_text:
            real_text:
            tokenizer:

        Returns:

        """
        cer_dist, cer_len, wer_dist, wer_len = [], [], [], []

        for i in range(hypo_text.size(0)):
            # remove the padding tokens
            hypo = hypo_text[i][hypo_text[i] != tokenizer.ignore_idx]
            real = real_text[i][real_text[i] != tokenizer.ignore_idx]

            # turn text tensors into strings for removing the blanks
            hypo_string = tokenizer.tensor2text(hypo)
            real_string = tokenizer.tensor2text(real)

            # calculate CER
            hypo_chars = hypo_string.replace(" ", "")
            real_chars = real_string.replace(" ", "")
            cer_dist.append(editdistance.eval(hypo_chars, real_chars))
            cer_len.append(len(real_chars))

            # calculate WER
            hypo_words = hypo_string.split(" ")
            real_words = real_string.split(" ")
            wer_dist.append(editdistance.eval(hypo_words, real_words))
            wer_len.append(len(real_words))

        return dict(
            cer=sum(cer_dist) / sum(cer_len),
            wer=sum(wer_dist) / sum(wer_len)
        )