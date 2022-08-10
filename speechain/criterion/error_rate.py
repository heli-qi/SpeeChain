"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import time
from typing import Dict

import editdistance
import edit_distance
import torch

from speechain.tokenizer.abs import Tokenizer
from speechain.criterion.abs import Criterion

class ErrorRate(Criterion):
    """

    """
    def forward(self, hypo_text: torch.Tensor, real_text: torch.Tensor, tokenizer: Tokenizer, do_aver: bool = False):
        """

        Args:
            hypo_text:
            real_text:
            tokenizer:
            do_aver:

        Returns:

        """
        cer_dist, cer_len, wer_dist, wer_len = [], [], [], []
        for i in range(hypo_text.size(0)):
            # remove the padding and sos/eos tokens
            hypo = hypo_text[i][torch.logical_and(hypo_text[i] != tokenizer.ignore_idx,
                                                  hypo_text[i] != tokenizer.sos_eos_idx)]
            real = real_text[i][torch.logical_and(real_text[i] != tokenizer.ignore_idx,
                                                  real_text[i] != tokenizer.sos_eos_idx)]

            # turn text tensors into strings for removing the blanks
            hypo_string = tokenizer.tensor2text(hypo)
            real_string = tokenizer.tensor2text(real)

            # calculate CER
            hypo_chars = hypo_string.replace(" ", "")
            real_chars = real_string.replace(" ", "")
            cer_dist.append(editdistance.eval(hypo_chars, real_chars))
            cer_len.append(len(real_chars))

            # calculate WER
            # Note that split(" ") is not equivalent to split() here
            # because split(" ") will give an extra '' at the end of the list if the string ends with a " "
            # while split() doesn't
            hypo_words = hypo_string.split()
            real_words = real_string.split()
            wer_dist.append(editdistance.eval(hypo_words, real_words))
            wer_len.append(len(real_words))

        cer, wer = [], []
        for i in range(len(cer_dist)):
            cer.append(cer_dist[i] / cer_len[i])
            wer.append(wer_dist[i] / wer_len[i])
        if do_aver:
            cer = sum(cer) / len(cer)
            wer = sum(wer) / len(wer)

        return dict(
            cer=cer,
            wer=wer
        )
