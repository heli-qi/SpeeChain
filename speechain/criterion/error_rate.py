"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
from typing import List

import editdistance
import torch

from speechain.tokenizer.abs import Tokenizer
from speechain.criterion.abs import Criterion


def text_preprocess(text, tokenizer: Tokenizer):
    # tensor input need to be recovered to string
    if isinstance(text, torch.Tensor):
        # remove the padding and sos/eos tokens
        proc_text = text[torch.logical_and(text != tokenizer.ignore_idx, text != tokenizer.sos_eos_idx)]
        # turn text tensors into strings for removing the blanks
        string = tokenizer.tensor2text(proc_text)
    # string input, no processing is done here
    elif isinstance(text, str):
        string = text
    else:
        raise RuntimeError

    return string

class ErrorRate(Criterion):
    """

    """
    def forward(self, hypo_text: torch.Tensor or List[str] or str, real_text: torch.Tensor or List[str] or str,
                tokenizer: Tokenizer, do_aver: bool = False):
        """

        Args:
            hypo_text:
            real_text:
            tokenizer:
            do_aver:

        Returns:

        """
        # make sure that hypo_text is a 2-dim tensor or a list of strings
        if isinstance(hypo_text, torch.Tensor) and hypo_text.dim() == 1:
            hypo_text = hypo_text.unsqueeze(0)
        elif isinstance(hypo_text, str):
            hypo_text = [hypo_text]
        # make sure that real_text is a 2-dim tensor or a list of strings
        if isinstance(real_text, torch.Tensor) and real_text.dim() == 1:
            real_text = real_text.unsqueeze(0)
        elif isinstance(real_text, str):
            real_text = [real_text]

        cer_dist, cer_len, wer_dist, wer_len = [], [], [], []
        for i in range(len(hypo_text)):
            # obtain the strings
            hypo_string = text_preprocess(hypo_text[i], tokenizer)
            real_string = text_preprocess(real_text[i], tokenizer)

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
