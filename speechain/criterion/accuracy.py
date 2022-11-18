"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import torch

from speechain.criterion.abs import Criterion
from speechain.utilbox.train_util import make_mask_from_len

class Accuracy(Criterion):
    """
    This criterion calculates the accuracy rate (0.0~1.0) between model predictions and target labels.
    This criterion doesn't have initialization function.

    """

    def forward(self,
                logits: torch.Tensor,
                text: torch.Tensor,
                text_len: torch.Tensor,
                **kwargs):
        """

        Args:
            logits: (batch, logits_maxlen, vocab_size)
                The token predictions from the model
            text: (batch, text_maxlen)
                The ground-truth token sequences
            text_len: (batch,)
                The length for the token predictions.
            **kwargs:

        Returns:
            The accuracy of the token predictions.

        """
        # For the text attached by a <sos/eos> at the beginning
        if logits.size(1) == text.size(1) - 1:
            # text_len must match the sequence dimension of text
            assert text_len.max() == text.size(1), \
                f"There is a mismatch of the sentence length between text and text_len. " \
                f"Expect text_len.max() is either equal to or 1 smaller than text.size(1), " \
                f"but got text_len.max()={text_len.max()} and text.size(1)={text.size(1)}."
            # remove the <sos/eos> at the beginning
            text = text[:, 1:].squeeze(dim=-1)
            # don't use text_len -= 1 here because it will also change the text_len outside this function
            text_len = text_len - 1
        # Otherwise, text must not have a <sos/eos> at the beginning (equal in length with logits)
        elif logits.size(1) != text.size(1):
            raise RuntimeError

        # mask generation for the input text length
        text_mask = make_mask_from_len(text_len).squeeze(dim=1)
        if text.is_cuda:
            text_mask = text_mask.cuda(text.device)

        # calculate the accuracy by the correct prediction
        if logits.dim() == text.dim() + 1:
            logits = logits.argmax(dim=-1)
        elif logits.dim() != text.dim():
            raise RuntimeError(f"logits.shape={logits.shape} but text.shape={text.shape}!")
        correct_num = logits.eq(text).masked_select(text_mask).sum()
        total_num = text_len.sum()
        return correct_num / total_num
