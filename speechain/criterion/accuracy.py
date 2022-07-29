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
        if text_len.max() == text.size(1):
            text_len -= 1
        elif text_len.max() != text.size(1) - 1:
            raise ValueError(f"There is a mismatch of the sentence length between text and text_len. "
                             f"Expect text_len.max() is either equal to or 1 smaller than text.size(1), "
                             f"but got text_len.max()={text_len.max()} and text.size(1)={text.size(1)}.")

        # mask generation for the input text length
        text_mask = make_mask_from_len(text_len).squeeze()
        if text.is_cuda:
            text_mask = text_mask.cuda(text.device)

        # remove the <sos/eos> at the beginning of each sentence if necessary
        text = text[:, 1:].squeeze()

        # calculate the accuracy by the correct prediction
        correct_num = logits.argmax(dim=-1).eq(text).masked_select(text_mask).sum()
        total_num = text_mask.sum()
        return correct_num / total_num
