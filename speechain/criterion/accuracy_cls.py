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
        
        
        # calculate the accuracy by the correct prediction
        correct_num = logits.argmax(dim=-1).eq(text).sum()#masked_select(text_mask).sum()
        total_num = text.size(0)
        return correct_num / total_num
