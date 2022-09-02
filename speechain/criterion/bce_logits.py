"""
    Author: Sashi Novitasari
    Affiliation: NAIST
    Date: 2022.08
"""
import torch
import numpy as np
from typing import Dict

from speechain.criterion.abs import Criterion
from speechain.utilbox.train_util import make_mask_from_len

class BCELogits(Criterion):
    """
    This criterion calculates the cross entropy between model predictions and target labels.
    In this implementation, we realize the following functions:
        1. Sentence normalization. The loss will be normalized according to the length of each sentence.
        2. Label smoothing. The target label will be transformed from a sharp one-hot vector to a smooth distribution vector.
        3. Token reweighting. The weight of each token in the cross entropy calculation can be customized manually.
        If you want to customize the weights, you need to give the token dictionary.

    """
    def criterion_init(self,coeff=1.0,coeff_bern_positive=1.0,device=None):
        self.coeff = coeff
        self.coeff_bern_positive = torch.Tensor([coeff_bern_positive])
        self.device= device
        pass

    def forward(self,
                pred: torch.Tensor,
                tgt: torch.Tensor):
        """

        Args:
            logits: (batch, text_maxlen, vocab_size)
                The model predictions for the text
            text: (batch, text_maxlen)
                The target text labels.
            text_len: (batch,)
                The text lengths

        Returns:
            The cross entropy between logits and text

        """
        if pred.is_cuda and self.device is None:
            self.device = pred.device
            self.coeff_bern_positive=self.coeff_bern_positive.cuda(self.device)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred,tgt,pos_weight=self.coeff_bern_positive) * self.coeff
        return loss

