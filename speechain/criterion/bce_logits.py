"""
    Author: Sashi Novitasari
    Affiliation: NAIST
    Date: 2022.08

    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.09
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
    def criterion_init(self, pos_weight: float = 5.0, is_normalized: bool = True):
        """

        Args:
            pos_weight:
            is_normalized:

        """
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.Tensor([pos_weight]))
        self.is_normalized = is_normalized


    def forward(self,
                pred: torch.Tensor,
                tgt: torch.Tensor,
                tgt_len: torch.Tensor):
        """

        Args:
            pred: (batch, text_maxlen, vocab_size)
                The model predictions for the text
            tgt: (batch, text_maxlen)
                The target text labels.
            tgt_len: (batch,)
                The text lengths

        Returns:
            The cross entropy between logits and text

        """
        batch_size, feat_maxlen = pred.size(0), pred.size(1)

        # mask production for the target labels
        tgt_mask = make_mask_from_len(tgt_len).squeeze()
        if tgt.is_cuda:
            tgt_mask = tgt_mask.cuda(tgt.device)

        # BCE loss calculation
        # (batch_size, feat_maxlen)
        loss = self.bce_loss(pred, tgt)
        # (batch_size, feat_maxlen) -> (batch_size * feat_maxlen)
        loss = loss.reshape(-1).masked_fill(~tgt_mask.reshape(-1), 0.0)

        # loss reshaping
        if self.is_normalized:
            # (batch_size * feat_maxlen) -> (1,)
            loss = loss.sum() / tgt_mask.sum()
        else:
            # (batch_size * feat_maxlen) -> (batch_size, feat_maxlen)
            loss = loss.reshape(batch_size, feat_maxlen)
            # (batch_size, feat_maxlen) -> (batch_size,) -> (1,)
            loss = loss.sum(dim=-1).mean()

        return loss
