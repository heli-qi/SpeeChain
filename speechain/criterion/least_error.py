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

class LeastError(Criterion):
    """
    This criterion calculates the cross entropy between model predictions and target labels.
    In this implementation, we realize the following functions:
        1. Sentence normalization. The loss will be normalized according to the length of each sentence.
        2. Label smoothing. The target label will be transformed from a sharp one-hot vector to a smooth distribution vector.
        3. Token reweighting. The weight of each token in the cross entropy calculation can be customized manually.
        If you want to customize the weights, you need to give the token dictionary.

    """
    def criterion_init(self,loss_type="L2",coeff=1,topn=None):
        self.loss_type=loss_type
        self.coeff=coeff
        self.topn=topn
        pass

    def forward(self,
                pred: torch.Tensor,
                tgt: torch.Tensor,
                tgt_len: torch.Tensor,
                size_average=True):
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
        if self.topn is not None:
            ndim = int(pred.size(-1) * self.topn)
            pred= pred[:, :, 0:ndim]
            tgt= tgt[:, :, 0:ndim]
            
        tgt_mask = make_mask_from_len(tgt_len).squeeze()
        if tgt.is_cuda:
            tgt_mask = tgt_mask.cuda(tgt.device)

        loss = 0
        if 'L1' in self.loss_type :
            loss += torch.abs(pred - tgt)
        if 'L2' in self.loss_type :
            loss += (pred - tgt)**2 # batch x len x ndim #
        

        loss = torch.mean(loss, 2) # batch x len #
        loss = loss * tgt_mask
        loss = torch.sum(loss) # sum all rest dim #

        if size_average :
            loss /= pred.size(0)

        return loss * self.coeff

