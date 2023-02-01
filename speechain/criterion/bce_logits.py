"""
    Author: Sashi Novitasari
    Affiliation: NAIST
    Date: 2022.08

    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.09
"""
import torch

from speechain.criterion.abs import Criterion
from speechain.utilbox.train_util import make_mask_from_len


class BCELogits(Criterion):
    """

    """

    def criterion_init(self, pos_weight: float = 5.0, is_normalized: bool = True):
        """

        Args:
            pos_weight: float = 5.0
                The weight putted on stop points for stop loss calculation.
            is_normalized: bool = True
                Controls whether the sentence normalization is performed for stop loss calculation.

        """
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.Tensor([pos_weight]))
        self.is_normalized = is_normalized

    def __call__(self,
                 pred: torch.Tensor,
                 tgt: torch.Tensor,
                 tgt_len: torch.Tensor):
        """

        Args:
            pred: (batch, text_maxlen)
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
        if tgt_len.is_cuda:
            tgt_mask = tgt_mask.cuda(tgt_len.device)

        # BCE loss calculation
        # make sure that the pos_weight is also on GPU
        if pred.is_cuda and not self.bce_loss.pos_weight.is_cuda:
            self.bce_loss.pos_weight = self.bce_loss.pos_weight.cuda(pred.device)
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
