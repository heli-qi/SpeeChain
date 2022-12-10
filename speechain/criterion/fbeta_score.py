"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import torch

from speechain.criterion.abs import Criterion
from speechain.utilbox.train_util import make_mask_from_len


class FBetaScore(Criterion):
    """

    """

    def criterion_init(self, beta: int = 1):
        """

        Args:
            beta: float

        """
        self.beta = beta

    def __call__(self, pred: torch.Tensor, tgt: torch.Tensor, tgt_len: torch.Tensor):
        """

        Args:
            pred:
            tgt:
            tgt_len:

        Returns:

        """
        # mask generation for the input text length
        tgt_mask = make_mask_from_len(tgt_len).squeeze()
        if tgt.is_cuda:
            tgt_mask = tgt_mask.cuda(tgt.device)

        # mask the pred and tgt, calculate the necessary components for precision and recall
        pred, tgt = pred.masked_select(tgt_mask), tgt.masked_select(tgt_mask)
        pred_pos, pred_neg, tgt_pos, tgt_neg = pred == 1, pred == 0, tgt == 1, tgt == 0
        true_pos, true_neg, false_pos, false_neg = \
            torch.logical_and(pred_pos, tgt_pos).sum(), \
            torch.logical_and(pred_neg, tgt_neg).sum(), \
            torch.logical_and(pred_pos, tgt_neg).sum(), \
            torch.logical_and(pred_neg, tgt_pos).sum()

        # calculate f_beta by precision and recall
        precision = true_pos / (true_pos + false_pos + 1e-10)
        recall = true_pos / (true_pos + false_neg + 1e-10)
        return (1 + self.beta ** 2) * (precision * recall) / (self.beta ** 2 * precision + recall + 1e-10)
