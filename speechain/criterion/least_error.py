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

class LeastError(Criterion):
    """

    """
    def criterion_init(self,
                       loss_type: str = "L2",
                       is_normalized: bool = True,
                       update_range: int or float = None):
        """

        Args:
            loss_type:
            is_normalized:
            update_range:

        """
        assert 'L1' in loss_type or 'L2' in loss_type, \
            "You input loss_type must contain 'L1' and 'L2' or both of them like 'L1+L2', 'L1&L2', and so on."
        if isinstance(update_range, int):
            assert update_range < 0, \
                f"For setting absolute updating range, you must input a negative integer, but got {update_range}."
        elif isinstance(update_range, float):
            assert 0 < update_range < 1, \
                f"For setting relative updating range, you must input a postive float number between 0 and 1, " \
                f"but got {update_range}."
        else:
            assert update_range is None

        self.loss_type = loss_type
        self.l1_loss = torch.nn.L1Loss(reduction='none')
        self.l2_loss = torch.nn.MSELoss(reduction='none')

        self.is_normalized = is_normalized
        self.update_range = update_range


    def forward(self,
                pred: torch.Tensor,
                tgt: torch.Tensor,
                tgt_len: torch.Tensor):
        """

        Args:
            pred: (batch, feat_maxlen, feat_dim * reduction_factor)
                The model predictions for the acoustic feature
            tgt: (batch, feat_maxlen, feat_dim * reduction_factor)
                The target acoustic feature labels.
            tgt_len: (batch,)
                The target label lengths

        Returns:
            The cross entropy between logits and text

        """
        batch_size, feat_maxlen = pred.size(0), pred.size(1)
        # updating range restriction, ndim is the dimension selected to be updated
        if self.update_range is not None:
            ndim = feat_maxlen * self.update_range if self.update_range > 0 else -self.update_range
            pred = pred[:, :, :ndim]
            tgt = tgt[:, :, :ndim]

        # mask production for the target labels
        tgt_mask = make_mask_from_len(tgt_len).squeeze()
        if tgt.is_cuda:
            tgt_mask = tgt_mask.cuda(tgt.device)

        # MSE & MAE loss calculation
        # (batch_size, feat_maxlen, ndim)
        loss = torch.zeros_like(pred)
        if 'L1' in self.loss_type:
            loss += self.l1_loss(pred, tgt)
        if 'L2' in self.loss_type:
            loss += self.l2_loss(pred, tgt)

        # loss reshaping
        # (batch_size, feat_maxlen, ndim) -> (batch_size, feat_maxlen)
        loss = loss.mean(dim=-1)
        # (batch_size, feat_maxlen) -> (batch_size * feat_maxlen)
        loss = loss.reshape(-1).masked_fill(~tgt_mask.reshape(-1), 0.0)
        if self.is_normalized:
            # (batch_size * feat_maxlen) -> (1,)
            loss = loss.sum() / tgt_mask.sum()
        else:
            # (batch_size * feat_maxlen) -> (batch_size, feat_maxlen)
            loss = loss.reshape(batch_size, feat_maxlen)
            # (batch_size, feat_maxlen) -> (batch_size,) -> (1,)
            loss = loss.sum(dim=-1).mean()

        return loss


    def extra_repr(self) -> str:
        return f"loss_type={self.loss_type}, is_normalized={self.is_normalized}"
