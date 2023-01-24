"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import math
import torch
import numpy as np
from typing import Dict

from speechain.criterion.abs import Criterion
from speechain.utilbox.train_util import make_mask_from_len


class CrossEntropy(Criterion):
    """
    This criterion calculates the cross entropy between model predictions and target labels.
    In this implementation, we realize the following functions:
        1. Sentence normalization. The loss will be normalized according to the length of each sentence.
        2. Label smoothing. The target label will be transformed from a sharp one-hot vector to a smooth distribution vector.
        3. Token reweighting. The weight of each token in the cross entropy calculation can be customized manually.
        If you want to customize the weights, you need to give the token dictionary.

    """

    def criterion_init(self,
                       length_normalized: bool = False,
                       label_smoothing: float = 0.0,
                       temperature: float = 1.0,
                       confid_threshold: float = 0.0,
                       confid_level: str = 'sentence',
                       token_vocab: str = None,
                       new_weights: Dict = None):
        """

        Args:
            length_normalized: bool
                Controls whether the sentence normalization is performed.
            label_smoothing: float
                Controls the scale of label smoothing. 0 means no smoothing.
            temperature: float
                Controls the temperature of the Softmax operation.
            confid_threshold: float
                Controls whether to ignore the prediction lower than the threshold for loss calculation.
            confid_level: str
                The level of confidence calculation. Either 'token' (token_level confidence) or 'sent' (sentence-level
                confidence). Default to be 'sentence'.
            token_vocab: str
                The path of the given token vocabulary list. Necessary if new_weights is not None.
            new_weights: Dict
                The customized token weights to calculate the cross entropy. Must be given in the format below:
                'new_weights:
                    token1: weight1
                    token2: weight2
                    ...'
        """

        assert 0 <= label_smoothing < 1.0, \
            f"The value of label_smoothing should be a float number in [0, 1), but got {label_smoothing}!"
        assert temperature >= 0.0, \
            f"The value of temperature should be a non-negative float number, but got {temperature}"
        assert 0 <= confid_threshold < 1.0, \
            f"The value of confid_threshold should be a float number in [0, 1), but got {label_smoothing}!"
        assert confid_level in ['sentence', 'token'], \
            f"confid_level must be one of ['sentence', 'token'], but got {confid_level}"

        # para recording
        self.length_normalized = length_normalized
        self.label_smoothing = label_smoothing
        self.temperature = temperature
        self.confid_threshold = confid_threshold
        self.confid_level = confid_level
        self.token_weights = None

        # update the token weights if new_weights is given
        if new_weights is not None:
            assert token_vocab is not None, \
                "Please specify a token dictionary by 'token_vocab' if you want to customize the token weights."

            token_dict = np.loadtxt(token_vocab, dtype=str, delimiter="\n")
            token_dict = dict(zip(token_dict, np.arange(0, token_dict.shape[0])))
            self.token_weights = torch.ones(len(token_dict)).cuda().detach()

            for token, weight in new_weights.items():
                self.token_weights[token_dict[token]] = weight

    def __call__(self,
                 logits: torch.Tensor,
                 text: torch.Tensor,
                 text_len: torch.Tensor):
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

        # reshape predictions and do log-softmax
        batch, seq_maxlen, vocab_size = logits.size()
        log_prob = torch.log_softmax(
            logits.contiguous().view(batch * seq_maxlen, vocab_size) / self.temperature, dim=-1)

        # reshape targets and calculate the loss
        log_prob_target = log_prob.gather(1, text.contiguous().view(-1, 1)).squeeze(dim=-1)
        if self.label_smoothing > 0:
            smooth_pos = 1 - self.label_smoothing
            smooth_neg = self.label_smoothing / vocab_size
            loss = (log_prob_target * smooth_pos) + (log_prob * smooth_neg).sum(dim=1)
        else:
            loss = log_prob_target

        # reweight each token in the calculated loss
        if self.token_weights is not None:
            loss = loss * self.token_weights.index_select(0, text.reshape(-1))

        # convert the text length into the bool masks
        text_mask = make_mask_from_len(text_len, return_3d=False)
        if text.is_cuda:
            text_mask = text_mask.cuda(text.device)

        # padding the extra part of each sentence by zeros
        loss_mask = ~text_mask.reshape(-1)
        if loss.is_cuda:
            loss_mask = loss_mask.cuda(loss.device)
        if self.confid_threshold > 0:
            # padding the token predictions whose token-level confidences are lower than the threshold
            if self.confid_level == 'token':
                # confid_mask: (batch * seq_maxlen,)
                confid_mask = log_prob_target <= math.log(self.confid_threshold)
                loss_mask = torch.logical_or(loss_mask, confid_mask)
                # update text_len by confid_mask for normalization (mask the extra part of each sentence)
                # (batch * seq_maxlen,) -> (batch, seq_maxlen) -> (batch,)
                text_len = (~confid_mask.reshape(batch, seq_maxlen)).masked_fill(~text_mask, False).sum(dim=-1)
                # whether a sentence is valid (i.e. contains unmasked token prediction): (batch,)
                valid_sent = text_len > 0
            # padding the whole sentence predictions whose sentence-level confidences are lower than the threshold
            elif self.confid_level == 'sentence':
                # sent_confid: (batch * seq_maxlen,) -> (batch, seq_maxlen) -> (batch, 1)
                sent_confid = log_prob_target.reshape(batch, seq_maxlen)\
                    .masked_fill(~text_mask, 0.0).sum(dim=-1, keepdim=True)
                # confid_mask: (batch, 1)
                confid_mask = sent_confid <= (text_len * math.log(self.confid_threshold)).unsqueeze(-1)
                # confid_mask: (batch, 1) -> (batch, seq_maxlen) -> (batch * seq_maxlen,)
                loss_mask = torch.logical_or(loss_mask, confid_mask.expand(-1, seq_maxlen).reshape(-1))
                # whether a sentence is valid (i.e. unmasked sentence): (batch, 1) -> (batch,)
                valid_sent = ~confid_mask.squeeze(-1)
            else:
                raise RuntimeError(f"confid_level must be one of ['sentence', 'token'], but got {self.confid_level}")
        else:
            valid_sent = None

        loss = loss.masked_fill(loss_mask, 0.0).reshape(batch, seq_maxlen).sum(dim=-1)

        # normalize the loss by the token sequence length if specified
        if self.length_normalized:
            loss /= text_len + 1e-10

        # valid_sent is used to calculate the number of valid sentence included in the loss
        return -loss.mean() if self.confid_threshold == 0 else -loss.sum() / (torch.sum(valid_sent) + 1e-10)
