"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import torch
from speechain.module.abs import Module


class DeltaFeature(Module):
    """

    """
    def module_init(self,
                    delta_order: int = 1,
                    delta_N: int = 2):
        """

        Args:
            delta_order:
            delta_N:

        Returns:

        """
        assert isinstance(delta_order, int) and delta_order in [1, 2], \
            f"delta_order must be an integer of 1 or 2, but got delta_order={delta_order}"
        self.delta_order = delta_order

        # initialze the filters for extracting delta features
        # modified from https://github.com/jameslyons/python_speech_features/blob/master/python_speech_features/base.py#L195
        assert isinstance(delta_N, int) and delta_N >= 1, \
            f"delta_N must be a positive integer equal to or larger than 1"
        _filter_weights = torch.arange(-delta_N, delta_N + 1) / (2 * sum([i ** 2 for i in range(1, delta_N + 1)]))
        _delta_filters = torch.nn.Conv2d(in_channels=1, out_channels=1,
                                         kernel_size=(2 * delta_N + 1, 1), padding=(delta_N, 0), bias=False)
        _delta_filters.weight = torch.nn.Parameter(_filter_weights.reshape(1, 1, -1, 1), requires_grad=False)

        # move the weight from _parameters to _buffers
        _para_keys = list(_delta_filters._parameters.keys())
        for name in _para_keys:
            _delta_filters._buffers[name] = _delta_filters._parameters.pop(name)
        self.delta_filters = _delta_filters


    def forward(self, feat: torch.Tensor, feat_len: torch.Tensor):
        """

        Args:
            feat:
            feat_len:

        Returns:

        """

        # first-order derivative
        feat_stack = [feat]
        delta_feat = self.delta_filters(feat.unsqueeze(1)).squeeze()
        feat_stack.append(delta_feat)

        # (Optional) second-order derivative
        if self.delta_order == 2:
            delta2_feat = self.delta_filters(delta_feat.unsqueeze(1)).squeeze()
            feat_stack.append(delta2_feat)

        # combine the original features with all delta features
        feat = torch.cat(feat_stack, dim=-1)

        # make sure the silence parts are zeros
        for i in range(feat.size(0)):
            if feat_len[i] < feat.size(1):
                feat[i][feat_len[i]:] = 0.0

        return feat, feat_len