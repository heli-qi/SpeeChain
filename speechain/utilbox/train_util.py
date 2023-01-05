"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
from typing import Dict, List

import torch
from speechain.criterion.abs import Criterion


# dummy activation #
class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, input):
        return input

    def __repr__(self):
        return self.__class__.__name__ + '()'


def make_mask_from_len(data_len: torch.Tensor, max_len: int = None,
                       mask_type: torch.dtype = torch.bool, return_3d: bool = True):
    """

    Args:
        data_len: (batch_size,)
            The length of each sequence in the batch
        max_len: int
            The max length of the mask matrix. Could be larger than the real length of data_len
        mask_type: torch.dtype
            The value type of the mask matric.
        return_3d: bool
            Whether to return a 3d mask tensors.
            If True, the returned mask tensor will be 3d (batch_size, 1, max_len)
            If False, the returned mask tensor will be 2d (batch_size, max_len)

    Returns:
        The corresponding mask for this batch.
        The parts at the end of the shorter sequence will be False or 0.0.

    """
    batch_size = data_len.size(0)
    if max_len is None:
        max_len = data_len.max()

    if return_3d:
        mask = torch.zeros((batch_size, 1, max_len), dtype=mask_type)
        for i in range(data_len.size(0)):
            mask[i, :, :data_len[i]] = 1.0
    else:
        mask = torch.zeros((batch_size, max_len), dtype=mask_type)
        for i in range(data_len.size(0)):
            mask[i, :data_len[i]] = 1.0

    return mask


def recur_criterion_init(input_conf: Dict, criterion_class: Criterion):
    """

    Args:
        input_conf:
        criterion_class:

    Returns:

    """
    #
    leaf_flags = [not isinstance(value, Dict) for value in input_conf.values()]

    #
    if sum(leaf_flags) == len(input_conf):
        return {key: recur_criterion_init(value, criterion_class) for key, value in input_conf.items()}
    #
    elif sum(leaf_flags) == 0:
        return criterion_class(**input_conf)
    else:
        raise RuntimeError


def text2tensor_and_len(text_list: List[str], text2tensor_func, ignore_idx: int) \
        -> (torch.LongTensor, torch.LongTensor):
    """

    Args:
        text_list:
        text2tensor_func:
        ignore_idx:

    Returns:

    """
    #
    for i in range(len(text_list)):
        text_list[i] = text2tensor_func(text_list[i])
    text_len = torch.LongTensor([len(t) for t in text_list])

    #
    text = torch.full((text_len.size(0), text_len.max().item()), ignore_idx, dtype=text_len.dtype)
    for i in range(text_len.size(0)):
        text[i][:text_len[i]] = text_list[i]

    return text, text_len


def spk2tensor(spk_list: List[str], spk2idx_dict: Dict) -> torch.LongTensor:
    """

    Args:
        spk_list:
        spk2idx_dict:

    Returns:

    """
    # turn the speaker id strings into the tensors
    return torch.LongTensor([spk2idx_dict[spk] if spk in spk2idx_dict.keys() else 0 for spk in spk_list])
