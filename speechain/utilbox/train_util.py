"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import torch
import random

from typing import Dict, List, Any, Union, Tuple

from speechain.criterion.abs import Criterion


# dummy activation #
class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input

    def __repr__(self):
        return self.__class__.__name__ + '()'


def swish_activation(x: torch.Tensor):
    return x * torch.sigmoid(x)


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
    else:
        assert max_len >= data_len.max(), "max_len should be larger than the maximum of data_len!"

    if return_3d:
        mask = torch.zeros((batch_size, 1, max_len), dtype=mask_type)
        for i in range(data_len.size(0)):
            mask[i, :, :data_len[i]] = 1.0
    else:
        mask = torch.zeros((batch_size, max_len), dtype=mask_type)
        for i in range(data_len.size(0)):
            mask[i, :data_len[i]] = 1.0

    if data_len.is_cuda:
        mask = mask.cuda(data_len.device)
    return mask


def make_len_from_mask(data_mask: torch.Tensor):
    if len(data_mask.shape) == 3:
        data_mask = data_mask.squeeze(1)
    return data_mask.sum(dim=-1)


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


def text2tensor_and_len(text_list: List[str or List[str]], text2tensor_func, ignore_idx: int) \
        -> (torch.LongTensor, torch.LongTensor):
    """

    Args:
        text_list:
        text2tensor_func:
        ignore_idx:

    Returns:

    """
    for i in range(len(text_list)):
        text_list[i] = text2tensor_func(text_list[i])
    text_len = torch.LongTensor([len(t) for t in text_list])

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


def float_near_round(input_float: float):
    """

    Round the float number in [X.0, X.5) to X and the float number in (X.5, {X+1}.0] to X+1

    """
    int_part = int(input_float)
    frac_part = input_float - int_part
    if frac_part < 0.5:
        return int_part
    else:
        return int_part + 1


def get_padding_by_dilation(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def get_min_indices_by_freq(freq_dict: Dict[Any, Union[int, float]],
                            shuffle: bool = True,
                            chosen_idx_num: int = 1,
                            freq_weights: Union[int, float] or List[Union[int, float]] = None) \
        -> Tuple[List, Dict[Any, Union[int, float]]]:
    """
    Get the specified number of indices with minimum values from the input frequency dictionary,
    optionally applying frequency weights, and return the updated frequency dictionary.

    Args:
        freq_dict (Dict[Any, Union[int, float]]):
            The input frequency dictionary containing the values to be compared.
        shuffle (bool, optional):
            If True, shuffle the input frequency dictionary before selecting the minimum indices. Defaults to True.
        chosen_idx_num (int, optional):
            The number of minimum indices to return. If not provided, all indices will be returned.
        freq_weights (List[Union[int, float]], optional):
            The frequency weights to apply when selecting the minimum indices.
            Should have the same length as the number of indices to return.
            If not provided, equal weights will be applied.

    Returns:
        Tuple[List[int], Dict[Any, Union[int, float]]]:
            A tuple containing a list of the selected minimum indices and the updated frequency dictionary.
    """

    # Ensure the frequency weights have the same length as the number of indices to return
    if freq_weights is not None:
        if not isinstance(freq_weights, List):
            freq_weights = [freq_weights]
        assert len(freq_weights) == chosen_idx_num
    else:
        freq_weights = [1 for _ in range(chosen_idx_num)]

    min_indices = []

    # Select the minimum indices based on the frequency weights
    for i in range(chosen_idx_num):
        index_freq_list = list(freq_dict.items())
        if shuffle:
            random.shuffle(index_freq_list)

        # Sort the indices based on their values in the input frequency dictionary
        sorted_indices = [idx for idx, _ in sorted(index_freq_list, key=lambda x: x[1])]

        # Append the last index (minimum value) to the list of minimum indices
        min_indices.append(sorted_indices[0])

        # Update the input frequency dictionary by adding the frequency weight to the value of the selected index
        freq_dict[sorted_indices[0]] += freq_weights[i]

    return min_indices, freq_dict
