"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import torch

# dummy activation #
class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, input):
        return input

    def __repr__(self):
        return self.__class__.__name__ + '()'

def generator_act_module(name):
    if not isinstance(name, str) :
        return name
    if name is None or name.lower() in ['none', 'null'] :
        return Identity()
    else :
        return getattr(torch.nn, name)()


def make_mask_from_len(data_len: torch.Tensor):
    """

    Args:
        data_len: (batch_size,)
            The length of each sequence in the batch

    Returns:
        The corresponding mask for this batch.
        The parts at the end of the shorter sequence will be False.

    """
    mask = []
    for i in range(data_len.size(0)):
        _tmp_mask = torch.ones((1, data_len[i]), dtype=torch.bool)

        # padding zeros if shorter than the longest utterance
        if data_len.max() - data_len[i] > 0:
            pad_zero = torch.zeros((1, data_len.max() - data_len[i]), dtype=torch.bool)
            _tmp_mask = torch.cat([_tmp_mask, pad_zero], dim=1)
        mask.append(_tmp_mask)

    return torch.stack(mask)