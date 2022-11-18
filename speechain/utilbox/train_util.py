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


def make_mask_from_len(data_len: torch.Tensor, max_len: int = None, mask_type: torch.dtype = torch.bool):
    """

    Args:
        data_len: (batch_size,)
            The length of each sequence in the batch
        max_len: int
            The max length of the mask matrix. Could be larger than the real length of data_len
        mask_type: torch.dtype
            The value type of the mask matric.

    Returns:
        The corresponding mask for this batch.
        The parts at the end of the shorter sequence will be False or 0.0.

    """
    batch_size = data_len.size(0)
    if max_len is None:
        max_len = data_len.max()

    mask = torch.zeros((batch_size, 1, max_len), dtype=mask_type)
    for i in range(data_len.size(0)):
        mask[i, :, :data_len[i]] = 1.0

    return mask
