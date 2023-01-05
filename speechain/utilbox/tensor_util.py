"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import torch
import numpy as np


def to_native(x, tgt: str):
    if hasattr(x, 'detach'):
        x = x.detach()
    if hasattr(x, 'cpu'):
        x = x.cpu()

    if isinstance(x, torch.Tensor):
        if len(x.shape) == 0:
            x = [x.item()]
            if tgt == 'numpy':
                x = np.array(x)
        else:
            if tgt == 'list':
                assert hasattr(x, 'tolist')
                x = x.tolist()
            elif tgt == 'numpy':
                assert hasattr(x, 'numpy')
                x = x.numpy()
    return x


def detach(inputs):
    if isinstance(inputs, tuple) or isinstance(inputs, list):
        inputs = tuple([detach(x) for x in inputs])
    elif isinstance(inputs, dict):
        inputs = {k: detach(v) for k, v in inputs.items()}
    elif isinstance(inputs, torch.Tensor):
        inputs = inputs.detach()
    return inputs


def clone(inputs):
    if isinstance(inputs, tuple) or isinstance(inputs, list):
        inputs = tuple([detach(x) for x in inputs])
    elif isinstance(inputs, dict):
        inputs = {k: detach(v) for k, v in inputs.items()}
    elif isinstance(inputs, torch.Tensor):
        inputs = inputs.detach()
    return inputs


def to_cuda(inputs, rank=0):
    if isinstance(inputs, tuple) or isinstance(inputs, list):
        inputs = tuple([to_cuda(x, rank) for x in inputs])
    elif isinstance(inputs, dict):
        inputs = {k: to_cuda(v, rank) for k, v in inputs.items()}
    elif isinstance(inputs, torch.Tensor) and not inputs.is_cuda:
        inputs = inputs.to(f'cuda:{rank}')
    return inputs


def to_cpu(inputs, tgt: str = 'list', batch_idx: int = None):
    if isinstance(inputs, tuple):
        inputs = tuple([to_cpu(x, tgt, batch_idx) for x in inputs])
    elif isinstance(inputs, list):
        inputs = [to_cpu(x, tgt, batch_idx) for x in inputs]
    elif isinstance(inputs, dict):
        inputs = {k: to_cpu(v, tgt, batch_idx) for k, v in inputs.items()}
    elif isinstance(inputs, torch.Tensor):
        if batch_idx is not None and len(inputs.shape) > 0:
            inputs = inputs[batch_idx]
        inputs = to_native(inputs, tgt.lower())
    return inputs


def from_batch(inputs, batch_idx=None):
    if isinstance(inputs, tuple) or isinstance(inputs, list):
        inputs = tuple([from_batch(x, batch_idx) for x in inputs])
    elif isinstance(inputs, dict):
        inputs = {k: from_batch(v, batch_idx) for k, v in inputs.items()}
    elif isinstance(inputs, torch.Tensor):
        if batch_idx is not None and len(inputs.shape) > 0:
            inputs = inputs[batch_idx]
    elif isinstance(inputs, np.ndarray):
        if batch_idx is not None and len(inputs.shape) > 0:
            inputs = inputs[batch_idx]
    return inputs

