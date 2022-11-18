"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import sys, os
import logging
import logging.handlers
import time
import numpy as np
import humanfriendly
import torch

from speechain.model.abs import Model
from contextlib import contextmanager


@contextmanager
def distributed_zero_first(distributed: bool, rank: int):
    """
    Decorator to make all other processes in distributed training wait for the master process to do something.
    Have no influence on the single-GPU training case.

    However, this ContextManager function will cause an extra GPU memory consumption for each process in the
    multi-GPU training setting. These memory occupations are neither allocated memory nor reserved memory,
    which may be the CUDA context memory. I haven't found any effective solutions to release them so far.
    """
    if distributed and rank != 0:
        torch.distributed.barrier()
    yield
    if distributed and rank == 0:
        torch.distributed.barrier()


def logger_stdout_file(log_path, file_name: str = None, distributed: bool = False,
                       rank: int = 0, name_candidate: int = 1000):
    """

    Args:
        log_path:
        file_name:
        rank:
        log_candidate:

    Returns:

    """

    # initialize the logger, use time.time() makes sure that we always get unique logger
    rootLogger = logging.getLogger(str(time.time()))
    rootLogger.setLevel(logging.INFO)

    # initialize the file handler
    logFormatter = logging.Formatter("[ %(asctime)s | %(levelname)s ] %(message)s", "%d/%m/%Y %H:%M:%S")
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    # return empty logger if no specified file
    if file_name is None:
        return rootLogger
    else:
        # looping all the candidate names
        result_log = None
        for i in range(name_candidate):
            result_log = os.path.join(log_path, f'{file_name}.log' if i == 0 else f'{file_name}{i}.log')
            # non-existing file is the target
            if not os.path.exists(result_log):
                break

    # the logger is only functional for single-GPU training or master process of the multi-GPU training
    if not distributed or rank == 0:
        # initialize the file handler for writing the info to the disk
        fileHandler = logging.FileHandler(result_log)
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

        # we don't initialize the console handler because showing the info on the console may have some problems
        # consoleHandler = logging.StreamHandler(sys.stdout)
        # consoleHandler.setFormatter(logFormatter)
        # rootLogger.addHandler(consoleHandler)

    # For the non-master multi-GPU training processes or testing processes, an empty logger will be returned
    return rootLogger


def model_summary(model: Model) -> str:
    """
    Return the information summary of the model for logging.

    Codes borrowed from
    https://github.com/espnet/espnet/blob/a2abaf11c81e58653263d6cc8f957c0dfd9677e7/espnet2/torch_utils/model_summary.py#L48

    Args:
        model:

    Returns:

    """

    def get_human_readable_count(number: int) -> str:
        """Return human_readable_count
        Originated from:
        https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/core/memory.py
        Abbreviates an integer number with K, M, B, T for thousands, millions,
        billions and trillions, respectively.
        Examples:
            >>> get_human_readable_count(123)
            '123  '
            >>> get_human_readable_count(1234)  # (one thousand)
            '1 K'
            >>> get_human_readable_count(2e6)   # (two million)
            '2 M'
            >>> get_human_readable_count(3e9)   # (three billion)
            '3 B'
            >>> get_human_readable_count(4e12)  # (four trillion)
            '4 T'
            >>> get_human_readable_count(5e15)  # (more than trillion)
            '5,000 T'
        Args:
            number: a positive integer number
        Return:
            A string formatted according to the pattern described above.
        """
        assert number >= 0
        labels = [" ", "K", "M", "B", "T"]
        num_digits = int(np.floor(np.log10(number)) + 1 if number > 0 else 1)
        num_groups = int(np.ceil(num_digits / 3))
        num_groups = min(num_groups, len(labels))  # don't abbreviate beyond trillions
        shift = -3 * (num_groups - 1)
        return f"{number * (10 ** shift):.2f} {labels[num_groups - 1]}"

    def to_bytes(dtype) -> int:
        # torch.float16 -> 16
        return int(str(dtype)[-2:]) // 8

    message = "Model structure:\n"
    message += str(model)
    tot_params = sum(p.numel() for p in model.parameters())
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percent_trainable = "{:.1f}".format(num_params * 100.0 / tot_params)
    tot_params = get_human_readable_count(tot_params)
    num_params = get_human_readable_count(num_params)
    message += "\n\nModel summary:\n"
    message += f"    Class Name: {model.__class__.__name__}\n"
    message += f"    Total Number of model parameters: {tot_params}\n"
    message += (
        f"    Number of trainable parameters: {num_params} ({percent_trainable}%)\n"
    )
    num_bytes = humanfriendly.format_size(
        sum(
            p.numel() * to_bytes(p.dtype) for p in model.parameters() if p.requires_grad
        )
    )
    message += f"    Size: {num_bytes}\n"
    dtype = next(iter(model.parameters())).dtype
    message += f"    Type: {dtype}"
    return message
