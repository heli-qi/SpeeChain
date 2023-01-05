"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import importlib
import functools
import os
import random
import warnings
from typing import List

from GPUtil import GPU, getGPUs


@functools.lru_cache(maxsize=None)
def import_class(class_string):
    class_string = class_string.split('.')
    module_name = '.'.join(class_string[:-1]).strip()
    class_name = class_string[-1].strip()
    return getattr(importlib.import_module(module_name), class_name)


def get_idle_port() -> str:
    """
    find an idle port to used for distributed learning

    """
    pscmd = "netstat -ntl |grep -v Active| grep -v Proto|awk '{print $4}'|awk -F: '{print $NF}'"
    procs = os.popen(pscmd).read()
    procarr = procs.split("\n")
    tt = str(random.randint(15000, 30000))
    if tt not in procarr:
        return tt
    else:
        return get_idle_port()


def get_idle_gpu(ngpu: int = 1, id_only: bool = False) -> List[GPU]:
    """

    find idle GPUs for distributed learning.

    """
    sorted_gpus = sorted(getGPUs(), key=lambda g: g.memoryUtil)
    if len(sorted_gpus) < ngpu:
        warnings.warn(f"Your machine doesn't have enough GPUs ({len(sorted_gpus)}) as you specified ({ngpu})! "
                      f"Currently, only {len(sorted_gpus)} GPUs are used.")
    sorted_gpus = sorted_gpus[:ngpu]

    if id_only:
        return [gpu.id for gpu in sorted_gpus]
    else:
        return sorted_gpus


def parse_path_args(input_path: str) -> str:
    """
    This function parses the input path string into valid path before its later usage.

    Args:
        input_path: str
            If input_path starts with '/', no parsing will be done;
            Otherwise, this function will return its absolute path in the toolkit root, i.e., SPEECHAIN_ROOT.

    Returns: str
        Parsed valid absolute path.

    """

    # do nothing for the absolute path
    if input_path.startswith('/'):
        return input_path

    # turn the general relative path into its absolute value
    elif input_path.startswith('.'):
        return os.path.abspath(input_path)

    # turn the in-toolkit relative path into its absolute value
    else:
        assert 'SPEECHAIN_ROOT' in os.environ.keys(), \
            "SPEECHAIN_ROOT doesn't exist in your environmental variables! " \
            "Please move to the toolkit root and execute envir_preparation.sh there to build the toolkit environment!"
        return os.path.join(os.environ['SPEECHAIN_ROOT'], input_path)
