"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import importlib
import functools
import os
import random


@functools.lru_cache(maxsize=None)
def import_class(class_string):
    class_string = class_string.split('.')
    module_name = '.'.join(class_string[:-1]).strip()
    class_name = class_string[-1].strip()
    return getattr(importlib.import_module(module_name), class_name)


def get_idle_port():
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
        return get_port()


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
    assert not input_path.startswith('.'), "Please don't give the relative path as the argument!"
    # return absolute path in the machine
    if input_path.startswith('/'):
        return input_path
    # return the absolute path in the toolkit root
    else:
        assert 'SPEECHAIN_ROOT' in os.environ.keys(), \
            "SPEECHAIN_ROOT doesn't exist in your environmental variables! " \
            "Please move to the toolkit root and execute envir_preparation.sh there to build the toolkit environment!"
        return os.path.join(os.environ['SPEECHAIN_ROOT'], input_path)
