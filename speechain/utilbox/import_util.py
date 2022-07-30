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


def get_port():
    """
    find a free port to used for distributed learning

    """
    pscmd = "netstat -ntl |grep -v Active| grep -v Proto|awk '{print $4}'|awk -F: '{print $NF}'"
    procs = os.popen(pscmd).read()
    procarr = procs.split("\n")
    tt= str(random.randint(15000, 30000))
    if tt not in procarr:
        return tt
    else:
        return get_port()