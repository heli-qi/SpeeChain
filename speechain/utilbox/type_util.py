"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.08
"""


def str2bool(input_str: str) -> bool:
    if input_str.lower() == 'true':
        return True
    elif input_str.lower() == 'false':
        return False
    else:
        raise ValueError
