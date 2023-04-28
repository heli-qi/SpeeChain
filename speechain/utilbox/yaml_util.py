"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.09
"""
import os

import ruamel.yaml
from ruamel.yaml.scalarfloat import ScalarFloat
from ruamel.yaml.scalarstring import PlainScalarString
from typing import Dict, List
from speechain.utilbox.regex_util import regex_angle_bracket


def reform_config_dict(input_config):
    if isinstance(input_config, Dict):
        return {str(key): reform_config_dict(value) for key, value in input_config.items()}
    elif isinstance(input_config, List):
        return [reform_config_dict(item) for item in input_config]
    else:
        # convert the rumel.yaml data type into normal python data type
        if isinstance(input_config, ScalarFloat):
            input_config = float(input_config)
        elif isinstance(input_config, PlainScalarString):
            input_config = str(input_config)
        return input_config


def remove_representer(parent_node, reference, curr_key=None):
    child_node = parent_node[curr_key] if curr_key is not None else parent_node

    if isinstance(child_node, Dict):
        return {key: remove_representer(child_node, reference, key) for key, value in child_node.items()}
    elif isinstance(child_node, List):
        return [remove_representer(child_node, reference, i) for i, item in enumerate(child_node)]
    else:
        # for the item with an !-prefixed representer
        if hasattr(child_node, 'tag'):
            # for '!ref' representer
            if child_node.tag.value == '!ref':
                ref_string = child_node.value
                # standalone '!ref' without <> reference or the anchor points where the reference is already done
                if regex_angle_bracket.search(ref_string) is None:
                    parent_node[curr_key] = parent_node[curr_key].value
                # '!ref' with <> reference
                else:
                    # <> reference-only without any additional information, retain the data type of the reference
                    if regex_angle_bracket.fullmatch(ref_string):
                        parent_node[curr_key] = reference[ref_string[1: -1]]
                    # reference with additional information, data type will be str
                    else:
                        ref_matches = regex_angle_bracket.findall(ref_string)

                        # loop each match surrounded by <>
                        for ref in ref_matches:
                            # get the reference key
                            ref_key = ref[1: -1]
                            # for the reference indicated by a yaml built-in representer
                            if hasattr(reference[ref_key], 'tag'):
                                assert reference[ref_key].tag.value != '!ref', \
                                    "The items indicated by '!ref' should be given in order."
                                reference[ref_key] = reference[ref_key].value
                            # only change the value during looping
                            parent_node[curr_key].value = parent_node[curr_key].value.replace(ref, str(reference[ref_key]))
                        # assign the value to input_config after looping
                        parent_node[curr_key] = parent_node[curr_key].value

            # turn the string '(x, x, ..., x)' into tuple by '!tuple'
            elif child_node.tag.value == '!tuple':
                parent_node[curr_key] = tuple([int(i) if i.isnumeric() else i
                                               for i in parent_node[curr_key].value[1: -1].replace(' ', '').split(',')])

            # turn the string '[x, x, ..., x]' into list by '!list'
            elif child_node.tag.value == '!list':
                parent_node[curr_key] = [int(i) if i.isnumeric() else i
                                         for i in parent_node[curr_key].value[1: -1].replace(' ', '').split(',')]

            # turn the numerical value into string by '!str'
            elif child_node.tag.value == '!str':
                parent_node[curr_key] = str(parent_node[curr_key].value)

        return parent_node[curr_key]


def load_yaml(yaml_file) -> Dict:
    """
    yaml parsing function for !-prefixed representers, inspired by
    https://github.com/speechbrain/speechbrain/blob/82cbc089347083cf4e87bce27011bb55d2924569/speechbrain/yaml.py#L17

    Args:
        yaml_file: str or file IO stream

    Returns:
        A Dict that contains the well-parsed configurations

    """
    # turn the input file path into file IO stream
    if isinstance(yaml_file, str):
        assert os.path.exists(yaml_file), f"Your input .yaml file {yaml_file} doesn't exist!"
        yaml_file = open(yaml_file, mode='r', encoding='utf-8')

    # parse the yaml file with !-prefixed representers
    ruamel_yaml = ruamel.yaml.YAML()
    yaml_config = reform_config_dict(ruamel_yaml.load(yaml_file))

    # modify the value of each item in yaml_config in-place if there is an !-prefixed representer
    yaml_config = remove_representer(yaml_config, yaml_config)

    return yaml_config
