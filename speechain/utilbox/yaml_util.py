"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.09
"""
import ruamel.yaml

from typing import Dict, List
from speechain.utilbox.regex_util import regex_angle_bracket


def reform_config_dict(input_config):
    if isinstance(input_config, Dict):
        return {key: reform_config_dict(value) for key, value in input_config.items()}
    elif isinstance(input_config, List):
        return [reform_config_dict(item) for item in input_config]
    else:
        return input_config


def remove_representer(input_config, reference):
    if isinstance(input_config, Dict):
        return {key: remove_representer(value, reference) for key, value in input_config.items()}
    elif isinstance(input_config, List):
        return [remove_representer(item, reference) for item in input_config]
    else:
        # for the item with an !-prefixed representer
        if hasattr(input_config, 'tag'):
            # for '!ref' representer
            if input_config.tag.value == '!ref':
                ref_string = input_config.value
                assert regex_angle_bracket.search(ref_string) is not None, \
                    "If you want to use '!ref' to simply your config, " \
                    "you should indicate the key names that you want to refer to by declaring them with <>."

                # reference only without any additional information, retain the data type of the reference
                if regex_angle_bracket.fullmatch(ref_string):
                    input_config = reference[ref_string[1: -1]]
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
                        input_config.value = input_config.value.replace(ref, str(reference[ref_key]))
                    # assign the value to input_config after looping
                    input_config = input_config.value

            # other yaml built-in representers, such as '!str'
            else:
                input_config = input_config.value

        return input_config


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
        yaml_file = open(yaml_file, mode='r', encoding='utf-8')

    # parse the yaml file with !-prefixed representers
    ruamel_yaml = ruamel.yaml.YAML()
    yaml_config = reform_config_dict(ruamel_yaml.load(yaml_file))

    # modify the value of each item in yaml_config in-place if there is an !-prefixed representer
    yaml_config = remove_representer(yaml_config, yaml_config)

    return yaml_config