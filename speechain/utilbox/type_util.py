"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.08
"""
from typing import Dict, List
from speechain.utilbox.regex_util import regex_square_bracket, regex_brace


def str2bool(input_str: str) -> bool:
    # spelling error tolerance for your convenience, lol~~~
    if input_str.lower() in ['true', 'ture']:
        return True
    elif input_str.lower() in ['false', 'flase']:
        return False
    else:
        raise ValueError


def str2dict(input_str: str) -> Dict:
    """
    Examples:
        input string:
            a:{b:12.3,c:{d:123,e:{g:xyz}}},g:xyz

        output dict:
            a:
                b: 12.3
                c:
                    d: 123
                    e:
                        g:xyz
            g: xyz

    Args:
        input_str: str
            The input string to be parsed into the corresponding nested dict

    Returns: Dict
        The nested dict after parsing

    """
    def recur_dict_init(unproc_string: str):
        assert (not unproc_string.startswith('{')) and (not unproc_string.endswith('}'))

        # there is no commas inside
        if ',' not in unproc_string:
            assert ':' in unproc_string and unproc_string.count(':') == 1
            key, value = unproc_string.split(':')[0], unproc_string.split(':')[-1]

            # continue the recursion for the register matches in the Dict
            if value.startswith('match_'):
                return {key: recur_dict_init(match_dict[value])}
            else:
                # convert the digital string into a integer
                if value.isdigit():
                    value = int(value)
                # convert the non-integer string into a float number
                elif '.' in value and value.replace('.', '').isdigit():
                    value = float(value)
                return {key: value}

        # there are commas inside
        else:
            # remove the brackets at the beginning and the end
            proc_list = unproc_string.split(',')
            proc_list = [recur_dict_init(ele) for ele in proc_list]
            proc_dict = proc_list[0]
            for i in range(1, len(proc_list)):
                proc_dict.update(proc_list[i])
            return proc_dict

    # input string checking
    assert (not input_str.startswith('{')) and (not input_str.endswith('}')), \
        "If you want the framework to automatically convert your input string into a list, " \
        "please don't surround it by a pair of braces '{}'."
    assert input_str.count('{') == input_str.count('}'), \
        "The number of left braces '{' doesn't match that of right braces '}'."
    # remove the blanks inside
    input_str = input_str.replace(' ', '')

    # register all the smallest []-surrounded sub-strings into a nested Dict
    match_dict, match_num = {}, 0
    while True:
        regex_matches = regex_brace.findall(input_str)
        if len(regex_matches) == 0:
            break

        for match in regex_matches:
            # remove the brackets and register the match sub-string
            match_dict[f"match_{match_num}"] = match[1:-1]
            input_str = input_str.replace(match, f"match_{match_num}", 1)
            match_num += 1

    return recur_dict_init(input_str)


def str2list(input_str: str) -> List:
    """
    Examples:
        input string:
            [a,[1,2,[1.1,2.2,3.3],[h,i,j,k]],c,[d,e,[f,g,[h,i,j,k]]]]

        output list:
            - 'a'
            - - 1
              - 2
              - - 1.1
                - 2.2
                - 3.3
              - - 'h'
                - 'i'
                - 'j'
                - 'k'
            - 'c'
            - - 'd'
              - 'e'
              - - 'f'
                - 'g'
                - - 'h'
                  - 'i'
                  - 'j'
                  - 'k'

    Args:
        input_str: str
            The input string to be parsed into the corresponding nested list

    Returns: List
        The nested list after parsing

    """
    def cast_single_string(single_string: str):
        # convert the digital string into a integer
        if single_string.isdigit():
            return int(single_string)
        # convert the non-integer string into a float number
        elif '.' in single_string and single_string.replace('.', '').isdigit():
            return float(single_string)
        # for other strings, remain the same type
        else:
            return single_string

    def recur_list_init(unproc_string: str):
        assert (not unproc_string.startswith('[')) and (not unproc_string.endswith(']'))

        # there is no commas inside
        if ',' not in unproc_string:
            # continue the recursion for the register matches in the Dict
            if unproc_string.startswith('match_'):
                return recur_list_init(match_dict[unproc_string])
            # cast the leaf string into its corresponding type
            else:
                return cast_single_string(unproc_string)

        # there are commas inside
        else:
            # remove the brackets at the beginning and the end
            proc_list = unproc_string.split(',')
            return [recur_list_init(ele) for ele in proc_list]

    # directly return a single string element in a one-element list
    if '[' not in input_str and ']' not in input_str and ',' not in input_str:
        return [cast_single_string(input_str)]

    # input string checking
    assert input_str.startswith('[') and input_str.endswith(']'), \
        "If you want the framework to automatically convert your input string into a list, " \
        "please surround it by a pair of square brackets '[]'."
    assert input_str.count('[') == input_str.count(']'), \
        "The number of left square brackets '[' doesn't match that of right square brackets ']'."
    # remove the blanks inside
    input_str = input_str.replace(' ', '')

    # register all the smallest []-surrounded sub-strings into a nested Dict
    match_dict, match_num = {}, 0
    while True:
        regex_matches = regex_square_bracket.findall(input_str)
        if len(regex_matches) == 0:
            break

        for match in regex_matches:
            # remove the brackets and register the match sub-string
            match_dict[f"match_{match_num}"] = match[1:-1]
            input_str = input_str.replace(match, f"match_{match_num}", 1)
            match_num += 1

    return recur_list_init(input_str)


if __name__ == '__main__':
    # test str2list function
    result_list = str2list('[a,[1,2,[1.1, 2.2, 3.3],[h,i, j, k]], c,[d, e,[f,g,[h,i,j,k]]]]')
    print(result_list)
    result_list = str2list('a')
    print(result_list)

    # test str2dict function
    result_dict = str2dict('a:{b:12.3,c:{d:123,e:{g:xyz}}},g:xyz')
    print(result_dict)
    result_dict = str2dict('a:xyz')
    print(result_dict)
