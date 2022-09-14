"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
from typing import List, Dict


def get_table_strings(contents: List[List] or List,
                      first_col: List = None, first_col_bold: bool = True,
                      headers: List = None, header_bold: bool = True):
    """
    Return the .md string for making a table.

    Args:
        contents: List[List] or List
            The main body of the table. Each list element corresponds to a row.
        first_col: List
            The values of the first column. If not given, no first column will be added.
        first_col_bold: bool
            Controls whether the values of the first column is bolded.
        headers: List
            The values of the table headers. If not given, no header will be added.
        header_bold: bool
            Controls whether the values of the header is bolded.

    Returns:
        The well-structured .md table string.

    """
    if not isinstance(contents[0], List):
        contents = [contents]
    if first_col is not None:
        assert len(first_col) == len(contents), "The lengths of first_col and contents don't match!"
    if headers is not None:
        if first_col is not None:
            assert len(headers) == len(contents[0]) + 1, "The lengths of headers and contents don't match!"
        else:
            assert len(headers) == len(contents[0]), "The lengths of headers and contents don't match!"

    table_strings = ""
    if headers is not None:
        table_strings += '|' + '|'.join([f'**{h}**' if header_bold else h for h in headers]) + '|\n'
    else:
        if first_col is None:
            table_strings += '|'
        table_strings += '|' + '|'.join(['' for _ in range(len(contents[0]))]) + '|\n'

    table_strings += '|---|' + ''.join(['---|' for _ in contents[0]]) + '\n'

    for i in range(len(contents)):
        if first_col is None:
            table_strings += "|"
        else:
            table_strings += f"|{f'**{first_col[i]}**' if first_col_bold else first_col[i]}|"
        table_strings += '|'.join(contents[i]) + '|\n'

    return table_strings


def get_list_strings(content_dict: Dict, header_bold: bool = True):
    """
    Return the .md string for making a list.

    Args:
        content_dict: Dict
            The main body of the list. Each key-value item corresponds to a row.
            The key is the header name and the value is the content.
        header_bold: bool
            Controls whether the header names are bolded.

    Returns:
        The well-structured .md list string.

    """

    list_strings = ""
    for header, content in content_dict.items():
        list_strings += f"* {f'**{header}:**' if header_bold else f'{header}:'} {content}\n"

    return list_strings
