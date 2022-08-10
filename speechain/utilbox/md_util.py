from typing import List, Dict


def get_table_strings(contents: List[List] or List, first_col: List, first_col_bold: bool = True,
                      headers: List = None, header_bold: bool = True):
    """

    Args:
        contents:
        first_col:
        first_col_bold:
        headers:
        header_bold:

    Returns:

    """
    if not isinstance(contents[0], List):
        contents = [contents]
    assert len(first_col) == len(contents), "The lengths of first_col and contents don't match!"

    table_strings = ""
    if headers is not None:
        assert len(headers) == len(contents[0]) + 1, "The lengths of headers and contents don't match!"
        table_strings += '|' + '|'.join([f'**{h}**' if header_bold else h for h in headers]) + '|\n'
    else:
        table_strings += '|' + '|'.join(['' for _ in range(len(contents[0]) + 1)]) + '|\n'

    table_strings += '|---|' + ''.join(['---|' for _ in contents[0]]) + '\n'

    for i in range(len(contents)):
        table_strings += f"|{f'**{first_col[i]}**' if first_col_bold else first_col[i]}|" + '|'.join(contents[i]) + '|\n'

    return table_strings


def get_list_strings(content_dict: Dict, header_bold: bool = True):
    """

    Args:
        content_dict:
        header_bold:

    Returns:

    """

    list_strings = ""
    for header, content in content_dict.items():
        list_strings += f"* {f'**{header}:**' if header_bold else f'{header}:'} {content}\n"

    return list_strings
