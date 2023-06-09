"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.11
"""
import re


def en_text_process(input_text: str, txt_format: str) -> str:
    """
    The function that processes the text strings for TTS datasets to the specified text format.
    Currently, available text formats:
        punc:
            Letter: lowercase
            Punctuation: single quotes, commas, periods, hyphens
        no-punc:
            Letter: lowercase
            Punctuation: single quotes

    Args:
        input_text: str
            Unprocessed raw sentence from the TTS datasets
        txt_format: str
            The text format you want the processed sentence to have

    Returns:
        Processed sentence string by your specified text format.

    """

    def is_punc(input_char: str):
        return not (input_char.isalpha() or input_char == ' ')

    # 1st stage: turn capital letters into their lower cases
    input_text = input_text.lower()

    # 2nd stage: convert non-English letters into English counterparts
    input_text = input_text.replace('è', 'e')
    input_text = input_text.replace('é', 'e')
    input_text = input_text.replace('ê', 'e')
    input_text = input_text.replace('â', 'a')
    input_text = input_text.replace('à', 'a')
    input_text = input_text.replace('ü', 'u')
    input_text = input_text.replace('ñ', 'n')
    input_text = input_text.replace('ô', 'o')
    input_text = input_text.replace('æ', 'ae')
    input_text = input_text.replace('œ', 'oe')

    # 3rd stage: convert all kinds of the quotes into half-angle single quotes '’'
    input_text = input_text.replace('’', '\'')
    input_text = input_text.replace('‘', '\'')
    input_text = input_text.replace('“', '\'')
    input_text = input_text.replace('”', '\'')
    input_text = input_text.replace('"', '\'')
    input_text = input_text.replace('\'\'', '\'')

    # 4th stage: process colons and semicolons
    input_text = input_text.replace(':\'', ',') # for the colons followed by a quote, turn them into commas
    input_text = input_text.replace(':', ',')
    input_text = input_text.replace(';', '.')

    # 5th stage: process double-hyphens and em dashes
    input_text = input_text.replace('--', '-')
    input_text = input_text.replace('—', '-')
    input_text = input_text.replace('¯', '-')
    input_text = input_text.replace('-', ',')
    input_text = input_text.replace('/', '.')

    # 7th stage: replace all the punctuation marks other than ',', '.', '\'', '!', '?' by a space
    _input_text_tmp = []
    for char in input_text:
        if not char.isalpha() and char not in [',', '.', '\'', '!', '?']:
            _input_text_tmp.append(' ')
            continue
        _input_text_tmp.append(char)
    input_text = ''.join(_input_text_tmp)

    # deal with single quotations by different cases
    _input_text_tmp = []
    for idx, char in enumerate(input_text):
        # save all the non-quotation characters
        if char != '\'':
            _input_text_tmp.append(char)
        # remove the quotations at the beginning or end
        elif idx == 0 or idx == len(input_text) - 1:
            continue
        # remove the quotations not surrounded by letters on both sides
        elif not input_text[idx - 1].isalpha() or not input_text[idx + 1].isalpha():
            # if a quotation is surrounded by a letter on the left and a blank on the right, turn it into a comma
            if input_text[idx - 1].isalpha() and input_text[idx + 1] == ' ':
                _input_text_tmp.append(',')
            # non-letter and non-blank character -> punctuation marks
            # turn the quotations surrounded by two punctuation marks into a blank
            elif is_punc(input_text[idx - 1]) and is_punc(input_text[idx + 1]):
                _input_text_tmp.append(' ')
            # in other cases, remove it
            else:
                continue
        # save the intra-word quotations
        else:
            _input_text_tmp.append(char)
    input_text = ''.join(_input_text_tmp)

    # 8th stage: question and exclamation marks
    input_text = re.sub('([.,!?]\s*)+!', '!', input_text)  # remove duplicated questions
    input_text = re.sub('([.,!?]\s*)+\?', '?', input_text)  # remove duplicated exclamations
    input_text = re.sub('([.,!?]\s*)+\.', '.', input_text)  # remove duplicated periods
    input_text = re.sub('([.,!?]\s*)+,', ',', input_text)  # remove duplicated commas

    # remove the blanks and punctuation marks at the beginning
    while input_text.startswith(' ') or is_punc(input_text[0]):
        input_text = ''.join(input_text[1:])
    # remove the blanks at the end
    while input_text.endswith(' '):
        input_text = ''.join(input_text[:-1])

    # remove useless blanks
    _input_text_tmp = []
    for idx, char in enumerate(input_text):
        if char == ' ':
            # remove consecutive blanks and replace them by a single blank
            if input_text[idx + 1] == ' ':
                continue
            # remove the blanks surrounded by letters on the left and punctuations on the right
            elif _input_text_tmp[-1].isalpha() and is_punc(input_text[idx + 1]):
                continue
        elif (is_punc(char) and char != '\'') and idx < len(input_text) - 1:
            # add a space between punctuation marks on the left and letters on the right
            if input_text[idx + 1].isalpha():
                _input_text_tmp.append(f'{char} ')
                continue
            # only retain the last one of consecutive punctuation marks
            elif is_punc(input_text[idx + 1]):
                continue
        _input_text_tmp.append(char)
    input_text = ''.join(_input_text_tmp)

    # remain all the punctuation marks
    if txt_format == 'punc':
        return input_text

    # remove all the punctuation marks other than single-quotations
    elif txt_format == 'no-punc':
        # remove all the punctuation symbols other than single quotes
        return ''.join([char for char in input_text if char.isalpha() or char in ["\'", ' ']])

    else:
        raise ValueError(f"txt_format must be one of 'punc' or 'no-punc'. But got {txt_format}!")



def get_readable_number(raw_number: int or float) -> str:
    if isinstance(raw_number, float):
        raw_number = int(raw_number)
    elif not isinstance(raw_number, int):
        raise TypeError

    read_number = ""
    # billion-level
    if raw_number // 1e9 > 0:
        read_number += f'{int(raw_number // 1e9)}b'
        raw_number %= 1e9
    # million-level
    if raw_number // 1e6 > 0:
        read_number += f'{int(raw_number // 1e6)}m'
        raw_number %= 1e6
    # kilo-level
    if raw_number // 1e3 > 0:
        read_number += f'{int(raw_number // 1e3)}k'
        raw_number %= 1e3
    # hundred-level
    if raw_number // 1e2 > 0:
        read_number += f'{int(raw_number // 1e2)}h'
        raw_number %= 1e2
    # 1~99
    if raw_number > 0:
        read_number += f"{raw_number:d}"

    return read_number


def parse_readable_number(read_number: str) -> int or float:
    raw_number = 0
    read_number = read_number.replace(' ', '').lower()

    def split_and_record(_raw_number: int, _read_number: str, spliter: str = None):
        if spliter is not None:
            read_number_split = _read_number.split(spliter)
            read_number_split.remove('')
            assert len(read_number_split) <= 2

            addend = int(read_number_split[0])
            if spliter == 'b':
                assert addend > 0
                addend *= 1e9
            elif spliter == 'm':
                assert 0 < addend < 1000
                addend *= 1e6
            elif spliter == 'k':
                assert 0 < addend < 1000
                addend *= 1e3
            elif spliter == 'h':
                assert 0 < addend < 10
                addend *= 1e2
            return _raw_number + int(addend), read_number_split[1] if len(read_number_split) == 2 else ''

        else:
            assert read_number.isdigit()
            addend = int(read_number)
            return _raw_number + addend

    # billion-level
    if 'b' in read_number:
        raw_number, read_number = split_and_record(raw_number, read_number, 'm')
    # million-level
    if 'm' in read_number:
        raw_number, read_number = split_and_record(raw_number, read_number, 'm')
    # kilo-level
    if 'k' in read_number:
        raw_number, read_number = split_and_record(raw_number, read_number, 'k')
    # hundred-level
    if 'h' in read_number:
        raw_number, read_number = split_and_record(raw_number, read_number, 'h')
    # 1~99
    if read_number.isdigit():
        raw_number = split_and_record(raw_number, read_number)
    elif read_number != '':
        raise RuntimeError

    return raw_number


def get_readable_memory(raw_number: int or float) -> str:
    if isinstance(raw_number, float):
        raw_number = int(raw_number)
    elif not isinstance(raw_number, int):
        raise TypeError

    read_number = ""
    # TB-level
    tb_memory = pow(2, 40)
    if raw_number // tb_memory > 0:
        read_number += f'{int(raw_number // tb_memory)}TB '
        raw_number %= tb_memory
    # GB-level
    gb_memory = pow(2, 30)
    if raw_number // gb_memory > 0:
        read_number += f'{int(raw_number // gb_memory)}GB '
        raw_number %= gb_memory
    # MB-level
    mb_memory = pow(2, 20)
    if raw_number // mb_memory > 0:
        read_number += f'{int(raw_number // mb_memory)}MB '
        raw_number %= mb_memory
    # KB-level
    kb_memory = pow(2, 10)
    if raw_number // kb_memory > 0:
        read_number += f'{int(raw_number // kb_memory)}KB '
        raw_number %= kb_memory
    # 1 ~ 1023
    if raw_number > 0:
        read_number += f"{raw_number:d}B"

    return read_number


if __name__ == '__main__':
    en_text_process(
        'Notes of admiration (!), of interrogation (?), of remonstrance, approval, or abuse, come pouring into mr',
        txt_format='punc')