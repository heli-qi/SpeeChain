"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.11
"""


def tts_text_process(input_text: str, txt_format: str) -> str:
    """
    The function that processes the text strings for TTS datasets to the specified text format.
    Currently, available text formats:
        normal:
            Letter: capital and lowercase
            Punctuation: single quotes, commas, periods, hyphens, parentheses, question and exclamation marks
        lowercase:
            Letter: lowercase
            Punctuation: single quotes, commas, periods, hyphens, parentheses, question and exclamation marks
        plain:
            Letter: lowercase
            Punctuation: single quotes, commas, periods, hyphens
        librispeech:
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
    # pre-normalization shared by all the text-processing formats
    # normalize abnormal non-English symbols into English letters
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
    # convert all the quotes into half-angle single quotes
    input_text = input_text.replace('‘', '\'')
    input_text = input_text.replace('“', '\'')
    input_text = input_text.replace('”', '\'')
    input_text = input_text.replace('"', '\'')
    # convert colons and semicolons into commas
    input_text = input_text.replace(':', ',')
    input_text = input_text.replace(';', ',')
    # remove double-hyphens and em dashes
    input_text = input_text.replace(' --', ',')
    input_text = input_text.replace('--', ' ')
    input_text = input_text.replace('—', '-')
    # retain only parentheses
    input_text = input_text.replace('{', '(')
    input_text = input_text.replace('}', ')')
    input_text = input_text.replace('[', '(')
    input_text = input_text.replace(']', ')')
    # remove useless symbols
    input_text = input_text.replace('¯', '')
    input_text = input_text.replace('/', ' ')

    # remain the original string format of LibriTTS
    if txt_format == 'normal':
        return input_text

    # text format without case distinction, may hurt a little of prosody
    elif txt_format == 'lowercase':
        # turn capital letters into their lower cases
        input_text = input_text.lower()

    # plain text format without too much emotion
    elif txt_format == 'plain':
        # convert all capital letters to their lowercase
        input_text = input_text.lower()
        # create more periods at the ends of sentences
        input_text = input_text.replace('?', '.')
        input_text = input_text.replace('!', '.')
        # remove all the punctuation symbols other than commas, periods, hyphens, and single quotes
        input_text = ''.join([char for char in input_text if char.isalpha() or char in [',', '.', '-', '\'', ' ']])

    # process the text string into the same format as the LibriSpeech corpus
    elif txt_format == 'librispeech':
        # convert all lowercase letters to their capitals
        input_text = input_text.lower()
        # remove all the punctuation symbols other than single quotes
        input_text = ''.join([char for char in input_text if char.isalpha() or char in ["\'", ' ']])

    else:
        raise ValueError

    return input_text


def asr_text_process(input_text: str, txt_format: str) -> str:
    """
    The function that processes the text strings for ASR datasets to the specified text format.
    Currently, available text formats:
        normal (LibriSpeech style):
            Letter: lowercase
            Punctuation: single quotes
        capital: Not implemented yet
            Letter: capital and lowercase
            Punctuation: single quotes
        punctuated: Not implemented yet
            Letter: capital and lowercase
            Punctuation: single quotes, commas, periods, hyphens

    Args:
        input_text: str
            Unprocessed raw sentence from the TTS datasets
        txt_format: str
            The text format you want the processed sentence to have

    Returns:
        Processed sentence string by your specified text format.

    """
    if txt_format == 'normal':
        # turn all the capital letters into their lower cases for better readability
        input_text = input_text.lower()
        # remove all the punctuation symbols other than single quotes
        return ''.join([char for char in input_text if char.isalpha() or char in ["\'", ' ']])

    if txt_format == 'capital':
        raise NotImplementedError

    elif txt_format == 'punctuated':
        raise NotImplementedError

    else:
        raise ValueError


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
