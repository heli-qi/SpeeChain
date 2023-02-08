"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.11
"""


def en_text_process(input_text: str, txt_format: str) -> str:
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
    # turn capital letters into their lower cases
    input_text = input_text.lower()

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
    '’'
    input_text = input_text.replace('’', '\'')
    input_text = input_text.replace('‘', '\'')
    input_text = input_text.replace('“', '\'')
    input_text = input_text.replace('”', '\'')
    input_text = input_text.replace('"', '\'')
    input_text = input_text.replace(':\'', ',')
    input_text = input_text.replace('\'\'', '\'')
    _input_text_tmp = []
    for idx, char in enumerate(input_text):
        # save all the non-quotation characters
        if char != '\'':
            _input_text_tmp.append(char)
        # remove the quotations at the beginning or end
        elif idx == 0 or idx == len(input_text) - 1:
            continue
        # remove the quotations not surrounded by letters
        elif not input_text[idx - 1].isalpha() or not input_text[idx + 1].isalpha():
            if input_text[idx - 1] == ' ' or input_text[idx + 1] == ' ':
                continue
            else:
                _input_text_tmp.append(' ')
        # save the intra-word quotations
        else:
            _input_text_tmp.append(char)
    input_text = ''.join(_input_text_tmp)

    # convert standalone colons and semicolons into commas or periods
    input_text = input_text.replace(':', ',')
    input_text = input_text.replace(';', '.')

    # remove double-hyphens and em dashes
    input_text = input_text.replace(' -- ', ' ')
    input_text = input_text.replace(' --', ',')
    input_text = input_text.replace(' - ', ' ')
    input_text = input_text.replace(' -', ',')
    input_text = input_text.replace('--', ' ')
    input_text = input_text.replace('—', '-')
    input_text = input_text.replace('-', '')

    # remove all kinds of parentheses
    input_text = input_text.replace('{', '')
    input_text = input_text.replace('}', '')
    input_text = input_text.replace('[', '')
    input_text = input_text.replace(']', '')
    input_text = input_text.replace('(', '')
    input_text = input_text.replace(')', '')

    # exclamation
    input_text = input_text.replace('!!!', '!')
    input_text = input_text.replace('!!', '!')

    # remove useless symbols
    input_text = input_text.replace('¯', '')
    input_text = input_text.replace('/', ' ')

    # remain the original string format of LibriTTS
    if txt_format == 'tts':
        return input_text

    # process the text string into the same format as the LibriSpeech corpus
    elif txt_format == 'asr':
        # remove all the punctuation symbols other than single quotes
        return ''.join([char for char in input_text if char.isalpha() or char in ["\'", ' ']])

    else:
        raise ValueError(f"txt_format must be one of 'tts' or 'asr'. But got {txt_format}!")



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
        "\"'What right have you, any more than the rest, to ask for an exception?'--'It is true.'--'But never mind,' continued Cucumetto, laughing, 'sooner or later your turn will come.' Carlini's teeth clinched convulsively.", 'txt')