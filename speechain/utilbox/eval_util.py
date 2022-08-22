import edit_distance
from speechain.utilbox.md_util import get_table_strings


def get_word_edit_alignment(hypo: str, real: str) -> (int, int, int, str):
    """

    Args:
        hypo:
        real:

    Returns:

    """
    # calculate the alignment between the hypothesis words and real words
    # Note that split(" ") is not equivalent to split() here
    # because split(" ") will give an extra '' at the end of the list if the string ends with a " "
    # while split() doesn't
    hypo_word_list, real_word_list = hypo.split(), real.split()
    opcodes = edit_distance.SequenceMatcher(a=hypo_word_list, b=real_word_list).get_opcodes()

    insertion, deletion, substitution = 0, 0, 0
    hypo_words, real_words, word_ops = [], [], []
    # loop each editing operation
    for i, op in enumerate(opcodes):
        # insertion operation
        if op[0] == 'insert':
            insertion += 1
            word_ops.append('I')
            hypo_words.append(' ')
            real_words.append(real_word_list[op[3]])
        # deletion operation
        elif op[0] == 'delete':
            deletion += 1
            word_ops.append('D')
            hypo_words.append(hypo_word_list[op[1]])
            real_words.append(' ')
        # substitution operation
        elif op[0] == 'replace':
            substitution += 1
            word_ops.append('S')
            hypo_words.append(hypo_word_list[op[1]])
            real_words.append(real_word_list[op[3]])
        # equal condition
        else:
            word_ops.append(' ')
            hypo_words.append(hypo_word_list[op[1]])
            real_words.append(real_word_list[op[3]])

    align_table = get_table_strings(contents=[hypo_words, word_ops, real_words],
                                    first_col=['Hypothesis', 'Alignment', 'Reference'])

    return insertion, deletion, substitution, align_table
