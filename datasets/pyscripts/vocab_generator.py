"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import argparse

import numpy as np
import os
import sentencepiece as spm
import shutil

from typing import List
from collections import Counter
from functools import partial

from speechain.utilbox.type_util import str2bool
from speechain.utilbox.dump_util import get_readable_number
from speechain.tokenizer.subword import SubwordTokenizer


def parse():
    parser = argparse.ArgumentParser(description='params')
    group = parser.add_argument_group("General arguments shared by all token types")
    group.add_argument('--text_path', type=str, required=True,
                       help="The path of the text data you want to use to get the tokenizer.")
    group.add_argument('--save_path', type=str, required=True,
                       help="The path where you want to save the vocabulary and tokenizer model.")
    group.add_argument('--token_type', type=str, required=True,
                       help="The type of the token you want to use in your tokenizer.")
    group.add_argument('--txt_format', type=str, default='normal',
                       help="The text processing format controlling how to process the transcript sentence of each "
                            "utterance before saving them into 'idx2sent' and 'text'.")
    group.add_argument('--vocab_size', type=int, default=None,
                       help="The number of tokens in the vocabulary of the tokenizer. (default: None)")
    group.add_argument('--tgt_subsets', type=str, required=True,
                       help="The target subsets that you want to calculate the sentence length by the tokenizer.")
    parser.add_argument('--ncpu', type=int, default=8,
                        help="The number of threads you want to use to collect the statistical information of the dataset.")

    group = parser.add_argument_group("Specific arguments used by the subword token type")
    group.add_argument('--subword_package', type=str, default="sp",
                       help="The package you want to use to parse your subword tokenizer model. (default: sp)")
    group.add_argument('--subword_type', type=str, default="bpe",
                       help="The type of subword for the tokenizer model. "
                            "Could be either 'bpe' or 'unigram'. (default: bpe)")
    group.add_argument('--character_coverage', type=float, default=1.0,
                       help="The coverage rate of characters in the subword tokenizer model. "
                            "We recommend you to set this number to 1.0 for the languages that have a small amount of "
                            "graphemes like English, "
                            "and set it to 0.95 for the languages that have a large amount of graphemes like Chinese. "
                            "(default: 1.0)")
    group.add_argument('--split_by_whitespace', type=str2bool, default=True,
                       help="Control whether there will be spaces inside tokens that are across multiple words."
                            "True means no middle spaces (tokens not beyond words). "
                            "Note: We don't recommend you to set this argument to False because most of time middle "
                            "spaces inside tokens will degrade ASR performance. "
                            "Also, middle spaces will disable lexicon calibration. (default: True)")

    return parser.parse_args()


def save_token_vocab(save_path: str, text_path: str, txt_format: str, text2tokens_func, vocab_size: int):
    """
    Obtain and save the token vocabulary for char and word tokenizers.
    The tokens in the vocabulary are sorted in descending order by their occurrence frequency in the text data.

    Args:
        save_path: str
            Where to save the token vocabulary
        text_path: str
            Where the text data used to get the vocabulary is placed
        txt_format: str

        text2tokens_func: function
            The function that transforms a sentence string into a list of tokens
        vocab_size: int
            The maximum number of tokens in the vocabulary.
            Useless if vocab_size is larger than the actual token number.

    Returns:
        The modified save_path where the tokenizer configuration is attached at the end

    """
    # read the one-sentence-per-line file and turn the information into a ndarray
    text = np.loadtxt(os.path.join(text_path, f"{txt_format}_text"), delimiter="\n", dtype=str)
    # collect all tokens in the text
    tokens = []
    for sent in text:
        tokens += text2tokens_func(sent)

    # collect the occurrence frequency of each token
    token2freq = dict(sorted(Counter(tokens).items(), key=lambda x: x[1], reverse=True))
    tokens = list(token2freq.keys())
    # change the token list and saving path only when the number of tokens is larger than vocab_size
    if vocab_size is not None and len(tokens) >= vocab_size - 3:
        tokens = tokens[: vocab_size - 3]
        save_path = os.path.join(save_path, get_readable_number(vocab_size), txt_format)
    else:
        save_path = os.path.join(save_path, 'full_tokens', txt_format)
    os.makedirs(save_path, exist_ok=True)

    # 0 is designed for the blank (the padding index)
    # -2 is designed for the unknowns
    # -1 is designed for the beginning and end of sentence
    token_vocab = ["<blank>"] + tokens + ['<unk>', '<sos/eos>']
    vocab_path = os.path.join(save_path, 'vocab')
    np.savetxt(vocab_path, token_vocab, fmt="%s")
    print(f"Token vocabulary has been successfully saved to {vocab_path}.")

    return save_path


def generate_text_len(save_path: str, text_path: str, txt_format: str, text2len_func,
                      tgt_subsets: List[str], ncpu: int):
    """

    Args:
        save_path: str
        text_path: str
        txt_format: str
        text2len_func: function
        tgt_subsets: List[str]
        ncpu: int

    Returns:

    """
    # the root path of the dataset (parent directory of subset_path)
    dataset_path = os.path.abspath(os.path.join(text_path, '../../speechain'))

    # only consider the target subsets
    for subset_name in tgt_subsets:
        # np.loadtxt may cause TypeError related to the '\n' delimiter
        # -----------------------------------------------------------------------------------------
        # idx2text = np.loadtxt(os.path.join(dataset_path, subset_name, f'idx2{txt_format}_text'),
        #                       dtype=str, delimiter="\n").tolist()
        # -----------------------------------------------------------------------------------------
        with open(os.path.join(dataset_path, subset_name, f'idx2{txt_format}_text'), mode='r') as f:
            idx2text = f.readlines()
            idx2text = [line.replace('\n', '') for line in idx2text]
        # since text2len_func is a lambda function that cannot be pickled,
        # multiprocessing.Pool cannot be used here to speed up the text_len generation
        idx2text_len = [[line.split(' ', 1)[0], text2len_func(line.split(' ', 1)[1])] for line in idx2text]

        # update subset_path to the place in save_path
        subset_path = os.path.join(save_path, subset_name)
        os.makedirs(subset_path, exist_ok=True)
        np.savetxt(os.path.join(subset_path, f'idx2text_len'), idx2text_len, fmt="%s")

    print(f"Text Length of {tgt_subsets} has been successfully saved to the 'idx2text_len' files in {save_path}.")


def generate_vocab_char(save_path: str, text_path: str, txt_format: str, vocab_size: int,
                        tgt_subsets: List[str], ncpu: int):
    # --- Vocabulary List Generation --- #
    save_path = save_token_vocab(
        save_path=save_path, text_path=text_path, txt_format=txt_format,
        text2tokens_func=lambda x: list(x), vocab_size=vocab_size
    )

    # --- Tokenized Text Length Generation --- #
    generate_text_len(save_path=save_path, text_path=text_path, txt_format=txt_format, tgt_subsets=tgt_subsets,
                      # consider all the tokens of this text plus a <sos/eos> (teacher-forcing setting)
                      text2len_func=lambda x: len(list(x)) + 1, ncpu=ncpu)


def generate_vocab_subword(save_path: str, text_path: str, tgt_subsets: List[str], ncpu: int,
                           txt_format: str, vocab_size: int, subword_package: str, subword_type: str,
                           character_coverage: float, split_by_whitespace: bool):
    # sentencepiece package
    if subword_package == 'sp':
        # --- Argument Initialization --- #
        subword_type = subword_type.lower()
        assert subword_type in ['unigram', 'bpe'], \
            "model_type must be either 'unigram' or 'bpe' if you want to use subword tokenizer."
        # update the output path and create the folder
        save_path = os.path.join(save_path, f"{subword_type}{get_readable_number(vocab_size)}", txt_format)
        os.makedirs(save_path, exist_ok=True)

        # --- Vocabulary List Generation --- #
        # the path of the subset of the given text (parent directory of text_path)
        subset_path = os.path.abspath(os.path.join(text_path, '../../speechain'))
        # skip if 'model' and 'vocab' exist at the same time
        if os.path.exists(os.path.join(save_path, 'model')) and os.path.exists(os.path.join(save_path, 'vocab')):
            print(f"Subword tokenizer model and vocabulary list already exist in {save_path}.")
            model_path = os.path.join(os.path.join(save_path, 'model'))
            vocab_path = os.path.join(save_path, 'vocab')
        else:
            # disable bos and eos. <sos>/<eos> will be added externally, so vocab_size need to be subtracted from 1
            # add <blank> and put <unk> to the end of the vocabulary
            spm.SentencePieceTrainer.train(input=os.path.join(text_path, f'{txt_format}_text'), model_prefix='m',
                                           vocab_size=vocab_size - 1, model_type=subword_type,
                                           character_coverage=character_coverage,
                                           split_by_whitespace=split_by_whitespace, user_defined_symbols='<blank>',
                                           unk_id=vocab_size - 2, bos_id=-1, eos_id=-1, minloglevel=1)
            model_path = os.path.join(os.path.abspath('../../speechain/utilbox'), 'm.model')
            vocab_path = os.path.join(os.path.abspath('../../speechain/utilbox'), 'm.vocab')

            # move the model file to the output_path
            shutil.move(model_path, os.path.join(save_path, 'm.model'))
            # update model_path after the original m.model is renamed
            model_path = os.path.join(os.path.join(save_path, 'model'))
            os.rename(os.path.join(save_path, 'm.model'), model_path)
            print(f"Subword tokenizer model has been successfully saved to {model_path}.")

            # modify the vocabulary list and save to the output_path
            token_vocab = np.loadtxt(vocab_path, dtype=str)[:, 0].tolist() + ['<sos/eos>']
            os.remove(vocab_path)
            # update vocab_path after the original m.vocab is removed
            vocab_path = os.path.join(save_path, 'vocab')
            np.savetxt(vocab_path, token_vocab, fmt="%s")
            print(f"Subword vocabulary has been successfully saved to {vocab_path}.")

        # initialize the tokenizer by the vocab and model obtained above
        tokenizer = SubwordTokenizer(token_vocab=vocab_path, token_model=model_path, model_package='sp')
        # decode the text without sos at the beginning and eos at the end
        text2tensor_func = partial(tokenizer.text2tensor, no_sos=True, no_eos=True)

    else:
        raise NotImplementedError("The tokenizer packages other than sentencepiece have not been implemented yet~~~")

    # --- Tokenized Sentence Length Generation --- #
    generate_text_len(
        save_path=save_path, text_path=text_path, txt_format=txt_format, tgt_subsets=tgt_subsets,
        # consider all the tokens of this text plus a <sos/eos> (teacher-forcing setting)
        text2len_func=lambda x: len(text2tensor_func(x)) + 1, ncpu=ncpu
    )


def generate_vocab_word(save_path: str, text_path: str, txt_format: str, vocab_size: int,
                        tgt_subsets: List[str], ncpu: int):
    """
    This function just segments text with whitespaces, so the punctuation symbols won't be treated as independent tokens.

    """
    # --- Vocabulary List Generation --- #
    save_path = save_token_vocab(
        save_path=save_path, text_path=text_path, txt_format=txt_format,
        text2tokens_func=lambda x: x.split(), vocab_size=vocab_size
    )

    # --- Tokenized Text Length Generation --- #
    generate_text_len(save_path=save_path, text_path=text_path, txt_format=txt_format, tgt_subsets=tgt_subsets,
                      # consider all the tokens of this sentence plus a <sos/eos> (teacher-forcing setting)
                      text2len_func=lambda x: len(x.split()) + 1, ncpu=ncpu)


def main(text_path: str, save_path: str, token_type: str, txt_format: str, vocab_size: int, tgt_subsets: str, ncpu: int,
         subword_package: str, subword_type: str, character_coverage: float, split_by_whitespace: bool):
    # --- Arguments Initialization --- #
    # here cannot be .split(" ") or .split(' ') because there will be an extra space at the end of args.tgt_subsets
    # giving space as the argument of .split will cause an extra '' at the end of tgt_subsets
    tgt_subsets = tgt_subsets.split()

    # --- Vocabulary Generation of Different Token Types --- #
    # character token branch
    if token_type.lower() == 'char':
        # generate character vocabulary list
        generate_vocab_char(save_path=save_path, text_path=text_path,
                            txt_format=txt_format, vocab_size=vocab_size, tgt_subsets=tgt_subsets, ncpu=ncpu)

    # subword token branch
    elif token_type.lower() == 'subword':
        # generate subword vocabulary list
        generate_vocab_subword(
            save_path=save_path, text_path=text_path, tgt_subsets=tgt_subsets, ncpu=ncpu,
            txt_format=txt_format, vocab_size=vocab_size, subword_package=subword_package, subword_type=subword_type,
            character_coverage=character_coverage, split_by_whitespace=split_by_whitespace
        )

    # word token branch
    elif token_type.lower() == 'word':
        generate_vocab_word(save_path=save_path, text_path=text_path,
                            txt_format=txt_format, vocab_size=vocab_size, tgt_subsets=tgt_subsets, ncpu=ncpu)

    # unknown token branch
    else:
        raise ValueError(f"token_type must be one of 'char', 'subword', and 'word', but got {token_type}!")


if __name__ == '__main__':
    args = parse()
    main(**vars(args))
