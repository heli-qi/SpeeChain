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

from speechain.utilbox.type_util import str2bool

def parse():
    parser = argparse.ArgumentParser(description='params')
    group = parser.add_argument_group("General arguments shared by all token types")
    group.add_argument('--text_path', type=str, default="../../datasets/speech_text/librispeech/data/raw/train_clean_100/text")
    group.add_argument('--output_path', type=str, default="../../datasets/speech_text/librispeech/data/char/train_clean_100")
    group.add_argument('--token_type', type=str, default="char")
    group.add_argument('--tgt_subsets', type=str, default="train valid test ")

    group = parser.add_argument_group("Specific arguments used by the subword token type")
    # tokenizer training-related
    group.add_argument('--vocab_size', type=int, default=5000)
    group.add_argument('--model_type', type=str, default="bpe")
    group.add_argument('--character_coverage', type=float, default=1.0)
    group.add_argument('--split_by_whitespace', type=str2bool, default=True,
                       help="Control whether there will be spaces inside tokens that are across multiple words."
                            "True means no middle spaces (tokens not beyond words). "
                            "Note: We don't recommend you to set this argument to False because most of time middle "
                            "spaces inside tokens will degrade ASR performance. "
                            "Also, middle spaces will disable lexicon calibration. (default: True)")
    # tokenizer encoding-related
    group.add_argument('--enable_sampling', type=bool, default=False)
    group.add_argument('--alpha', type=float, default=0.0)
    group.add_argument('--nbest_size', type=int, default=-1)

    return parser.parse_args()


def generate_vocab_char(output_path: str, text_path: str, tgt_subsets: List[str]):
    # --- Vocabulary List Generation --- #
    if os.path.exists(os.path.join(output_path, 'vocab')):
        print(f"Char vocabulary list already exists in {output_path}.")
    else:
        # read the one-sentence-per-line file and turn the information into a ndarray
        text = np.loadtxt(text_path, delimiter="\n", dtype=str)

        # collect all chars in the text
        tokens = []
        for sent in text:
            tokens += list(sent)

        # 0 is designed for the blank (the padding index)
        # -2 is designed for the unknowns
        # -1 is designed for the beginning and end of sentence
        token_vocab = ["<blank>"] + sorted(set(tokens)) + ['<unk>', '<sos/eos>']
        np.savetxt(os.path.join(output_path, 'vocab'), token_vocab, fmt="%s")
        print(f"Char vocabulary has been successfully saved to {os.path.join(output_path, 'vocab')}.")


    # --- Tokenized Sentence Length Generation --- #
    subset_path = os.path.dirname(text_path)
    dataset_path = os.path.abspath(os.path.join(subset_path, '..'))
    for file_name in os.listdir(dataset_path):
        # only consider the folders of the target subsets
        if file_name not in tgt_subsets:
            continue

        idx2sent = np.loadtxt(os.path.join(dataset_path, file_name, 'idx2sent'), dtype=str, delimiter="\n")
        idx2sent_len = []
        for line in idx2sent:
            idx, sent = line.split(' ', 1)
            # consider all the tokens of this sentence plus a <sos/eos> (teacher-forcing setting)
            idx2sent_len.append([idx, len([char for char in sent]) + 1])

        os.makedirs(os.path.join(output_path, file_name), exist_ok=True)
        np.savetxt(os.path.join(output_path, file_name, 'idx2sent_len'), idx2sent_len, fmt="%s")
        print(f"Sentence Length of {file_name} calculated by {os.path.join(output_path, 'vocab')} "
              f"has been successfully saved to {os.path.join(output_path, file_name, 'idx2sent_len')}.")


def generate_vocab_subword(output_path: str, text_path: str, tgt_subsets: List[str],
                           vocab_size: int, character_coverage: float,
                           model_type: str, split_by_whitespace: bool,
                           enable_sampling: bool, alpha: float, nbest_size: int):
    # --- Vocabulary List Generation --- #
    if os.path.exists(os.path.join(output_path, 'model')) and os.path.exists(os.path.join(output_path, 'vocab')):
        print(f"Subword tokenizer model and vocabulary list already exist in {output_path}.")
    else:
        # disable bos and eos. <sos>/<eos> will be added externally, so vocab_size need to be subtracted from 1
        # add <blank> and put <unk> to the end of the vocabulary
        spm.SentencePieceTrainer.train(input=text_path, model_prefix='m', vocab_size=vocab_size - 1,
                                       character_coverage=character_coverage, model_type=model_type,
                                       split_by_whitespace=split_by_whitespace, user_defined_symbols='<blank>',
                                       unk_id=vocab_size - 2, bos_id=-1, eos_id=-1, minloglevel=1)
        model_path = os.path.join(os.path.abspath('.'), 'm.model')
        vocab_path = os.path.join(os.path.abspath('.'), 'm.vocab')

        # move the model file to the output_path
        shutil.move(model_path, os.path.join(output_path, 'm.model'))
        os.rename(os.path.join(output_path, 'm.model'), os.path.join(output_path, 'model'))
        print(f"Subword tokenizer model has been successfully saved to {os.path.join(output_path, 'model')}.")

        # modify the vocabulary list and save to the output_path
        token_vocab = np.loadtxt(vocab_path, dtype=str)[:, 0].tolist() + ['<sos/eos>']
        np.savetxt(os.path.join(output_path, 'vocab'), token_vocab, fmt="%s")
        os.remove(vocab_path)
        print(f"Subword vocabulary has been successfully saved to {os.path.join(output_path, 'vocab')}.")


    # --- Tokenized Sentence Length Generation --- #
    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(output_path, 'model'))

    subset_path = os.path.dirname(text_path)
    dataset_path = os.path.abspath(os.path.join(subset_path, '..'))
    for file_name in os.listdir(dataset_path):
        # only consider the folders of the target subsets
        if file_name not in tgt_subsets:
            continue

        idx2sent = np.loadtxt(os.path.join(dataset_path, file_name, 'idx2sent'), dtype=str, delimiter="\n")
        idx2sent_len = []
        for line in idx2sent:
            idx, sent = line.split(' ', 1)
            tokens = sp.encode_as_ids(sent, enable_sampling=enable_sampling, alpha=alpha, nbest_size=nbest_size)
            # consider all the tokens of this sentence plus a <sos/eos> (teacher-forcing setting)
            idx2sent_len.append([idx, len(tokens) + 1])

        os.makedirs(os.path.join(output_path, file_name), exist_ok=True)
        np.savetxt(os.path.join(output_path, file_name, 'idx2sent_len'), idx2sent_len, fmt="%s")
        print(f"Sentence Length of {file_name} calculated by {os.path.join(output_path, 'model')} "
              f"has been successfully saved to {os.path.join(output_path, file_name, 'idx2sent_len')}.")


def generate_vocab_word(output_path: str, text_path: str, tgt_subsets: List[str]):
    """
    This function just segments sentences with whitespaces, so the punctuation symbols won't be independent tokens.

    """
    # --- Vocabulary List Generation --- #
    if os.path.exists(os.path.join(output_path, 'vocab')):
        print(f"Word vocabulary list already exist in {output_path}.")
    else:
        # read the one-sentence-per-line file and turn the information into a ndarray
        text = np.loadtxt(text_path, delimiter="\n", dtype=str)

        # collect all words in the text
        tokens = []
        for sent in text:
            tokens += sent.split()

        # 0 is designed for the blank (the padding index)
        # -2 is designed for the unknowns
        # -1 is designed for the beginning and end of sentence
        token_vocab = ["<blank>"] + sorted(set(tokens)) + ['<unk>', '<sos/eos>']
        np.savetxt(os.path.join(output_path, 'vocab'), token_vocab, fmt="%s")
        print(f"Word vocabulary has been successfully saved to {os.path.join(output_path, 'vocab')}.")


    # --- Tokenized Sentence Length Generation --- #
    subset_path = os.path.dirname(text_path)
    dataset_path = os.path.abspath(os.path.join(subset_path, '..'))
    for file_name in os.listdir(dataset_path):
        # only consider the folders of the target subsets
        if file_name not in tgt_subsets:
            continue

        idx2sent = np.loadtxt(os.path.join(dataset_path, file_name, 'idx2sent'), dtype=str, delimiter="\n")
        idx2sent_len = []
        for line in idx2sent:
            idx, sent = line.split(' ', 1)
            # consider all the tokens of this sentence plus a <sos/eos> (teacher-forcing setting)
            idx2sent_len.append([idx, len(sent.split()) + 1])

        os.makedirs(os.path.join(output_path, file_name), exist_ok=True)
        np.savetxt(os.path.join(output_path, file_name, 'idx2sent_len'), idx2sent_len, fmt="%s")
        print(f"Sentence Length of {file_name} calculated by {os.path.join(output_path, 'vocab')} "
              f"has been successfully saved to {os.path.join(output_path, file_name, 'idx2sent_len')}.")


def get_readable_number(number: int):
    output = ""
    if number // 1e6 > 0:
        output += f'{int(number // 1e6)}m'
        number %= 1e6

    if number // 1e3 > 0:
        output += f'{int(number // 1e3)}k'
        number %= 1e3

    if number // 1e2 > 0:
        output += f'{int(number // 1e2)}h'

    return output


def main(args):
    os.makedirs(args.output_path, exist_ok=True)
    # here cannot be .split(" ") or .split(' ') because there will be an extra space at the end of args.tgt_subsets
    # giving space as the argument of .split will cause an extra '' at the end of tgt_subsets
    tgt_subsets = args.tgt_subsets.split()

    # character token branch
    if args.token_type.lower() == 'char':
        # generate character vocabulary list
        generate_vocab_char(args.output_path, args.text_path, tgt_subsets)

    # subword token branch
    elif args.token_type.lower() == 'subword':
        # update the output path and create the folder
        output_path = os.path.join(args.output_path, f"{args.model_type}{get_readable_number(args.vocab_size)}")
        os.makedirs(output_path, exist_ok=True)

        # generate subword vocabulary list
        generate_vocab_subword(output_path=output_path, text_path=args.text_path, tgt_subsets=tgt_subsets,
                               vocab_size=args.vocab_size, character_coverage=args.character_coverage,
                               model_type=args.model_type, split_by_whitespace=args.split_by_whitespace,
                               enable_sampling=args.enable_sampling, alpha=args.alpha, nbest_size=args.nbest_size)

    # word token branch
    elif args.token_type.lower() == 'word':
        generate_vocab_word(args.output_path, args.text_path, tgt_subsets)

    # unknown token branch
    else:
        raise ValueError(f"token_type must be one of 'char', 'subword', and 'word', but got {args.token_type}!")


if __name__ == '__main__':
    main(parse())
