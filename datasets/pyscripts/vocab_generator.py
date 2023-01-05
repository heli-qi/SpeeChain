"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import os
import shutil
import argparse

import numpy as np
import sentencepiece as spm

from g2p_en import G2p
from collections import Counter

from speechain.utilbox.type_util import str2bool
from speechain.utilbox.dump_util import get_readable_number
from speechain.utilbox.data_loading_util import load_idx2data_file, parse_path_args


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

    group = parser.add_argument_group("Specific arguments used by the sentencepiece token type")
    group.add_argument('--model_type', type=str, default="bpe",
                       help="The type of the sentencepiece tokenizer model. "
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


def save_token_vocab(save_path: str, text_path: str, txt_format: str, text2tokens_func, vocab_size: int,
                     save_idx2text: bool = False):
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
        save_idx2text: bool = False

    Returns:
        The modified save_path where the tokenizer configuration is attached at the end

    """
    # --- 1. Data Initialization --- #
    # read the index-to-text file and turn the information into a Dict
    idx2text = load_idx2data_file(os.path.join(text_path, f"idx2{txt_format}_text"))
    # convert each string sentence into its token sentence
    idx2text_token = {idx: text2tokens_func(text) for idx, text in idx2text.items()}
    if vocab_size is not None:
        save_path = os.path.join(save_path, get_readable_number(vocab_size), txt_format)
    else:
        save_path = os.path.join(save_path, 'full_tokens', txt_format)
    os.makedirs(save_path, exist_ok=True)

    # --- 2. Token Vocabulary Saving --- #
    # gather the tokens of all the sentences into a single list
    tokens = []
    for value in idx2text_token.values():
        tokens += value
    # collect the occurrence frequency of each token
    token2freq = sorted(Counter(tokens).items(), key=lambda x: x[1], reverse=True)
    token_vocab = [token for token, _ in token2freq]
    # change the token list and saving path only when the number of token_vocab is larger than vocab_size
    if vocab_size is not None and len(token_vocab) >= vocab_size - 3:
        token_vocab = token_vocab[: vocab_size - 3]

    # 0 is designed for the blank (the padding index)
    # -2 is designed for the unknowns
    # -1 is designed for the beginning and end of sentence
    token_vocab = ["<blank>"] + token_vocab + ['<unk>', '<sos/eos>']
    vocab_path = os.path.join(save_path, 'vocab')
    np.savetxt(vocab_path, token_vocab, fmt="%s")
    print(f"Token vocabulary has been successfully saved to {vocab_path}.")

    # --- 3. Tokenized Text and its Length Saving --- #
    if save_idx2text:
        text_token_path = os.path.join(save_path, 'idx2text')
        np.savetxt(text_token_path, [[idx, str(text_token)] for idx, text_token in idx2text_token.items()], fmt="%s")
        print(f"Tokenized text has been successfully saved to {text_token_path}.")

        text_len_path = os.path.join(save_path, 'idx2text_len')
        np.savetxt(text_len_path, [[idx, len(text_token)] for idx, text_token in idx2text_token.items()], fmt="%s")
        print(f"The length of tokenized text has been successfully saved to {text_len_path}.")


def generate_vocab_char(save_path: str, text_path: str, txt_format: str, vocab_size: int):
    # --- Vocabulary List Generation --- #
    save_token_vocab(
        save_path=save_path, text_path=text_path, txt_format=txt_format,
        text2tokens_func=lambda x: list(x), vocab_size=vocab_size, save_idx2text=True
    )


def generate_vocab_sentencepiece(save_path: str, text_path: str, txt_format: str, vocab_size: int,
                                 model_type: str, character_coverage: float, split_by_whitespace: bool):
    # --- Argument Initialization --- #
    model_type = model_type.lower()
    assert model_type in ['unigram', 'bpe'], \
        "model_type must be either 'unigram' or 'bpe' if you want to construct a SentencePiece tokenizer."
    # update the output path and create the folder
    save_path = os.path.join(save_path, f"{model_type}{get_readable_number(vocab_size)}", txt_format)
    os.makedirs(save_path, exist_ok=True)

    # --- Vocabulary List Generation --- #
    # skip if 'model' and 'vocab' exist at the same time
    if not os.path.exists(os.path.join(save_path, 'model')) or not os.path.exists(os.path.join(save_path, 'vocab')):
        # disable bos and eos. <sos>/<eos> will be added externally, so vocab_size need to be subtracted from 1
        # add <blank> and put <unk> to the end of the vocabulary
        spm.SentencePieceTrainer.train(input=os.path.join(text_path, f'{txt_format}_text'), model_prefix='m',
                                       vocab_size=vocab_size - 1, model_type=model_type,
                                       character_coverage=character_coverage,
                                       split_by_whitespace=split_by_whitespace, user_defined_symbols='<blank>',
                                       unk_id=vocab_size - 2, bos_id=-1, eos_id=-1, minloglevel=1)
        model_path = os.path.abspath('./m.model')
        vocab_path = os.path.abspath('./m.vocab')

        # move the model file to the output_path
        shutil.move(model_path, os.path.join(save_path, 'm.model'))
        # update model_path after the original m.model is renamed
        model_path = os.path.join(os.path.join(save_path, 'model'))
        os.rename(os.path.join(save_path, 'm.model'), model_path)
        print(f"SentencePiece tokenizer model has been successfully saved to {model_path}.")

        # modify the vocabulary list and save to the output_path
        token_vocab = np.loadtxt(vocab_path, dtype=str)[:, 0].tolist() + ['<sos/eos>']
        os.remove(vocab_path)
        # update vocab_path after the original m.vocab is removed
        vocab_path = os.path.join(save_path, 'vocab')
        np.savetxt(vocab_path, token_vocab, fmt="%s")
        print(f"SentencePiece vocabulary has been successfully saved to {vocab_path}.")


def generate_vocab_g2p(save_path: str, text_path: str, txt_format: str, vocab_size: int):
    g2p = G2p()
    # --- Vocabulary List Generation --- #
    save_token_vocab(
        save_path=save_path, text_path=text_path, txt_format=txt_format,
        text2tokens_func=lambda x: g2p(x), vocab_size=vocab_size, save_idx2text=True
    )


def generate_vocab_word(save_path: str, text_path: str, txt_format: str, vocab_size: int):
    """
    This function just segments text with whitespaces, so the punctuation symbols won't be treated as independent tokens.

    """
    # --- Vocabulary List Generation --- #
    save_token_vocab(
        save_path=save_path, text_path=text_path, txt_format=txt_format,
        text2tokens_func=lambda x: x.split(), vocab_size=vocab_size, save_idx2text=False
    )


def main(text_path: str, save_path: str, token_type: str, txt_format: str, vocab_size: int,
         model_type: str, character_coverage: float, split_by_whitespace: bool):
    # --- Arguments Initialization --- #
    text_path, save_path = parse_path_args(text_path), parse_path_args(save_path)

    # --- Vocabulary Generation of Different Token Types --- #
    # character token branch
    if token_type.lower() == 'char':
        # generate character vocabulary list
        generate_vocab_char(save_path=save_path, text_path=text_path, txt_format=txt_format, vocab_size=vocab_size)

    # sentencepiece token branch
    elif token_type.lower() == 'sentencepiece':
        # generate sentencepiece vocabulary list
        generate_vocab_sentencepiece(
            save_path=save_path, text_path=text_path,
            txt_format=txt_format, vocab_size=vocab_size, model_type=model_type,
            character_coverage=character_coverage, split_by_whitespace=split_by_whitespace
        )

    # g2p_en token branch
    elif token_type.lower() == 'g2p':
        generate_vocab_g2p(save_path=save_path, text_path=text_path, txt_format=txt_format, vocab_size=vocab_size)

    # word token branch
    elif token_type.lower() == 'word':
        generate_vocab_word(save_path=save_path, text_path=text_path, txt_format=txt_format, vocab_size=vocab_size)

    # unknown token branch
    else:
        raise ValueError(f"token_type must be one of ['char', 'sentencepiece', 'g2p', 'word'], but got {token_type}!")


if __name__ == '__main__':
    args = parse()
    main(**vars(args))
