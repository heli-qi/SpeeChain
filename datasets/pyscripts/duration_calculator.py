"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2023.02
"""
import argparse
import os
import textgrid as tg
import numpy as np

from decimal import Decimal
from tqdm import tqdm
from collections import Counter

from speechain.utilbox.import_util import parse_path_args
from speechain.utilbox.data_loading_util import load_idx2data_file, search_file_in_subfolder


def parse():
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--textgrid_path', type=str, required=True,
                        help="The path of the text data you want to use to get the tokenizer.")
    parser.add_argument('--save_path', type=str, default=None,
                        help="The path where you want to save the vocabulary and tokenizer model.")
    parser.add_argument('--proc_dataset', type=str, required=True,
                        help="The type of the token you want to use in your tokenizer.")
    return parser.parse_args()


def main(textgrid_path: str, save_path: str, proc_dataset: str):
    textgrid_path = parse_path_args(textgrid_path)
    if save_path is None:
        save_path = os.path.dirname(textgrid_path)

    proc_dataset = proc_dataset.lower()
    assert proc_dataset in ['ljspeech', 'libritts']

    print(f'Start to summarize all the .TextGrid files in {textgrid_path}......')
    tg_file_list = search_file_in_subfolder(textgrid_path, tgt_match_fn=lambda x: x.endswith('.TextGrid'))
    idx2text, idx2duration = {}, {}
    for tg_file in tqdm(tg_file_list):
        file_name = tg_file.split('/')[-1].split('.')[0]
        idx2text[file_name], idx2duration[file_name] = [], []

        text_grid = tg.TextGrid.fromFile(tg_file)
        for ele in text_grid.tiers[1].intervals:
            # ensure the accuracy of the float numbers for data storage
            idx2duration[file_name].append(float(Decimal(str(ele.maxTime)) - Decimal(str(ele.minTime))))
            if ele.mark not in ['sp', 'spn', '', 'sil']:
                idx2text[file_name].append(ele.mark)
            else:
                idx2text[file_name].append('<space>')

    proc_dataset_path = parse_path_args(f'datasets/{proc_dataset}/data/wav')
    for i in os.listdir(proc_dataset_path):
        if proc_dataset == 'libritts' and \
                i not in ['dev', 'dev-clean', 'dev-other', 'test-clean', 'test-other', 'train-clean-100',
                          'train-clean-360', 'train-clean-460', 'train-other-500', 'train-960']:
            continue

        if proc_dataset == 'ljspeech' and i not in ['train', 'valid', 'test']:
            continue

        # get the subset-specific idx2text and idx2duration Dicts
        subset_indices = list(load_idx2data_file(os.path.join(proc_dataset_path, i, 'idx2wav')).keys())
        subset_idx2text = {index: idx2text[index] for index in subset_indices if index in idx2text.keys()}
        subset_idx2duration = {index: idx2duration[index] for index in subset_indices if index in idx2duration.keys()}

        subset_path = os.path.join(save_path, i)
        os.makedirs(subset_path, exist_ok=True)
        # save the subet-specific idx2text Dict to a metadata file
        text_path = os.path.join(subset_path, 'idx2text')
        np.savetxt(text_path, [[idx, str(text)] for idx, text in subset_idx2text.items()], fmt="%s")
        print(f"Tokenized text has been successfully saved to {text_path}.")

        # save the length information of the subet-specific idx2text Dict to a metadata file
        text_len_path = os.path.join(subset_path, 'idx2text_len')
        np.savetxt(text_len_path, [[idx, len(text)] for idx, text in subset_idx2text.items()], fmt="%s")
        print(f"The length of tokenized text has been successfully saved to {text_len_path}.")

        # save the subet-specific idx2duration Dict to a metadata file
        duration_path = os.path.join(subset_path, 'idx2duration')
        np.savetxt(duration_path, [[idx, str(duration)] for idx, duration in subset_idx2duration.items()], fmt="%s")
        print(f"The duration of tokenized text has been successfully saved to {duration_path}.")

        subset_phns = []
        for text in subset_idx2text.values():
            subset_phns += text
        # collect the occurrence frequency of each phoneme
        phn2freq = sorted(Counter(subset_phns).items(), key=lambda x: x[1], reverse=True)
        # <sos/eos> is added here for the compatibility with autoregressive TTS model
        subset_phn_vocab = ["<blank>"] + [phn for phn, _ in phn2freq] + ['<unk>', '<sos/eos>']

        vocab_path = os.path.join(save_path, i, 'vocab')
        np.savetxt(vocab_path, subset_phn_vocab, fmt="%s")
        print(f"Phoneme vocabulary has been successfully saved to {vocab_path}.")

if __name__ == '__main__':
    args = parse()
    main(**vars(args))
