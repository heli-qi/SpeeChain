import os
import argparse
from typing import List

from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

from speechain.utilbox.import_util import parse_path_args
from speechain.utilbox.data_loading_util import load_idx2data_file
from speechain.utilbox.type_util import str2bool


def parse():
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--corpus_path', type=str, required=True,
                        help="The path where your dumped dataset is placed.")
    parser.add_argument('--lab_cover_flag', type=str2bool, default=False,
                        help="The number of processes you want to use to generate .lab files. (default: 8)")
    parser.add_argument('--ncpu', type=int, default=8,
                        help="The number of processes you want to use to generate .lab files. (default: 8)")
    return parser.parse_args()


def save_lab_files(idx2wav_text: List[List[str]], lab_cover_flag: bool = False):
    for idx, wav_path, text in tqdm(idx2wav_text):
        lab_path = os.path.join(os.path.dirname(wav_path), f'{idx}.lab')
        if os.path.exists(lab_path) and not lab_cover_flag:
            return

        with open(lab_path, mode='w') as f:
            f.write(text)


def proc_subset(idx2wav_path: str, lab_cover_flag: bool = False, ncpu: int = 8):
    subset_path = os.path.dirname(idx2wav_path)
    idx2text_path = os.path.join(subset_path, 'idx2no-punc_text')
    if not os.path.exists(idx2text_path):
        raise RuntimeError(f"idx2no-punc_text doesn't exist in {subset_path}! "
                           f"Please generate the metadata files again with '--txt_format no-punc'."
                           f"In SpeeChain, MFA alignment is done by non-punctuated text for better results.")

    print(f'Start to generate .lab files by {idx2wav_path} and {idx2text_path}...')
    idx2wav = load_idx2data_file(idx2wav_path)
    idx2text = load_idx2data_file(idx2text_path)
    idx_wav_text_list = [[idx, idx2wav[idx], idx2text[idx]] for idx in idx2wav.keys()]
    func_args = [idx_wav_text_list[i::ncpu] for i in range(ncpu)]
    save_lab_files_func = partial(save_lab_files, lab_cover_flag=lab_cover_flag)

    # generating the .lab files to the disk
    with Pool(ncpu) as executor:
        executor.map(save_lab_files_func, func_args)

def main(corpus_path: str, lab_cover_flag: bool = False, ncpu: int = 8):
    corpus_path = parse_path_args(corpus_path)
    # corpus_path is the folder path of a subset
    if os.path.exists(os.path.join(corpus_path, 'idx2wav')):
        proc_subset(idx2wav_path=os.path.join(corpus_path, 'idx2wav'), lab_cover_flag=lab_cover_flag, ncpu=ncpu)

    # corpus_path is the folder path of a dataset (including many subsets as sub-folders)
    else:
        for subset in os.listdir(corpus_path):
            # only consider the sub-folder of each subset in corpus_path
            if not os.path.isdir(os.path.join(corpus_path, subset)):
                continue

            idx2wav_path = os.path.join(corpus_path, subset, 'idx2wav')
            if os.path.exists(idx2wav_path):
                proc_subset(idx2wav_path=idx2wav_path, lab_cover_flag=lab_cover_flag, ncpu=ncpu)
            else:
                raise RuntimeError(f"idx2wav doesn't exist in {os.path.join(corpus_path, subset)}!")


if __name__ == '__main__':
    args = parse()
    main(**vars(args))
