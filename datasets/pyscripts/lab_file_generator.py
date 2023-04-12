import os
import argparse
import warnings
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
    parser.add_argument('--cover_flag', type=str2bool, default=False,
                        help="Whether to cover the existing .lab files. (default: False)")
    parser.add_argument('--ncpu', type=int, default=8,
                        help="The number of processes you want to use to generate .lab files. (default: 8)")
    return parser.parse_args()


def save_lab_files(idx2wav_text: List[List[str]], cover_flag: bool):
    for idx, wav_path, text in tqdm(idx2wav_text):
        lab_path = '/'.join(wav_path.split('/')[:-1] + [f'{idx}.lab'])
        if not cover_flag and os.path.exists(lab_path):
            continue
        with open(lab_path, mode='w') as f:
            f.write(text)


def proc_subset(idx2wav_path: str, cover_flag: bool, ncpu: int):
    subset_path = os.path.dirname(idx2wav_path)
    idx2text_path = os.path.join(subset_path, 'idx2asr_text')
    if not os.path.exists(idx2text_path):
        idx2text_path = os.path.join(subset_path, 'idx2tts_text')
        assert os.path.exists(idx2text_path), \
            f"Either idx2asr_text or idx2tts_text doesn't exist in {subset_path}!"
        warnings.warn(
            f"idx2asr_text doesn't exist in {subset_path}, so idx2tts_text will be used. "
            f"But the existence of punctuation marks may influence the duration prediction, "
            f"so we recommend you to stop this job and dump the idx2asr_text if possible for you.")

    print(f'Start to generate .lab files by {idx2wav_path}...')
    idx2wav = load_idx2data_file(idx2wav_path)
    idx2text = load_idx2data_file(idx2text_path)
    idx_wav_text_list = [[idx, idx2wav[idx], idx2text[idx]] for idx in idx2wav.keys()]
    func_args = [idx_wav_text_list[i::ncpu] for i in range(ncpu)]

    # generating the .lab files to the disk
    with Pool(ncpu) as executor:
        save_lab_files_func = partial(save_lab_files, cover_flag=cover_flag)
        executor.map(save_lab_files_func, func_args)


def main(corpus_path: str, cover_flag: bool, ncpu: int):
    corpus_path = parse_path_args(corpus_path)
    # corpus_path is the folder path of a subset
    if os.path.exists(os.path.join(corpus_path, 'idx2wav')):
        proc_subset(idx2wav_path=os.path.join(corpus_path, 'idx2wav'), cover_flag=cover_flag, ncpu=ncpu)

    # corpus_path is the folder path of a dataset (including many subsets as sub-folders)
    else:
        for subset in os.listdir(corpus_path):
            # only consider the sub-folder of each subset in corpus_path
            if not os.path.isdir(os.path.join(corpus_path, subset)):
                continue

            idx2wav_path = os.path.join(corpus_path, subset, 'idx2wav')
            if os.path.exists(idx2wav_path):
                proc_subset(idx2wav_path=idx2wav_path, cover_flag=cover_flag, ncpu=ncpu)
            else:
                raise RuntimeError(f"idx2wav doesn't exist in {os.path.join(corpus_path, subset)}!")


if __name__ == '__main__':
    args = parse()
    main(**vars(args))
