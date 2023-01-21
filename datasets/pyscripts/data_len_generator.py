"""
    Author: Sashi Novitasari
    Affiliation: NAIST
    Date: 2022.07

    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import os
import argparse
import numpy as np

from tqdm import tqdm
from typing import List
from multiprocessing import Pool
from speechain.utilbox.data_loading_util import read_data_by_path, parse_path_args, load_idx2data_file


def parse():
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--src_file', type=str, required=True,
                        help="The absolute path of your input 'idx2wav' or 'idx2feat'. The length file 'idx2wav_len' "
                             "or 'idx2feat_len' will be saved to the directory of data_path.")
    parser.add_argument('--ncpu', type=int, default=8,
                        help="The number of processes you want to use to extract the feature lengths.")
    return parser.parse_args()


def get_feat_length(idx2data: List[List[str]]):
    idx2data_len = []
    # loop each source data wav in the given chunk
    for idx, data_path in tqdm(idx2data):
        idx2data_len.append([idx, read_data_by_path(data_path).shape[0]])
    return idx2data_len


def main(src_file: str, ncpu: int):
    src_file = parse_path_args(src_file)
    src_path = os.path.dirname(src_file)
    src_file_name = os.path.basename(src_file)
    tgt_file_name = '_'.join([src_file_name, 'len'])
    tgt_path = os.path.join(src_path, tgt_file_name)

    # skip the length dumping process if there has already been a idx2wav_len or idx2feat_len
    if not os.path.exists(tgt_path):
        idx2data = load_idx2data_file(src_file)
        idx2data = [[idx, data] for idx, data in idx2data.items()]
        func_args = [idx2data[i::ncpu] for i in range(ncpu)]

        # read all the assigned data files and get their lengths
        with Pool(ncpu) as executor:
            idx2data_len_list_nproc = executor.map(get_feat_length, func_args)

        idx2data_len = []
        for idx2data_len_list in idx2data_len_list_nproc:
            idx2data_len += idx2data_len_list
        np.savetxt(tgt_path, sorted(idx2data_len, key=lambda x: x[0]), fmt='%s')
    else:
        print(f"Data length file {tgt_file_name} have already existed in {src_path}, "
              f"so the length dumping process is skipped.")

    print("\n")


if __name__ == '__main__':
    args = parse()
    main(**vars(args))
