import os
import argparse
import numpy as np

from tqdm import tqdm
from typing import List
from multiprocessing import Pool
from speechain.utilbox.data_loading_util import read_data_by_path, parse_path_args, load_idx2data_file


def parse():
    """
        Parse command line arguments using argparse.

        Returns:
            argparse.Namespace: Parsed arguments as a namespace object.
    """
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--src_file', type=str, required=True,
                        help="The absolute path of your input 'idx2wav' or 'idx2feat' file. "
                             "The corresponding length file 'idx2wav_len' or 'idx2feat_len' will be saved to the same directory.")
    parser.add_argument('--ncpu', type=int, default=8,
                        help="The number of CPU cores to use for parallel processing.")
    return parser.parse_args()


def get_data_length(idx2data: List[List[str]]):
    """
        Calculate the length of each feature in the given chunk of idx2data.

        Args:
            idx2data (List[List[str]]):
                A list of [idx, data_path] pairs.

        Returns:
            List[List[str]]: A list of [idx, data_length] pairs.
    """
    # loop through each source data file in the given chunk
    return [[idx, read_data_by_path(data_path).shape[0]] for idx, data_path in tqdm(idx2data, desc="Processing data files")]


def main(src_file: str, ncpu: int):
    """
        Main function that extracts feature lengths and saves them to file.

        Args:
            src_file (str):
                The absolute path of the input 'idx2wav' or 'idx2feat' file.
            ncpu (int):
                The number of processes to use for the calculation.
    """
    src_file = parse_path_args(src_file)
    src_path = os.path.dirname(src_file)
    src_file_name = os.path.basename(src_file)
    tgt_file_name = f"{src_file_name}_len"
    tgt_path = os.path.join(src_path, tgt_file_name)

    # skip the length dumping process if there has already been a length file
    if not os.path.exists(tgt_path):
        idx2data = [[idx, data] for idx, data in load_idx2data_file(src_file).items()]
        func_args = [idx2data[i::ncpu] for i in range(ncpu)]

        # read all the assigned data files and get their lengths
        with Pool(ncpu) as executor:
            data_length_list_nproc = executor.map(get_data_length, func_args)

        data_length_list = [item for sublist in data_length_list_nproc for item in sublist]
        np.savetxt(tgt_path, sorted(data_length_list, key=lambda x: x[0]), fmt='%s')
    else:
        print(f"Data length file {tgt_file_name} already exists in {src_path}, "
              f"so the length dumping process is skipped.")
    print("\n")


if __name__ == '__main__':
    args = parse()
    main(**vars(args))
