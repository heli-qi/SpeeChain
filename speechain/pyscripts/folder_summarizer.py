import argparse
import os
import numpy as np

from tqdm import tqdm

from speechain.utilbox.import_util import parse_path_args
from speechain.utilbox.data_loading_util import search_file_in_subfolder


def parse():
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--src_folder', type=str, required=True,
                        help="The source folder where your target files are placed.")
    parser.add_argument('--tgt_path', type=str, default=None,
                        help="The target path you want to save the summary file. "
                             "If not given, the summary file will be saved to the parent directory of 'src_folder'.")
    parser.add_argument('--sum_file_name', type=str, default=None, help="The name of the summary file.")
    return parser.parse_args()


def main(src_folder: str, sum_file_name: str = None, tgt_path: str = None):
    if tgt_path is None:
        tgt_path = '/'.join(src_folder.split('/')[:-1])
    if sum_file_name is None:
        sum_file_name = f"idx2{src_folder.split('/')[-1]}"

    src_folder, tgt_path = parse_path_args(src_folder), parse_path_args(tgt_path)

    file_path_summary, empty_file_summary = {}, {}
    for file_path in tqdm(search_file_in_subfolder(src_folder)):
        # get rid of the extension
        file_index = '.'.join(os.path.basename(file_path).split('.')[:-1])
        file_path_summary[file_index] = file_path

    np.savetxt(os.path.join(tgt_path, sum_file_name), sorted(file_path_summary.items(), key=lambda x: x[0]), fmt='%s')


if __name__ == '__main__':
    args = parse()
    main(**vars(args))
