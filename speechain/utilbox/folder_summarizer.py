import argparse
import os
import numpy as np

from speechain.utilbox.import_util import parse_path_args


def parse():
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--src_folder', type=str, required=True,
                        help="The source folder where your target files are placed.")
    parser.add_argument('--tgt_path', type=str, default=None,
                        help="The target path you want to save the summary file. "
                             "If not given, the summary file will be saved to the parent directory of 'src_folder'.")
    parser.add_argument('--sum_file_name', type=str, required=True, help="The name of the summary file.")
    return parser.parse_args()


def main(src_folder: str, tgt_path: str, sum_file_name: str):
    if tgt_path is None:
        tgt_path = '/'.join(src_folder.split('/')[:-1])
    src_folder, tgt_path = parse_path_args(src_folder), parse_path_args(tgt_path)

    file_summary = {}
    for file_name in os.listdir(src_folder):
        if os.path.isdir(file_name):
            continue
        # get rid of the extension
        file_summary['.'.join(file_name.split('.')[:-1])] = os.path.join(src_folder, file_name)

    np.savetxt(os.path.join(tgt_path, sum_file_name), sorted(file_summary.items(), key=lambda x: x[0]), fmt='%s')


if __name__ == '__main__':
    args = parse()
    main(**vars(args))
