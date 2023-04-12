import argparse
import os
import numpy as np

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


def main(src_folder: str, tgt_path: str, sum_file_name: str):
    if tgt_path is None:
        tgt_path = '/'.join(src_folder.split('/')[:-1])
    if sum_file_name is None:
        sum_file_name = f"idx2{src_folder.split('/')[-1]}"

    src_folder, tgt_path = parse_path_args(src_folder), parse_path_args(tgt_path)

    file_summary = {}
    file_path_list = search_file_in_subfolder(src_folder)
    for file_path in file_path_list:
        file_name = file_path.split('/')[-1]
        # get rid of the extension
        file_summary['.'.join(file_name.split('.')[:-1])] = os.path.join(src_folder, file_name)

    np.savetxt(os.path.join(tgt_path, sum_file_name), sorted(file_summary.items(), key=lambda x: x[0]), fmt='%s')


if __name__ == '__main__':
    # args = parse()
    # main(**vars(args))

    main(src_folder="recipes/offline_tts2asr/tts_syn_speech/libritts/train-other-500/default_inference/10_train_loss_average/seed=0_token=g2p_spk-emb=libritts-train-clean-100-ecapa_model=recipes%tts%libritts%train-clean-100%exp%16khz_ecapa_g2p_transformer-v3_accum1_20gb/gl_wav",
         tgt_path=None, sum_file_name='idx2gl_wav')
