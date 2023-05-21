import argparse
import os
import numpy as np

from tqdm import tqdm
from typing import List
from multiprocessing import Pool

from speechain.utilbox.import_util import parse_path_args
from speechain.utilbox.data_loading_util import search_file_in_subfolder, read_data_by_path, load_idx2data_file


def parse():
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--src_folder', type=str, required=True,
                        help="The source folder where your target files are placed.")
    parser.add_argument('--ncpu', type=int, default=8,
                        help="The source folder where your target files are placed.")
    return parser.parse_args()

def get_empty_file(wav_and_len_list: List[List[str or int]]):
    empty_file_list = []
    # loop through each source data file in the given chunk
    for wav_path, wav_len in tqdm(wav_and_len_list):
        wav = read_data_by_path(wav_path, return_tensor=True)
        if wav.size(0) != wav_len:
            empty_file_list.append(wav_path)
            print(f"Find an audio file with incorrect length (should be {wav_len}, but be {wav.size(0)}): {wav_path}")
    return empty_file_list


def main(src_folder: str, vocoder: str = 'hifigan', ncpu: int = 16):
    src_folder = parse_path_args(src_folder)
    idx2wav, idx2wav_len = load_idx2data_file(os.path.join(src_folder, f"idx2{vocoder}_wav")), load_idx2data_file(os.path.join(src_folder, f"idx2{vocoder}_wav_len"), data_type=int)
    wav_and_len_list = [[idx2wav[idx], idx2wav_len[idx]] for idx in idx2wav.keys()]
    func_args = [wav_and_len_list[i::ncpu] for i in range(ncpu)]

    # read all the assigned data files and get their lengths
    with Pool(ncpu) as executor:
        empty_file_list_nproc = executor.map(get_empty_file, func_args)

    empty_file_list = [item for sublist in empty_file_list_nproc for item in sublist]
    print(empty_file_list)

if __name__ == '__main__':
    # args = parse()
    # main(**vars(args))

    main(src_folder="recipes/offline_tts2asr/tts_syn_speech/librispeech/train-clean-360/return_sr=16000/seed=0_ngpu=2_batch-len=2000_spk-emb=libritts-train-clean-100-xvector-aver_model=recipes%tts%libritts%train-clean-100%exp%22.05khz_xvector_mfa_fastspeech2")
