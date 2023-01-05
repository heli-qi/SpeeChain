"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.12
"""
import os
import argparse
import numpy as np

from typing import Dict
from functools import partial
from multiprocessing import Pool

from speechain.utilbox.data_loading_util import parse_path_args, load_idx2data_file, read_data_by_path
from speechain.utilbox.yaml_util import load_yaml
from speechain.utilbox.feat_util import convert_wav_to_stft, convert_wav_to_logmel, convert_wav_to_mfcc


def parse():
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--idx2wav', type=str, required=True,
                        help="The absolute path of the 'idx2wav' containing the addresses of all the hypothesis "
                             "utterances to be evaluated.")
    parser.add_argument('--feat_type', type=str, required=True, help="The type of your target acoustic features.")
    parser.add_argument('--feat_config', type=str, required=True,
                        help="The path of the configuration file of your target acoustic feature extraction.")
    parser.add_argument('--feat_path', type=str, required=True,
                        help="The path where you want to save the extracted acoustic feature files. (default: None)")
    parser.add_argument('--ncpu', type=int, default=8,
                        help="The number of processes you want to use to extract acoustic features. (default: 8)")
    return parser.parse_args()


def extract_and_save_feat(data_index: str, wav_path: str, feat_type: str, feat_config: Dict, feat_path: str):
    # read the waveform data into the memory
    wav, sample_rate = read_data_by_path(wav_path, return_sample_rate=True)
    if 'sr' in feat_config.keys():
        feat_config.pop('sr')

    # extract the acoustic features from the waveform by the configuration
    if feat_type == 'stft':
        feat = convert_wav_to_stft(wav, sr=sample_rate, **feat_config)
    elif feat_type == 'logmel':
        feat = convert_wav_to_logmel(wav, sr=sample_rate, **feat_config)
    elif feat_type == 'mfcc':
        feat = convert_wav_to_mfcc(wav, sr=sample_rate, **feat_config)
    else:
        raise ValueError(f"Unknown acoustic feature: {feat_type}! It should be one of ['stft', 'logmel', 'mfcc'].")

    # save the acoustic features to the disk
    np.savez(os.path.join(feat_path, f"{data_index}.npz"), feat=feat, sample_rate=sample_rate)
    return data_index, os.path.join(feat_path, f"{data_index}.npz")


def main(idx2wav: str, feat_type: str, feat_config: str, feat_path: str, ncpu: int):
    # configuration initialization
    feat_config = load_yaml(parse_path_args(feat_config))

    # path initialization
    idx2wav = load_idx2data_file(parse_path_args(idx2wav))
    feat_path = parse_path_args(feat_path)
    os.makedirs(feat_path, exist_ok=True)

    # skip the extraction process if there has already been an idx2feat
    if not os.path.exists(os.path.join(feat_path, 'idx2feat')):
        idx2wav_list = list(idx2wav.items())
        with Pool(ncpu) as executor:
            extract_and_save_feat_func = partial(extract_and_save_feat,
                                                 feat_type=feat_type, feat_config=feat_config, feat_path=feat_path)
            idx2feat_list = executor.starmap(extract_and_save_feat_func, idx2wav_list)

        # debug use
        # idx2feat_list = [extract_and_save_feat_func(*ele) for ele in idx2wav_list]

        # save the idx2feat file to feat_path as the reference
        np.savetxt(os.path.join(feat_path, 'idx2feat'), sorted(idx2feat_list, key=lambda x: x[0]), fmt='%s')

    else:
        print(f"Acoustic features have already existed in {feat_path}, so the extraction process is skipped.")


if __name__ == '__main__':
    args = parse()
    main(**vars(args))
