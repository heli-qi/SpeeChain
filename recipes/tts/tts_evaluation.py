"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.12
"""
import math
import os
import argparse
import librosa
import numpy as np
import torch
import torchaudio

from typing import List, Dict
from functools import partial
from multiprocessing import Pool

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from speechain.utilbox.type_util import str2list
from speechain.utilbox.data_loading_util import parse_path_args, load_idx2data_file, read_data_by_path
from speechain.utilbox.yaml_util import load_yaml
from speechain.utilbox.feat_util import convert_wav_to_mfcc
from speechain.utilbox.md_util import get_list_strings


def parse():
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--hypo_idx2wav', type=str, required=True,
                        help="The absolute path of the 'idx2wav' containing the addresses of all the hypothesis "
                             "utterances to be evaluated.")
    parser.add_argument('--refer_idx2wav', type=str, default=None,
                        help="The absolute path of the 'idx2wav' containing the addresses of all the reference "
                             "utterances used as the ground-truth. "
                             "This argument is required if your selected metrics are comparison-based, such as MCD."
                             "(default: None)")
    parser.add_argument('--metric_list', type=str2list, default=['mcd'],
                        help="The list of metrics you want to use to evaluate your given hypothesis utterances. "
                             "Please give this argument as a string surrounded by a pair of square brackets where your "
                             "metrics are split by commas. (default: [mcd])")
    parser.add_argument('--result_path', type=str, default=None,
                        help="The path where the evaluated results are placed. If not given, the results will be saved "
                             "to the same directory as your given 'hypo_idx2wav'. (default: None)")
    parser.add_argument('--ncpu', type=int, default=8,
                        help="The number of processes you want to use to calculate the evaluation metrics.")
    parser.add_argument('--topn_num', type=int, default=30,
                        help="The number of top-n bad cases you want to show in the result md file. (default: 30)")
    return parser.parse_args()


def calculate_mcd(idx: str, hypo_path: str, refer_path: str, mfcc_config: Dict):
    # read the waveforms into the memory
    hypo_wav, hypo_sample_rate = read_data_by_path(hypo_path, return_sample_rate=True)
    refer_wav, refer_sample_rate = read_data_by_path(refer_path, return_sample_rate=True)

    # deal with the difference of sampling rates
    if hypo_sample_rate < refer_sample_rate:
        refer_wav = librosa.resample(refer_wav.squeeze(-1), orig_sr=refer_sample_rate, target_sr=hypo_sample_rate)
        refer_sample_rate = hypo_sample_rate
    elif hypo_sample_rate > refer_sample_rate:
        hypo_wav = librosa.resample(hypo_wav.squeeze(-1), orig_sr=hypo_sample_rate, target_sr=refer_sample_rate)
        hypo_sample_rate = refer_sample_rate

    # extract the MFCC features from the waveforms
    if 'sr' in mfcc_config.keys():
        mfcc_config.pop('sr')
    hypo_mfcc = convert_wav_to_mfcc(hypo_wav, sr=hypo_sample_rate, **mfcc_config)
    refer_mfcc = convert_wav_to_mfcc(refer_wav, sr=refer_sample_rate, **mfcc_config)

    # MCD calculation
    mfcc_path = fastdtw(hypo_mfcc, refer_mfcc, dist=euclidean)[1]
    hypo_mfcc = hypo_mfcc[[ele[0] for ele in mfcc_path]]
    refer_mfcc = refer_mfcc[[ele[1] for ele in mfcc_path]]

    # if the coefficient in the traditional MCD calculation equation is multiplied, the MCD result is weirdly large
    # as https://github.com/facebookresearch/fairseq/blob/3c1abb59f581bfce68822b6179d1c1b37b304259/fairseq/tasks/text_to_speech.py#L355
    # fairseq also doesn't multiply this coefficient into the MCD results
    coeff = 10 / np.log(10) * np.sqrt(2)
    mcd = coeff * np.mean(np.sqrt(np.sum((hypo_mfcc - refer_mfcc) ** 2, axis=1)))

    return idx, mcd


def save_results(idx2result_list: List[List], metric_name: str, result_path: str, vocoder_name: str,
                 desec_sort: True, topn_num: int):
    # save the idx2feat file to feat_path as the reference
    np.savetxt(os.path.join(result_path, f'idx2{vocoder_name}_{metric_name}'),
               sorted(idx2result_list, key=lambda x: x[0]), fmt='%s')

    # record the overall results
    result_mean = np.mean([idx2result[1] for idx2result in idx2result_list])
    result_std = np.std([idx2result[1] for idx2result in idx2result_list])
    md_report = f"# Overall {metric_name} Result: (mean ± std)\n" \
                f"{result_mean:.4f} ± {result_std:.4f}\n" \
                f"# Top{topn_num} Bad Cases for {metric_name}\n"

    # record the data instances with the top-n largest results
    idx2result_list = sorted(idx2result_list, key=lambda x: x[1], reverse=desec_sort)[: topn_num]
    idx2result_dict = {idx: f"{mcd:.4f}" for idx, mcd in idx2result_list}
    md_report += get_list_strings(idx2result_dict)
    np.savetxt(os.path.join(result_path, f'{vocoder_name}_{metric_name}_results.md'), [md_report], fmt='%s')


def main(hypo_idx2wav: str, refer_idx2wav: str, metric_list: List[str], result_path: str, ncpu: int, topn_num: int):
    """

    Args:
        hypo_idx2wav: str
        refer_idx2wav: str
            The path of the 'idx2wav' of
        metric_list: List[str]
            Your specified metrics used to evaluate your given hypothesis utterances.
        result_path: str
            The path to place the evaluation results. If not given, the evaluation results will be saved to the same
            directory as your given hypo_idx2wav.
        ncpu: int
            The number of processes used to evaluate the specified hypothesis utterances.
        topn_num: int
            The number of top-n bad cases you want to show in the result md file.

    """
    # --- 1. Argument Preparation stage --- #
    # argument checking
    for i in range(len(metric_list)):
        metric_list[i] = metric_list[i].lower()
        assert metric_list[i] in ['mcd'],\
            f"Your input metric should be one of ['mcd', 'mosnet'], but got {metric_list[i]}!"

        if metric_list[i] in ['mcd']:
            assert refer_idx2wav is not None, \
                "If you want to evaluate your hypothesis utterances by the comparison-based metrics, " \
                "refer_idx2wav must be given!"

    assert hypo_idx2wav.endswith('wav'), "please include the name of your target 'idx2wav' file in hypo_idx2wav!"
    if refer_idx2wav is not None:
        assert refer_idx2wav.endswith('wav'), \
            "please include the name of your target 'idx2wav' file in refer_idx2wav!"

    # path initialization
    hypo_idx2wav = parse_path_args(hypo_idx2wav)
    if refer_idx2wav is not None:
        refer_idx2wav = parse_path_args(refer_idx2wav)
    # register the paths of 'idx2wav' files for later use
    hypo_idx2wav_path, refer_idx2wav_path = hypo_idx2wav, refer_idx2wav

    # automatically initialize result_path and vocoder_name
    result_path = os.path.dirname(hypo_idx2wav) if result_path is None else parse_path_args(result_path)
    vocoder_name = os.path.basename(hypo_idx2wav).split('2')[-1].split('_')[0]

    # read the file data into a Dict by the path
    hypo_idx2wav = load_idx2data_file(hypo_idx2wav)
    if refer_idx2wav is not None:
        refer_idx2wav = load_idx2data_file(refer_idx2wav)

    # key checking for hypo_idx2wav and refer_idx2wav
    assert len(set(hypo_idx2wav.keys()).difference(set(refer_idx2wav.keys()))) == 0, \
        "The indices of some hypothesis utterances in hypo_idx2wav don't exist in your given refer_idx2wav!"

    # --- 2. Metric-wise Evaluation --- #
    # loop each given metric
    for metric in metric_list:
        if metric == 'mcd':
            mfcc_config = load_yaml(parse_path_args('config/feat/mfcc/mcd_for_tts.yaml'))
            idx2hypo_refer_list = [[idx, hypo_idx2wav[idx], refer_idx2wav[idx]] for idx in hypo_idx2wav.keys()]
            with Pool(ncpu) as executor:
                calculate_mcd_func = partial(calculate_mcd, mfcc_config=mfcc_config)
                idx2mcd_list = executor.starmap(calculate_mcd_func, idx2hypo_refer_list)

            save_results(idx2result_list=idx2mcd_list, metric_name='mcd', result_path=result_path,
                         vocoder_name=vocoder_name, desec_sort=True, topn_num=topn_num)
        else:
            raise NotImplementedError

    # --- 3. Waveform Length Ratio Evaluation --- #
    if refer_idx2wav is not None:
        hypo_idx2wav_len, refer_idx2wav_len = hypo_idx2wav_path + '_len', refer_idx2wav_path + '_len'
        if os.path.exists(hypo_idx2wav_len) and os.path.exists(refer_idx2wav_len):
            hypo_idx2wav_len, refer_idx2wav_len = \
                load_idx2data_file(hypo_idx2wav_len, int), load_idx2data_file(refer_idx2wav_len, int)

            # key checking for hypo_idx2wav_len and refer_idx2wav_len
            assert len(set(hypo_idx2wav_len.keys()).difference(set(refer_idx2wav_len.keys()))) == 0, \
                "The indices of some hypothesis utterances in hypo_idx2wav_len don't exist in your given " \
                "refer_idx2wav_len!"
            idx2wav_len_ratio = {key: hypo_idx2wav_len[key] / refer_idx2wav_len[key] for key in hypo_idx2wav_len.keys()}
            idx2wav_len_ratio_list = list(idx2wav_len_ratio.items())

            save_results(idx2result_list=idx2wav_len_ratio_list, metric_name='wav_len_ratio', result_path=result_path,
                         vocoder_name=vocoder_name, desec_sort=True, topn_num=topn_num)


if __name__ == '__main__':
    args = parse()
    main(**vars(args))
