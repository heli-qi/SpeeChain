"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.12
"""
import os
import argparse
import librosa
import numpy as np
from tqdm import tqdm

from typing import List
from functools import partial
from multiprocessing import Pool

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from speechain.utilbox.type_util import str2list
from speechain.utilbox.data_loading_util import parse_path_args, load_idx2data_file, read_data_by_path, search_file_in_subfolder
from speechain.utilbox.feat_util import convert_wav_to_mfcc, convert_wav_to_logmel
from speechain.utilbox.md_util import get_list_strings


def parse():
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--hypo_path', type=str, required=True,
                        help="The path of your TTS experimental folder. All the files named 'idx2xxx_wav' will be "
                            "automatically found out and used for TTS objective evaluation. You can also directly "
                            "specify the path of your target 'idx2xxx_wav' file by this argument.")
    parser.add_argument('--refer_path', type=str, default=None,
                        help="The path of the ground-truth data folder. All the files named 'idx2wav' will be "
                             "automatically found out and used as the reference. The hypo 'idx2xxx_wav' and refer "
                             "'idx2wav' will be matched by the data indices. You can also directly specify the path of "
                             "your target 'idx2wav' file by this argument. This argument is required if you want to "
                             "evaluate the MCD or MSD.")
    parser.add_argument('--metric_list', type=str2list, default=['mcd', 'msd'],
                        help="The list of metrics you want to use to evaluate your given hypothesis utterances. "
                             "Please give this argument as a string surrounded by a pair of square brackets where your "
                             "metrics are split by commas. (default: [mcd, msd])")
    parser.add_argument('--resule_path', type=str, default=None,
                        help="The path where the evaluated results are placed. If not given, the results will be saved "
                             "to the same directory as the hypo 'idx2xxx_wav' found in 'hypo_path'. (default: None)")
    parser.add_argument('--ncpu', type=int, default=8,
                        help="The number of processes used to calculate the evaluation metrics. (default: 8)")
    parser.add_argument('--topn_num', type=int, default=30,
                        help="The number of top-n bad cases you want to show in the result md file. (default: 30)")
    return parser.parse_args()


def calculate_mcd_msd(idx2hypo_refer_list: List[str], tgt_metric: str = 'mcd'):
    tgt_metric = tgt_metric.lower()
    assert tgt_metric in ['mcd', 'msd'], f"tgt_metric must be either 'mcd' or 'msd', but got {tgt_metric}"

    output = dict()
    for idx, hypo_path, refer_path in tqdm(idx2hypo_refer_list):
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
        if tgt_metric == 'mcd':
            hypo_feat = convert_wav_to_mfcc(hypo_wav, sr=hypo_sample_rate,
                                            n_mfcc=13, win_length=0.05, hop_length=0.0125)
            refer_feat = convert_wav_to_mfcc(refer_wav, sr=refer_sample_rate,
                                             n_mfcc=13, win_length=0.05, hop_length=0.0125)
        # extract the Log-Mel features from the waveforms
        else:
            hypo_feat = convert_wav_to_logmel(hypo_wav, sr=hypo_sample_rate,
                                              n_mels=80, win_length=0.05, hop_length=0.0125)
            refer_feat = convert_wav_to_logmel(refer_wav, sr=refer_sample_rate,
                                               n_mels=80, win_length=0.05, hop_length=0.0125)

        # MCD calculation
        dtw_path = fastdtw(hypo_feat, refer_feat, dist=euclidean)[1]
        hypo_feat = hypo_feat[[ele[0] for ele in dtw_path]]
        refer_feat = refer_feat[[ele[1] for ele in dtw_path]]

        coeff = 10 / np.log(10) * np.sqrt(2)
        output[idx] = coeff * np.mean(np.sqrt(np.sum((hypo_feat - refer_feat) ** 2, axis=1)))

    return output


def save_results(idx2result_list: List[List], metric_name: str, save_path: str, vocoder_name: str,
                 desec_sort: True, topn_num: int):
    # save the idx2feat file to feat_path as the reference
    np.savetxt(os.path.join(save_path, f'idx2{f"{vocoder_name}_" if vocoder_name is not None else ""}{metric_name}'),
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
    np.savetxt(
        os.path.join(save_path, f'{f"{vocoder_name}_" if vocoder_name is not None else ""}{metric_name}_results.md'),
        [md_report], fmt='%s')


def main(hypo_path: str, refer_path: str, metric_list: List[str], resule_path: str, ncpu: int, topn_num: int):
    """

    Args:
        hypo_path: str
        refer_path: str
            The path of the 'idx2wav' of
        metric_list: List[str]
            Your specified metrics used to evaluate your given hypothesis utterances.
        resule_path: str
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
        assert metric_list[i] in ['mcd', 'msd'],\
            f"Your input metric should be one of ['mcd', 'msd'], but got {metric_list[i]}!"

        if metric_list[i] in ['mcd', 'msd']:
            assert refer_path is not None, \
                "If you want to evaluate your hypothesis utterances by the comparison-based metrics, " \
                "refer_path must be given!"

    hypo_path = parse_path_args(hypo_path)
    # for folder input, automatically find out all the idx2xxx_wav candidates as hypo_idx2wav
    if os.path.isdir(hypo_path):
        hypo_idx2wav_list = search_file_in_subfolder(hypo_path, lambda x: x.startswith('idx2') and x.endswith('wav'))
    # for file input, directly use it as hypo_idx2feat
    else:
        hypo_idx2wav_list = [hypo_path]

    if refer_path is not None:
        refer_path = parse_path_args(refer_path)
        # for folder input, automatically find out all the idx2xxx_wav candidates as refer_idx2wav
        if os.path.isdir(refer_path):
            refer_idx2wav_list = search_file_in_subfolder(refer_path,
                                                          lambda x: x.startswith('idx2') and x.endswith('wav'))
        # for file input, directly use it as hypo_idx2wav
        else:
            refer_idx2wav_list = [refer_path]
    else:
        refer_idx2wav_list = None

    for hypo_idx2wav in hypo_idx2wav_list:
        # automatically initialize save_path and vocoder_name
        save_path = os.path.dirname(hypo_idx2wav) if resule_path is None else parse_path_args(resule_path)
        wav_name = os.path.basename(hypo_idx2wav).split('2')[-1]
        vocoder_name = wav_name.split('_')[0] if '_' in wav_name else None

        # read the file data into a Dict by the path
        # register the paths of hypo 'idx2wav' files for later use
        hypo_idx2wav_path = hypo_idx2wav
        hypo_idx2wav = load_idx2data_file(hypo_idx2wav)

        if refer_idx2wav_list is not None:
            hypo_idx2wav_keys = set(hypo_idx2wav.keys())
            key_match_flag = False

            for refer_idx2wav in refer_idx2wav_list:
                # register the paths of refer 'idx2wav' files for later use
                refer_idx2wav_path = refer_idx2wav
                refer_idx2wav = load_idx2data_file(refer_idx2wav)
                refer_idx2wav_keys = set(refer_idx2wav.keys())

                key_match_flag = len(hypo_idx2wav_keys.difference(refer_idx2wav_keys)) == 0
                if key_match_flag:
                    break

            if not key_match_flag:
                print(f"None of the refer 'idx2wav' in your given refer_path {refer_path} match {hypo_idx2wav_path}, "
                      f"so it will be skipped!")
                continue
        else:
            refer_idx2wav_path, refer_idx2wav = None, None

        # --- 2. Metric-wise Evaluation --- #
        # loop each given metric
        for metric in metric_list:
            report_name = f'{f"{vocoder_name}_" if vocoder_name is not None else ""}{metric}_results.md'
            if os.path.exists(os.path.join(save_path, report_name)):
                print(f"Evaluation report {report_name} has already existed in {save_path}! "
                      f"{metric} evaluation for {hypo_idx2wav_path} will be skipped...")
                continue
            else:
                print(f"Start {metric} evaluation for {hypo_idx2wav_path}!")

            if metric == 'mcd':
                idx2hypo_refer_list = [[idx, hypo_idx2wav[idx], refer_idx2wav[idx]] for idx in hypo_idx2wav.keys()]
                func_args = [idx2hypo_refer_list[i::ncpu] for i in range(ncpu)]

                with Pool(ncpu) as executor:
                    calculate_mcd_func = partial(calculate_mcd_msd, tgt_metric='mcd')
                    idx2mcd_dict_list = executor.map(calculate_mcd_func, func_args)
                    idx2mcd_list = []
                    for idx2mcd_dict in idx2mcd_dict_list:
                        idx2mcd_list += list(idx2mcd_dict.items())
                save_results(idx2result_list=idx2mcd_list, metric_name='mcd', save_path=save_path,
                             vocoder_name=vocoder_name, desec_sort=True, topn_num=topn_num)

            elif metric == 'msd':
                idx2hypo_refer_list = [[idx, hypo_idx2wav[idx], refer_idx2wav[idx]] for idx in hypo_idx2wav.keys()]
                func_args = [idx2hypo_refer_list[i::ncpu] for i in range(ncpu)]

                with Pool(ncpu) as executor:
                    calculate_msd_func = partial(calculate_mcd_msd, tgt_metric='msd')
                    idx2msd_dict_list = executor.map(calculate_msd_func, func_args)
                    idx2msd_list = []
                    for idx2msd_dict in idx2msd_dict_list:
                        idx2msd_list += list(idx2msd_dict.items())
                save_results(idx2result_list=idx2msd_list, metric_name='msd', save_path=save_path,
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
                idx2wav_len_ratio = {key: hypo_idx2wav_len[key] / refer_idx2wav_len[key]
                                     for key in hypo_idx2wav_len.keys()}
                idx2wav_len_ratio_list = list(idx2wav_len_ratio.items())

                save_results(idx2result_list=idx2wav_len_ratio_list, metric_name='wav_len_ratio',
                             save_path=save_path, vocoder_name=vocoder_name, desec_sort=True, topn_num=topn_num)
        print("\n")


if __name__ == '__main__':
    args = parse()
    main(**vars(args))
