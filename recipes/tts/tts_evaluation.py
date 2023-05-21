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
from speechain.utilbox.feat_util import convert_wav_to_mfcc, convert_wav_to_logmel, convert_wav_to_pitch
from speechain.utilbox.md_util import get_list_strings


def parse():
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--hypo_path', type=str, required=True,
                        help="The path of your TTS experimental folder. All the files named 'idx2xxx_wav' will be "
                            "automatically found out and used for TTS objective evaluation. You can also directly "
                            "specify the path of your target 'idx2xxx_wav' file by this argument.")
    parser.add_argument('--refer_path', type=str, required=True,
                        help="The path of the ground-truth data folder. All the files named 'idx2wav' will be "
                             "automatically found out and used as the reference. The hypo 'idx2xxx_wav' and refer "
                             "'idx2wav' will be matched by the data indices. You can also directly specify the path of "
                             "your target 'idx2wav' file by this argument. This argument is required if you want to "
                             "evaluate the MCD or MSD.")
    parser.add_argument('--metric_list', type=str2list, default=['mcd', 'log-f0'],
                        help="The list of metrics you want to use to evaluate your given hypothesis utterances. "
                             "Please give this argument as a string surrounded by a pair of square brackets where your "
                             "metrics are split by commas. (default: [mcd, log-f0])")
    parser.add_argument('--result_path', type=str, default=None,
                        help="The path where the evaluated results are placed. If not given, the results will be saved "
                             "to the same directory as the hypo 'idx2xxx_wav' found in 'hypo_path'. (default: None)")
    parser.add_argument('--ncpu', type=int, default=8,
                        help="The number of processes used to calculate the evaluation metrics. (default: 8)")
    parser.add_argument('--topn_num', type=int, default=30,
                        help="The number of top-n bad cases you want to show in the result md file. (default: 30)")
    return parser.parse_args()


def calculate_metric(idx2hypo_refer_list: List[str], tgt_metric: str = 'mcd'):
    """
        This function calculates a given metric (MCD, MSD, or Log-F0) for each pair of hypothesis-reference audio files.

        Args:
            idx2hypo_refer_list (List[str]):
                A list of triples containing the index, the path to the hypothesis audio file, and the path to the reference audio file.
            tgt_metric (str, optional):
                The target metric to be calculated. It can be 'mcd', 'msd', or 'log-f0'. Defaults to 'mcd'.

        Returns:
            output (dict):
                A dictionary mapping from the index to the calculated metric for each pair of audio files.
    """
    tgt_metric = tgt_metric.lower()
    assert tgt_metric in ['mcd', 'msd', 'log-f0'], \
        f"tgt_metric must be one of ['mcd', 'msd', 'log-f0'], but got {tgt_metric}!"

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
        elif tgt_metric == 'msd':
            hypo_feat = convert_wav_to_logmel(hypo_wav, sr=hypo_sample_rate,
                                              n_mels=80, win_length=0.05, hop_length=0.0125)
            refer_feat = convert_wav_to_logmel(refer_wav, sr=refer_sample_rate,
                                               n_mels=80, win_length=0.05, hop_length=0.0125)
        # extract the pitch contours from the waveforms
        elif tgt_metric == 'log-f0':
            hypo_feat = convert_wav_to_pitch(hypo_wav, sr=hypo_sample_rate, hop_length=0.0125, continuous_f0=False)
            refer_feat = convert_wav_to_pitch(refer_wav, sr=refer_sample_rate, hop_length=0.0125, continuous_f0=False)
        else:
            raise NotImplementedError

        # DTW calculation
        dtw_path = fastdtw(hypo_feat, refer_feat, dist=euclidean)[1]
        hypo_feat = hypo_feat[[ele[0] for ele in dtw_path]]
        refer_feat = refer_feat[[ele[1] for ele in dtw_path]]

        if tgt_metric != 'log-f0':
            coeff = 10 / np.log(10) * np.sqrt(2)
            output[idx] = coeff * np.mean(np.sqrt(np.sum((hypo_feat - refer_feat) ** 2, axis=1)))
        else:
            try:
                # Get voiced part of log-f0
                nonzero_idxs = np.where((hypo_feat != 0) & (refer_feat != 0))[0]
                hypo_feat = np.log(hypo_feat[nonzero_idxs])
                refer_feat = np.log(refer_feat[nonzero_idxs])
                # calculate rmse
                rmse = np.sqrt(np.mean((hypo_feat - refer_feat) ** 2))
            except RuntimeWarning:
                rmse = None

            if rmse is not None and not np.isnan(rmse):
                output[idx] = rmse

    return output


def save_results(idx2metric_list: List[List], metric_name: str, save_path: str, vocoder_name: str,
                 desec_sort: True, topn_num: int = 30):
    """
        This function saves the results of the calculated metrics into a file.

        Args:
            idx2metric_list (List[List]):
                A list of pairs containing the index and the calculated metric.
            metric_name (str):
                The name of the calculated metric.
            save_path (str):
                The path where to save the results.
            vocoder_name (str):
                The name of the vocoder used.
            desec_sort (bool):
                If True, the results are sorted in descending order.
            topn_num (int, optional):
                The number of top bad cases to record. Defaults to 30.
    """
    # save the idx2feat file to feat_path as the reference
    np.savetxt(os.path.join(save_path, f'idx2{f"{vocoder_name}_" if vocoder_name is not None else ""}{metric_name}'),
               sorted(idx2metric_list, key=lambda x: x[0]), fmt='%s')

    # record the overall results
    result_mean = np.mean([idx2result[1] for idx2result in idx2metric_list])
    result_std = np.std([idx2result[1] for idx2result in idx2metric_list])
    md_report = f"# Overall {metric_name} Result: (mean ± std)\n" \
                f"{result_mean:.4f} ± {result_std:.4f}\n" \
                f"# Top{topn_num} Bad Cases for {metric_name}\n"

    # record the data instances with the top-n largest results
    idx2metric_list = sorted(idx2metric_list, key=lambda x: x[1], reverse=desec_sort)[: topn_num]
    idx2result_dict = {idx: f"{mcd:.4f}" for idx, mcd in idx2metric_list}
    md_report += get_list_strings(idx2result_dict)
    np.savetxt(
        os.path.join(save_path, f'{f"{vocoder_name}_" if vocoder_name is not None else ""}{metric_name}_results.md'),
        [md_report], fmt='%s')


def main(hypo_path: str, refer_path: str, metric_list: List[str], result_path: str = None, ncpu: int = 8, topn_num: int = 30):
    """
        This is the main function that organizes the calculation of the metrics for the audio files.
        It first prepares the paths to the audio files, then for each metric in the list, it calculates
        the metric and saves the results. Finally, it performs a waveform length ratio evaluation.

        Args:
            hypo_path (str):
                The path to the hypothesis audio files.
            refer_path (str):
                The path to the reference audio files.
            metric_list (List[str]):
                The list of metrics to be calculated.
            result_path (str, optional):
                The path where to save the results. If not provided, the results are saved in the same directory as
                the hypothesis audio files.
            ncpu (int, optional):
                The number of cores to use for parallel processing. Defaults to 8.
            topn_num (int, optional):
                The number of top bad cases to record. Defaults to 30.
    """
    # --- 1. Argument Preparation stage --- #
    # argument checking
    for i in range(len(metric_list)):
        metric_list[i] = metric_list[i].lower()
        assert metric_list[i] in ['mcd', 'msd', 'log-f0'],\
            f"Your input metric should be one of ['mcd', 'msd', 'log-f0'], but got {metric_list[i]}!"

        if metric_list[i] in ['mcd', 'msd', 'log-f0']:
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
            refer_idx2wav_list = search_file_in_subfolder(
                refer_path, lambda x: x.startswith('idx2') and x.endswith('wav'))
        # for file input, directly use it as hypo_idx2wav
        else:
            refer_idx2wav_list = [refer_path]
    else:
        refer_idx2wav_list = None

    for hypo_idx2wav in hypo_idx2wav_list:
        # automatically initialize save_path and vocoder_name
        save_path = os.path.dirname(hypo_idx2wav) if result_path is None else parse_path_args(result_path)
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

            idx2hypo_refer_list = [[idx, hypo_idx2wav[idx], refer_idx2wav[idx]] for idx in hypo_idx2wav.keys()]
            func_args = [idx2hypo_refer_list[i::ncpu] for i in range(ncpu)]

            with Pool(ncpu) as executor:
                calculate_metric_func = partial(calculate_metric, tgt_metric=metric)
                idx2metric_dict_list = executor.map(calculate_metric_func, func_args)
                idx2metric_list = []
                for idx2metric_dict in idx2metric_dict_list:
                    idx2metric_list += list(idx2metric_dict.items())
            save_results(idx2metric_list=idx2metric_list, metric_name=metric, save_path=save_path,
                         vocoder_name=vocoder_name, desec_sort=True, topn_num=topn_num)

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

                save_results(idx2metric_list=idx2wav_len_ratio_list, metric_name='wav_len_ratio',
                             save_path=save_path, vocoder_name=vocoder_name, desec_sort=True, topn_num=topn_num)
        print("\n")


if __name__ == '__main__':
    args = parse()
    main(**vars(args))
