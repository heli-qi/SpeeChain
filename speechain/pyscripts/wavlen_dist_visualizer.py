import argparse
import math
import os
import matplotlib.pyplot as plt
from collections import Counter

from speechain.utilbox.data_loading_util import load_idx2data_file
from speechain.utilbox.import_util import parse_path_args
from speechain.snapshooter import HistPlotter


def parse():
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--dump_path', type=str, default=None,
                        help="The source folder where your target files are placed.")
    parser.add_argument('--plot_path', type=str, default=None,
                        help="The source folder where your target files are placed.")
    parser.add_argument('--dataset', type=str, required=True,
                        help="The source folder where your target files are placed.")
    parser.add_argument('--subset', type=str, required=True,
                        help="The target path you want to save the summary file. "
                             "If not given, the summary file will be saved to the parent directory of 'src_folder'.")
    parser.add_argument('--token_type', type=str, required=True,
                        help="The source folder where your target files are placed.")
    parser.add_argument('--token_num', type=str, default=None,
                        help="The target path you want to save the summary file. "
                             "If not given, the summary file will be saved to the parent directory of 'src_folder'.")
    parser.add_argument('--txt_format', type=str, default='asr',
                        help="The source folder where your target files are placed.")
    parser.add_argument('--mfa_model', type=str, default='english_us_arpa',
                        help="The source folder where your target files are placed.")
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help="The sampling rate of your target waveforms. (default: 16000)")
    parser.add_argument('--hop_len', type=int or float, default=256,
                        help="The sampling rate of your target waveforms. (default: 16000)")
    return parser.parse_args()


def main(dataset: str, subset: str, sample_rate: int, token_type: str, token_num: str = None,
         txt_format: str = 'asr', mfa_model: str = 'english_us_arpa', hop_len: int or float = 256,
         dump_path: str = None, plot_path: str = None):

    if dump_path is None:
        dump_path = parse_path_args('datasets')

    idx2wav_len_path = os.path.join(dump_path, dataset, 'data', f'wav{sample_rate}', subset, 'idx2wav_len')
    if not os.path.exists(idx2wav_len_path):
        idx2wav_len_path = os.path.join(dump_path, dataset, 'data', 'wav', subset, 'idx2wav_len')
    idx2wav_len = load_idx2data_file(idx2wav_len_path, data_type=int)

    hop_len = int(hop_len * sample_rate) if isinstance(hop_len, float) else hop_len
    idx2feat_len = {idx: int(wav_len / hop_len) for idx, wav_len in idx2wav_len.items()}

    if token_type == 'mfa':
        idx2text_len = load_idx2data_file(
            os.path.join(dump_path, dataset, 'data', 'mfa', mfa_model, subset, 'idx2text_len'), data_type=int)
    else:
        idx2text_len = load_idx2data_file(
            os.path.join(dump_path, dataset, 'data', token_type, subset, token_num, txt_format, 'idx2text_len'), data_type=int)

    if plot_path is None:
        plot_path = parse_path_args('tmp_figures')
    os.makedirs(plot_path, exist_ok=True)

    hist_plotter = HistPlotter()
    # setup the figure plotter and initialize the figure
    fig = plt.figure(figsize=[6.4 * 2, 4.8 * 2], num=4)

    # return the necessary information for the specific plotter to plot
    ax = fig.add_subplot(221)
    wav_len = list(idx2wav_len.values())
    hist_plotter.plot(ax=ax, material=wav_len,
                      fig_name=f'Distribution of waveform length ({sample_rate} ps)',
                      xlabel=f'Num of sampling points '
                             f'(min={min(wav_len)}, mean={int(sum(wav_len) / len(wav_len))}, max={max(wav_len)})',
                      ylabel='Frequency')

    # return the necessary information for the specific plotter to plot
    ax = fig.add_subplot(222)
    feat_len = list(idx2feat_len.values())
    hist_plotter.plot(ax=ax, material=feat_len,
                      fig_name=f'Distribution of acoustic frame sequence lengths (hopping length: {hop_len})',
                      xlabel=f'Num of acoustic frames '
                             f'(min={min(feat_len)}, mean={int(sum(feat_len) / len(feat_len))}, max={max(feat_len)})',
                      ylabel='Frequency')

    # return the necessary information for the specific plotter to plot
    ax = fig.add_subplot(223)
    text_len = list(idx2text_len.values())
    hist_plotter.plot(ax=ax, material=list(idx2text_len.values()),
                      fig_name=f'Distribution of token sequence lengths (tokenizer: {token_type})',
                      xlabel=f'Num of tokens '
                             f'(min={min(text_len)}, mean={int(sum(text_len) / len(text_len))}, max={max(text_len)})',
                      ylabel='Frequency')

    # return the necessary information for the specific plotter to plot
    ax = fig.add_subplot(224)
    f2t_ratio = [idx2feat_len[idx] / idx2text_len[idx] for idx in idx2feat_len.keys() if idx in idx2text_len.keys()]
    min_f2t, mean_f2t, max_f2t = min(f2t_ratio), sum(f2t_ratio) / len(f2t_ratio), max(f2t_ratio)
    xlabel = 'Frame-to-token ratio'
    if max(f2t_ratio) > 100:
        f2t_ratio = [math.log(f) for f in f2t_ratio]
        xlabel += ' (log scale)'
    hist_plotter.plot(ax=ax, material=f2t_ratio,
                      fig_name='Distribution of Frame-to-token ratio',
                      xlabel=xlabel + f' (min={min_f2t:.2f}, mean={mean_f2t:.2f}, max={max_f2t:.2f})', ylabel='Frequency')

    # save the plotted figure
    tokenizer = f'{token_type}-{token_num}' if token_type != 'mfa' else f'mfa-{mfa_model}'
    plt.savefig(os.path.join(plot_path,
                             f'datalen-dist_data={dataset}-{subset}_sr={sample_rate}_hop={hop_len}_tokenizer={tokenizer}'))
    plt.close(fig=fig)

if __name__ == '__main__':
    # args = parse()
    # main(**vars(args))

    main(dataset='libritts',
         subset='train-clean-100',
         sample_rate=16000,
         token_type='mfa',
         mfa_model='librispeech_train-clean-100',
         hop_len=256)
