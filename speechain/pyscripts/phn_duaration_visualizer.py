from collections import Counter
from tqdm import tqdm

import argparse
import os
import matplotlib.pyplot as plt

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
    parser.add_argument('--mfa_model', type=str, required=True,
                        help="The target path you want to save the summary file. "
                             "If not given, the summary file will be saved to the parent directory of 'src_folder'.")
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help="The sampling rate of your target waveforms. (default: 16000)")
    parser.add_argument('--hop_len', type=int or float, default=256,
                        help="The sampling rate of your target waveforms. (default: 16000)")
    return parser.parse_args()



def main(dataset: str, subset: str, mfa_model: str,
         sample_rate: int = 16000, hop_len: int or float = 256, dump_path: str = None, plot_path: str = None):
    if dump_path is None:
        dump_path = parse_path_args('datasets')

    idx2wav_len_path = os.path.join(dump_path, dataset, 'data', f'wav{sample_rate}', subset, 'idx2wav_len')
    if not os.path.exists(idx2wav_len_path):
        idx2wav_len_path = os.path.join(dump_path, dataset, 'data', 'wav', subset, 'idx2wav_len')
    idx2wav_len = load_idx2data_file(idx2wav_len_path, data_type=int)

    idx2text = load_idx2data_file(os.path.join(dump_path, f'{dataset}/data/mfa/{mfa_model}/{subset}/idx2text'))
    idx2duration = load_idx2data_file(
        os.path.join(dump_path, f'{dataset}/data/mfa/{mfa_model}/{subset}/idx2duration'))

    total_duration_list, space_duration_list = [], []
    hop_len = int(hop_len * sample_rate) if isinstance(hop_len, float) else hop_len
    for idx in tqdm(idx2wav_len.keys()):
        if idx not in idx2duration.keys():
            continue
        feat_len = idx2wav_len[idx] / hop_len
        duration = [float(d) for d in idx2duration[idx][1:-1].split(', ')]
        duration = [round(d / sum(duration) * feat_len) for d in duration]
        total_duration_list += duration

        text = [phn[1:-1] for phn in idx2text[idx][1:-1].split(', ')]
        space_duration = [duration[i] for i, phn in enumerate(text) if phn == '<space>' and i not in [0, len(text) - 1]]
        space_duration_list += space_duration

    if plot_path is None:
        plot_path = parse_path_args('tmp_figures')
    os.makedirs(plot_path, exist_ok=True)

    hist_plotter = HistPlotter()
    # setup the figure plotter and initialize the figure
    fig = plt.figure(figsize=[12.8, 4.8], num=2)

    # return the necessary information for the specific plotter to plot
    ax = fig.add_subplot(121)
    hist_plotter.plot(ax=ax, material=total_duration_list, fig_name='duration distribution of all the tokens',
                      xlabel=f'Num of frames '
                             f'(min={min(total_duration_list)}, '
                             f'mean={int(sum(total_duration_list) / len(total_duration_list))}, '
                             f'max={max(total_duration_list)})', ylabel='frequency')

    ax = fig.add_subplot(122)
    hist_plotter.plot(ax=ax, material=space_duration_list, fig_name='duration distribution of all the <space> token',
                      xlabel='Num of frames '
                             f'(min={min(space_duration_list)}, '
                             f'mean={int(sum(space_duration_list) / len(space_duration_list))}, '
                             f'max={max(space_duration_list)})', ylabel='frequency')

    # save the plotted figure
    plt.savefig(os.path.join(
        plot_path, f'duration-dist_mfa={mfa_model}_data={dataset}-{subset}_sr={sample_rate}_hop={hop_len}'))
    plt.close(fig=fig)


if __name__ == '__main__':
    # args = parse()
    # main(**vars(args))

    main(
        dataset='librispeech', subset='train-clean-100', mfa_model='librispeech_train-clean-100'
    )
