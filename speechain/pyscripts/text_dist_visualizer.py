import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

from collections import Counter

from speechain.utilbox.data_loading_util import load_idx2data_file
from speechain.utilbox.text_util import text2word_list
from speechain.utilbox.import_util import parse_path_args

from speechain.tokenizer.char import CharTokenizer
from speechain.tokenizer.g2p import GraphemeToPhonemeTokenizer
from speechain.tokenizer.sp import SentencePieceTokenizer

from speechain.snapshooter import HistPlotter


def parse():
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--dump_path', type=str, default=None,
                        help="The source folder where your target files are placed.")
    parser.add_argument('--plot_path', type=str, default=None,
                        help="The source folder where your target files are placed.")
    parser.add_argument('--source_dataset', type=str, required=True,
                        help="The source folder where your target files are placed.")
    parser.add_argument('--source_subset', type=str, required=True,
                        help="The target path you want to save the summary file. "
                             "If not given, the summary file will be saved to the parent directory of 'src_folder'.")
    parser.add_argument('--token_type', type=str, required=True,
                        help="The target path you want to save the summary file. "
                             "If not given, the summary file will be saved to the parent directory of 'src_folder'.")
    parser.add_argument('--token_num', type=str, required=True,
                        help="The target path you want to save the summary file. "
                             "If not given, the summary file will be saved to the parent directory of 'src_folder'.")
    parser.add_argument('--txt_format', type=str, default='asr',
                        help="The target path you want to save the summary file. "
                             "If not given, the summary file will be saved to the parent directory of 'src_folder'.")
    parser.add_argument('--target_dataset', type=str, default=None,
                        help="The source folder where your target files are placed. (default: None)")
    parser.add_argument('--target_subset', type=str, required=True,
                        help="The source folder where your target files are placed. (default: None)")
    return parser.parse_args()


def main(source_dataset: str, source_subset: str, token_type: str, token_num: str, target_subset: str,
         dump_path: str = None, plot_path: str = None, txt_format: str = 'asr', target_dataset: str = None):

    if target_dataset is None:
        target_dataset = source_dataset

    if dump_path is None:
        dump_path = 'datasets'
    token_path = os.path.join(dump_path, source_dataset, 'data', token_type, source_subset, token_num, txt_format)

    if plot_path is None:
        plot_path = parse_path_args('tmp_figures')
    os.makedirs(plot_path, exist_ok=True)

    if token_type == 'char':
        tokenizer = CharTokenizer(token_path=token_path)
    elif token_type in ['g2p', 'mfa']:
        tokenizer = GraphemeToPhonemeTokenizer(token_path=token_path)
    elif token_type == 'sentencepiece':
        tokenizer = SentencePieceTokenizer(token_path=token_path)
    else:
        raise ValueError(f'Unknown token_type: {token_type}!')

    source_text_path = os.path.join(dump_path, source_dataset, 'data', 'wav', source_subset, f'idx2{txt_format}_text')
    source_text = load_idx2data_file(source_text_path)
    source_tokens, source_words = [], []
    for text in source_text.values():
        source_tokens += tokenizer.text2tensor(text, return_tensor=False)
        source_words += text2word_list(text)
    # collect the occurrence frequency of each token and word
    source_token2freq = Counter(source_tokens)
    source_freq = [source_token2freq[token] if token in source_token2freq.keys() else 0 for token in tokenizer.idx2token.keys()]
    source_freq = np.array(source_freq) / max(source_freq)
    source_words = set(source_words)

    target_text_path = os.path.join(dump_path, target_dataset, 'data', 'wav', target_subset, f'idx2{txt_format}_text')
    target_text = load_idx2data_file(target_text_path)
    target_tokens, target_words = [], []
    for text in target_text.values():
        target_tokens += tokenizer.text2tensor(text, no_sos=True, no_eos=True, return_tensor=False)
        target_words += text2word_list(text)
    # collect the occurrence frequency of each token and word
    target_token2freq = Counter(target_tokens)
    target_freq = [target_token2freq[token] if token in target_token2freq.keys() else 0 for token in tokenizer.idx2token.keys()]
    target_freq = np.array(target_freq) / max(target_freq)
    target_words = set(target_words)

    print(f"Total {len(target_words)} in {target_subset} of {target_dataset}: "
          f"{len(target_words.difference(source_words))} words ({len(target_words.difference(source_words)) / len(target_words):.2%}) don't exist in {source_subset} of {source_dataset}!")
    print(f"MSE between normalized token-frequency distribution ({txt_format}-{token_type}-{token_num}) between {source_subset} of {source_dataset} and {target_subset} of {target_dataset}: "
          f"{np.mean(np.square(source_freq - target_freq)):.2e}.")

    hist_plotter = HistPlotter()
    # setup the figure plotter and initialize the figure
    fig = plt.figure(figsize=[12.8, 4.8], num=2)


    # return the necessary information for the specific plotter to plot
    ax = fig.add_subplot(121)
    hist_plotter.plot(ax=ax, material=source_tokens, fig_name=f'{source_subset} of {source_dataset}',
                      xlabel='token ids', ylabel='frequency')

    ax = fig.add_subplot(122)
    hist_plotter.plot(ax=ax, material=target_tokens, fig_name=f'{target_subset} of {target_dataset}',
                      xlabel='token ids', ylabel='frequency')

    # save the plotted figure
    fig.tight_layout()
    plt.savefig(os.path.join(plot_path, f'{txt_format}-{token_type}-{token_num}_{source_dataset}-{source_subset}_vs_{target_dataset}-{target_subset}'))
    plt.close(fig=fig)


if __name__ == '__main__':
    # args = parse()
    # main(**vars(args))

    main(source_dataset='librispeech',
         source_subset='train-clean-100',
         token_type='sentencepiece',
         token_num='bpe1k',
         target_subset='train-clean-360')

    main(source_dataset='librispeech',
         source_subset='train-clean-100',
         token_type='sentencepiece',
         token_num='bpe1k',
         target_dataset='libritts',
         target_subset='train-clean-100')

    main(source_dataset='librispeech',
         source_subset='train-clean-100',
         token_type='sentencepiece',
         token_num='bpe1k',
         target_dataset='libritts',
         target_subset='train-clean-360')

    main(source_dataset='librispeech',
         source_subset='train-clean-100',
         token_type='sentencepiece',
         token_num='bpe5k',
         target_subset='train-clean-360')

    main(source_dataset='librispeech',
         source_subset='train-clean-100',
         token_type='sentencepiece',
         token_num='bpe5k',
         target_dataset='libritts',
         target_subset='train-clean-100')

    main(source_dataset='librispeech',
         source_subset='train-clean-100',
         token_type='sentencepiece',
         token_num='bpe5k',
         target_dataset='libritts',
         target_subset='train-clean-360')
