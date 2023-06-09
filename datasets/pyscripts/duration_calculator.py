"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2023.02
"""
import argparse
import os
from typing import Dict, List

import textgrid as tg
import numpy as np

from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from collections import Counter

from speechain.utilbox.import_util import parse_path_args
from speechain.utilbox.data_loading_util import load_idx2data_file, search_file_in_subfolder
from speechain.utilbox.type_util import str2bool, str2none


def parse():
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--data_path', type=str, required=True,
                        help="The path where you place the dumped data.")
    parser.add_argument('--save_path', type=str, default=None,
                        help="The path where you want to save the duration metadata files. If not given, the files will be saved to {data_path}/mfa. (default: None)")
    parser.add_argument('--save_folder_name', type=str, required=True,
                        help="The name of the pretrained model you have used to get the .TextGrid files.")
    parser.add_argument('--retain_punc', type=str2bool, default=False,
                        help="Whether to retain the punctuation marks in the phoneme sequences. (default: False)")
    parser.add_argument('--retain_stress', type=str2bool, default=True,
                        help="Whether to retain the stress indicators at the end of each vowel phonemes. (default: True)")
    parser.add_argument('--dataset_name', type=str, required=True,
                        help="The name of the dataset you want to process.")
    parser.add_argument('--subset_name', type=str2none, default=None,
                        help="The name of the subset in your given dataset you want to process. (default: None)")
    parser.add_argument("--ncpu", type=int, default=8,
                        help="The number of processes you want to use to calculate the phoneme duration. (default: 8)")
    return parser.parse_args()


def convert_phns_to_vocab(phn_list: List[str]):
    """
        Converts a list of phonemes to a vocabulary list, sorted by frequency.

        Args:
            phn_list (List[str]):
                A list of phonemes.

        Returns:
            List[str]:
                A list of unique phonemes sorted by frequency, with additional tokens for special purposes.

        This function calculates the frequency of each phoneme in the input list, sorts them in descending order, and
        returns a new list with the unique phonemes along with some special tokens for compatibility with
        autoregressive TTS models.
    """
    # Collect the occurrence frequency of each phoneme
    phn2freq = sorted(Counter(phn_list).items(), key=lambda x: x[1], reverse=True)

    # Extract only the phonemes from the frequency list
    phn_list = [phn for phn, _ in phn2freq]

    # Remove special tokens: '<blank>', '<unk>', and '<sos/eos>' if present, as they will be added later
    if '<unk>' in phn_list:
        phn_list.remove('<unk>')
    if '<blank>' in phn_list:
        phn_list.remove('<blank>')
    if '<sos/eos>' in phn_list:
        phn_list.remove('<sos/eos>')

    # Add special tokens: '<blank>', '<unk>', and '<sos/eos>'
    # <sos/eos> is added here for the compatibility with autoregressive TTS model
    return ["<blank>"] + phn_list + ['<unk>', '<sos/eos>']


def dump_subset_metadata(dataset_path: str, save_path: str, subset_name: str, retain_punc: bool, retain_stress: bool,
                         idx2text: Dict, idx2duration: Dict):
    """
        Save the subset-specific idx2text and idx2duration dictionaries as metadata files.

        Args:
            dataset_path (str):
                The path to the dataset directory.
            save_path (str):
                The path to save the metadata files.
            subset_name (str):
                The name of the subset to process.
            retain_stress (bool):
                Whether to retain the stress indicators at the end of each vowel phonemes.
            idx2text (Dict):
                The dictionary mapping file indices to tokenized text.
            idx2duration (Dict):
                The dictionary mapping file indices to phoneme durations.
    """

    # get the subset-specific idx2text and idx2duration Dicts
    subset_indices = list(load_idx2data_file(os.path.join(dataset_path, subset_name, 'idx2wav')).keys())
    subset_idx2text = {index: idx2text[index] for index in subset_indices if index in idx2text.keys()}
    subset_idx2duration = {index: idx2duration[index] for index in subset_indices if index in idx2duration.keys()}
    subset_path = os.path.join(save_path, subset_name, 'stress' if retain_stress else 'no-stress',
                               'punc' if retain_punc else 'no-punc')

    # --- idx2text & idx2text_len Saving --- #
    os.makedirs(subset_path, exist_ok=True)
    # save the subet-specific idx2text Dict to a metadata file
    text_path = os.path.join(subset_path, 'idx2text')
    np.savetxt(text_path, [[idx, str(text)] for idx, text in subset_idx2text.items()], fmt="%s")
    print(f"Tokenized text has been successfully saved to {text_path}.")

    # save the length information of the subet-specific idx2text Dict to a metadata file
    text_len_path = os.path.join(subset_path, 'idx2text_len')
    np.savetxt(text_len_path, [[idx, len(text)] for idx, text in subset_idx2text.items()], fmt="%s")
    print(f"The length of tokenized text has been successfully saved to {text_len_path}.")

    # --- vocab Saving --- #
    subset_phns = []
    for text in subset_idx2text.values():
        subset_phns += text
    subset_phn_vocab = convert_phns_to_vocab(subset_phns)
    vocab_path = os.path.join(subset_path, 'vocab')
    np.savetxt(vocab_path, subset_phn_vocab, fmt="%s")
    print(f"Phoneme vocabulary has been successfully saved to {vocab_path}.")

    # --- idx2duration Saving --- #
    os.makedirs(subset_path, exist_ok=True)
    # save the subet-specific idx2duration Dict to a metadata file
    duration_path = os.path.join(subset_path, 'idx2duration')
    np.savetxt(duration_path, [[idx, str(duration)] for idx, duration in subset_idx2duration.items()], fmt="%s")
    print(f"The duration of tokenized text has been successfully saved to {duration_path}.")

def cal_duration_by_tg(tg_file_list: List[str], idx2punc_text: Dict[str, str], retain_stress: bool):
    """
        Calculate the phoneme duration for each TextGrid file and return idx2text and idx2duration dictionaries.

        Args:
            tg_file_list (List[str]):
                List of paths to TextGrid files.
            retain_stress (bool):
                Whether to retain the stress indicators at the end of each vowel phonemes.

        Returns:
            Tuple[Dict, Dict]:
                A tuple containing two dictionaries:
                    1. idx2text: mapping file indices to tokenized text
                    2. idx2duration: mapping file indices to phoneme durations
    """
    idx2text, idx2duration = {}, {}
    for tg_file in tqdm(tg_file_list):
        file_name = os.path.basename(tg_file).split('.')[0]
        idx2text[file_name], idx2duration[file_name] = [], []

        text_grid = tg.TextGrid.fromFile(tg_file)
        # whether a time point is a word boundary that divide two words
        # 'sp' and 'sil' are treated as words here (which are meant to represent short silence)
        word_boundary_dict = {round(word.maxTime, 2): True if word.mark != '' else False
                              for word in text_grid.tiers[0].intervals}
        for phn in text_grid.tiers[1].intervals:
            # ensure the accuracy of the float numbers for data storage
            phn_duration, phn_boundary = round(phn.maxTime - phn.minTime, 2), round(phn.maxTime, 2)

            # for the non-silence phoneme tokens, repeating of non-silence phonemes is retained
            if phn.mark not in ['sp', '', 'sil']:
                # remove the stress number at the end if retain_stress is set to False
                if not retain_stress and phn.mark[-1].isdigit():
                    phn_token = phn.mark[:-1]
                else:
                    phn_token = phn.mark

                idx2text[file_name].append(phn_token if phn_token != 'spn' else '<unk>')
                idx2duration[file_name].append(phn_duration)
            # for the silence tokens
            else:
                # when encounter a new space token, record it
                if len(idx2text[file_name]) == 0 or idx2text[file_name][-1] != '<space>':
                    idx2text[file_name].append('<space>')
                    idx2duration[file_name].append(phn_duration)
                # for the repeated space tokens, add them into a large space
                else:
                    idx2duration[file_name][-1] += phn_duration

            # add a 0-duration space when meeting a word boundary
            if phn_boundary in word_boundary_dict.keys() and word_boundary_dict[phn_boundary]:
                idx2text[file_name].append('<space>')
                idx2duration[file_name].append(round(0, 2))

        # remove the utterances that only contains a space token
        if len(idx2text[file_name]) == 1 and idx2text[file_name][0] == '<space>':
            idx2text.pop(file_name)
            idx2duration.pop(file_name)
        # add punctuation marks in phoneme sequence with 0 duration
        else:
            punc_text = idx2punc_text[file_name] if idx2punc_text is not None else None
            if punc_text is not None:
                word_list = punc_text.split(' ')

                _tmp_text, _tmp_duration = [], []
                curr_phn_idx = 0
                if idx2text[file_name][curr_phn_idx] == '<space>':
                    _tmp_text.append(idx2text[file_name][curr_phn_idx])
                    _tmp_duration.append(idx2duration[file_name][curr_phn_idx])
                    curr_phn_idx += 1

                for word in word_list:
                    try:
                        while idx2text[file_name][curr_phn_idx] != '<space>':
                            _tmp_text.append(idx2text[file_name][curr_phn_idx])
                            _tmp_duration.append(idx2duration[file_name][curr_phn_idx])

                            curr_phn_idx += 1
                            if curr_phn_idx == len(idx2text[file_name]):
                                break
                    except IndexError:
                        raise IndexError(f"file_name={file_name}")

                    if not word[-1].isalpha():
                        _tmp_text.append(word[-1])
                        _tmp_duration.append(round(0, 2))

                    try:
                        if idx2text[file_name][curr_phn_idx] == '<space>' and curr_phn_idx < len(idx2text[file_name]):
                            _tmp_text.append(idx2text[file_name][curr_phn_idx])
                            _tmp_duration.append(idx2duration[file_name][curr_phn_idx])
                            curr_phn_idx += 1
                    except IndexError:
                        raise IndexError(f"file_name={file_name}")

                idx2text[file_name] = _tmp_text
                idx2duration[file_name] = _tmp_duration

    return idx2text, idx2duration


def combine_subsets(subset_path_list: List[str], new_subset_name: str, save_path: str):
    """
        Combines the metadata of multiple subsets into a single new subset.

        Args:
            subset_path_list (List[str]):
                A list of paths to the individual subset directories.
            new_subset_name (str):
                The name of the new combined subset directory.
            save_path (str):
                The path where the new combined subset directory will be saved.

        This function computes the intersection of directories from the input subsets, merges their metadata,
        calculates the phoneme vocabulary, and saves the new combined metadata into the specified save path.
    """
    def get_folder_intersection(dir_path_list: List[str]):
        # Get directory content for each subset path
        dir_content_list = [os.listdir(dir_path) for dir_path in dir_path_list]

        # Calculate the intersection of directories
        exclu_folder_set = set(dir_content_list[0])
        for lst in dir_content_list[1:]:
            exclu_folder_set.intersection_update(set(lst))

        return list(exclu_folder_set)

    # Initialize metadata dictionary
    meta_data_dict = {first_level_folder:
                          {second_level_folder: {'idx2duration': {}, 'idx2text': {}, 'idx2text_len': {}}
                           for second_level_folder in get_folder_intersection([os.path.join(subset_path, first_level_folder) for subset_path in subset_path_list])}
                      for first_level_folder in get_folder_intersection(subset_path_list)}

    # Merge metadata from all subsets
    for first_level_folder in meta_data_dict.keys():
        for second_level_folder in meta_data_dict[first_level_folder].keys():
            for subset_path in subset_path_list:
                curr_metadata_path = os.path.join(subset_path, first_level_folder, second_level_folder)

                for metadata in os.listdir(curr_metadata_path):
                    if not metadata.startswith('idx2'):
                        continue
                    meta_data_dict[first_level_folder][second_level_folder][metadata].update(load_idx2data_file(os.path.join(curr_metadata_path, metadata)))

            # Calculate the phoneme vocabulary
            subset_phns = []
            for text in meta_data_dict[first_level_folder][second_level_folder]['idx2text'].values():
                subset_phns += [phoneme[1:-1] for phoneme in text[1:-1].split(', ')]
            meta_data_dict[first_level_folder][second_level_folder]['vocab'] = convert_phns_to_vocab(subset_phns)

    # Save the new combined metadata
    new_subset_path = os.path.join(save_path, new_subset_name)
    for first_level_folder in meta_data_dict.keys():
        for second_level_folder in meta_data_dict[first_level_folder].keys():
            folder_path = os.path.join(new_subset_path, first_level_folder, second_level_folder)
            os.makedirs(folder_path, exist_ok=True)

            for metadata in meta_data_dict[first_level_folder][second_level_folder].keys():
                metadata_path = os.path.join(folder_path, metadata)

                if isinstance(meta_data_dict[first_level_folder][second_level_folder][metadata], Dict):
                    np.savetxt(metadata_path, [[idx, str(m_d)] for idx, m_d in meta_data_dict[first_level_folder][second_level_folder][metadata].items()],
                               fmt="%s")
                else:
                    np.savetxt(metadata_path, meta_data_dict[first_level_folder][second_level_folder][metadata], fmt="%s")

                print(f"{metadata} has successfully been saved to {metadata_path}!")


def main(data_path: str, save_folder_name: str, dataset_name: str, subset_name: str = None,
         retain_punc: bool = False, retain_stress: bool = True, save_path: str = None, ncpu: int = 8):
    """
        Main function to process TextGrid files and save metadata files for a given dataset and subset.

        Args:
            data_path (str):
                The path where you placed the dumped data.
            save_folder_name (str):
                The name of the pretrained model you have used to get the .TextGrid files.
            retain_stress: bool
                Whether to retain the stress indicators at the end of each vowel phonemes.
            dataset_name (str):
                The name of the dataset you want to process.
            subset_name (str):
                The name of the subset in your given dataset you want to process.
            save_path (str, optional):
                The path where you want to save the duration metadata files.
                If not given, the files will be saved to {data_path}/mfa. Defaults to None.
            ncpu (int, optional):
                The number of processes you want to use to calculate the phoneme duration. Defaults to 8.
    """

    data_path = parse_path_args(data_path)
    if save_path is None:
        save_path = os.path.join(data_path, 'mfa', save_folder_name)

    # initialize the textgrid information and calculate the duration (in seconds)
    textgrid_path = os.path.join(data_path, 'mfa', save_folder_name, 'TextGrid')
    if subset_name is not None:
        textgrid_path = os.path.join(textgrid_path, subset_name)

    print(f'Start to summarize all the .TextGrid files in {textgrid_path}......')
    tg_file_list = search_file_in_subfolder(textgrid_path, tgt_match_fn=lambda x: x.endswith('.TextGrid'))
    if len(tg_file_list) == 0:
        raise RuntimeError(f".TextGrid files have not been successfully saved to {textgrid_path}!")

    if retain_punc:
        if subset_name is None:
            _search_path = os.path.join(data_path, 'wav')
        else:
            _search_path = os.path.join(data_path, 'wav', subset_name)

        idx2punc_text = load_idx2data_file(
            search_file_in_subfolder(_search_path, tgt_match_fn=lambda x: x == 'idx2punc_text'))
        assert len(idx2punc_text) > 0
    else:
        idx2punc_text = None

    func_args = [tg_file_list[i::ncpu] for i in range(ncpu)]
    cal_duration_by_tg_func = partial(cal_duration_by_tg, idx2punc_text=idx2punc_text, retain_stress=retain_stress)
    # start the executing jobs
    with Pool(ncpu) as executor:
        text_duration_results = executor.map(cal_duration_by_tg_func, func_args)

    # gather the results from all the processes
    idx2text, idx2duration = {}, {}
    for _idx2text, _idx2duration in text_duration_results:
        idx2text.update(_idx2text)
        idx2duration.update(_idx2duration)

    # dump the vocabulary and duration metadata files
    dataset_path = os.path.join(data_path, 'wav')
    # process all the subsets in the given dataset
    if subset_name is None:
        for file_name in os.listdir(textgrid_path):
            # skip the non-directory file
            if os.path.isfile(os.path.join(textgrid_path, file_name)):
                continue
            dump_subset_metadata(dataset_path=dataset_path, save_path=save_path, subset_name=file_name,
                                 retain_punc=retain_punc, retain_stress=retain_stress, idx2text=idx2text,
                                 idx2duration=idx2duration)
    else:
        dump_subset_metadata(dataset_path=dataset_path, save_path=save_path, subset_name=subset_name,
                             retain_punc=retain_punc, retain_stress=retain_stress, idx2text=idx2text,
                             idx2duration=idx2duration)

    # post-processing: combine specific subsets into a larger one
    if dataset_name == 'libritts':
        # train-clean-460 = train-clean-100 + train-clean-360
        clean_100_path = os.path.join(save_path, 'train-clean-100')
        clean_360_path = os.path.join(save_path, 'train-clean-360')
        if os.path.exists(clean_100_path) and os.path.exists(clean_360_path):
            combine_subsets(subset_path_list=[clean_100_path, clean_360_path],
                            new_subset_name='train-clean-460', save_path=save_path)

            # train-960 = train-clean-460 + train-other-500
            clean_460_path = os.path.join(save_path, 'train-clean-460')
            other_500_path = os.path.join(save_path, 'train-other-500')
            if os.path.exists(clean_460_path) and os.path.exists(other_500_path):
                combine_subsets(subset_path_list=[clean_460_path, other_500_path],
                                new_subset_name='train-960', save_path=save_path)

        # dev = dev-clean + dev-other
        dev_clean_path = os.path.join(save_path, 'dev-clean')
        dev_other_path = os.path.join(save_path, 'dev-other')
        if os.path.exists(dev_clean_path) and os.path.exists(dev_other_path):
            combine_subsets(subset_path_list=[dev_clean_path, dev_other_path],
                            new_subset_name='dev', save_path=save_path)


if __name__ == '__main__':
    args = parse()
    main(**vars(args))
