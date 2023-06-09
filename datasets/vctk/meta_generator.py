"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import argparse
from typing import Dict, List

import os
import re
from tqdm import tqdm

from datasets.meta_generator import SpeechTextMetaGenerator
from speechain.utilbox.dump_util import en_text_process
from speechain.utilbox.type_util import str2list
from speechain.utilbox.data_loading_util import search_file_in_subfolder


class VCTKMetaGenerator(SpeechTextMetaGenerator):
    """
        Class for generating metadata from the VCTK dataset, including specific speakers for validation and testing sets.
        This class extends from SpeechTextMetaGenerator.
    """

    @staticmethod
    def add_parse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
            Function to add argument parsing related to VCTK Dataset.

            Args:
                parser: Argument parser object.

            Returns:
                parser: Updated parser object with additional argument group for VCTK Dataset.
        """
        group = parser.add_argument_group("Specific Arguments for the VCTK Dataset.")
        group.add_argument('--valid_speakers', type=str2list, default=['p226', 'p228', 'p265', 'p241', 'p306', 'p311'],
                           help="The speakers used as the validation set. The default validation speakers are picked up "
                                "from the three most frequent accents (English, Scottish, American). For each accent, "
                                "we select a male speaker and a female speaker with the least amount of data."
                                "Default to ['p226' (English, M), 'p228' (English, F), 'p265' (Scottish, F), 'p241' (Scottish, M), 'p306' (American, F), 'p311' (American, M)]")
        group.add_argument('--test_speakers', type=str2list, default=['p225', 'p256', 'p249', 'p237', 'p362', 'p345'],
                           help="The speakers used as the testing set. The default testing speakers are picked up "
                                "from the three most frequent accents (English, Scottish, American). For each accent, "
                                "we select a male speaker and a female speaker with the least amount of data."
                                "Default to ['p225' (English, F), 'p256' (English, M), 'p249' (Scottish, F), 'p237' (Scottish, M), 'p362' (American, F), 'p345' (American, M)]")
        return parser

    def generate_meta_dict(self, src_path: str, txt_format: str, valid_speakers: List = None, test_speakers: List = None) \
            -> Dict[str, Dict[str, Dict[str, str] or List[str]]]:
        """
            Function to generate metadata dictionary from the given source path.
            We treat two microphone version of an original speaker as two speakers.

            Args:
                src_path: Source path of the dataset.
                txt_format: Text format to be used in the output metadata.
                valid_speakers: List of speaker ids to be used as validation speakers.
                test_speakers: List of speaker ids to be used as test speakers.

            Returns:
                meta_dict: A dictionary of metadata including train, valid, and test subsets.
        """

        # --- 0. Argument Checking --- #
        # Check if valid_speakers or test_speakers are None and set default values. Also check for any overlapping speakers.
        if valid_speakers is None:
            valid_speakers = ['p226', 'p228', 'p265', 'p241', 'p306', 'p311']
        if test_speakers is None:
            test_speakers = ['p225', 'p256', 'p249', 'p237', 'p362', 'p345']
        assert len(set(valid_speakers)) == len(valid_speakers) and len(set(test_speakers)) == len(test_speakers)
        assert len(set(valid_speakers).difference(set(test_speakers))) == len(valid_speakers), \
            f"Your input valid_section and test_section have overlapping elements! " \
            f"Got valid_section={valid_speakers} & test_section={test_speakers}."

        # --- 1. Text Data Reading and Recording --- #
        # Create an empty dictionary for storing metadata and loop over subsets ('train', 'valid', 'test')
        meta_dict = dict()
        for subset in ['train-mic1', 'valid-mic1', 'test-mic1', 'train-mic2', 'valid-mic2', 'test-mic2']:
            meta_dict[subset] = dict(idx2wav=dict(), idx2spk=dict(), idx2age=dict(), idx2gen=dict(), idx2acc=dict())
            meta_dict[subset][f'idx2{txt_format}_text'] = dict()

        # Read speaker information file and process it into separate dictionaries for age, gender, accent per speaker.
        with open(os.path.join(src_path, 'speaker-info.txt'), mode='r') as f:
            spk_info = f.readlines()[1:]
        spk_info = [re.sub(' +', ' ', row.replace('\n', '')).split(' ', 4) for row in spk_info]
        spk2age, spk2gender, spk2accent = {}, {}, {}
        for i in range(len(spk_info)):
            spk2age[spk_info[i][0]], spk2gender[spk_info[i][0]], spk2accent[spk_info[i][0]] = \
                spk_info[i][1], spk_info[i][2], spk_info[i][3]

        # Initialize dictionaries for storing wav, text, speaker and mic info.
        idx2wav, idx2text, idx2spk, idx2mic = {}, {}, {}, {}
        print("Start to collect information for each speaker......")
        wav_folder_path, txt_folder_path = os.path.join(src_path, 'wav48_silence_trimmed'), os.path.join(src_path, 'txt')

        # Loop over each speaker, read wav and text files, process text, and store all info in dictionaries.
        for spk in tqdm(spk2age.keys()):
            if spk not in os.listdir(wav_folder_path) or spk not in os.listdir(txt_folder_path):
                continue

            wav_file_list, txt_file_list = \
                search_file_in_subfolder(os.path.join(wav_folder_path, spk), tgt_match_fn=lambda x: x.endswith('.flac')), \
                search_file_in_subfolder(os.path.join(txt_folder_path, spk), tgt_match_fn=lambda x: x.endswith('.txt'))

            idx2wav_spk = {os.path.basename(wav_file).replace('.flac', ''): wav_file for wav_file in wav_file_list}
            idx2text_spk = \
                {os.path.basename(txt_file).replace('.txt', ''): en_text_process(open(txt_file, mode='r').readline().replace('\n', ''), txt_format)
                 for txt_file in txt_file_list}

            for wav_file_idx in idx2wav_spk.keys():
                text_file_idx, mic_idx = '_'.join(wav_file_idx.split('_')[:-1]), wav_file_idx.split('_')[-1]

                idx2wav[wav_file_idx] = idx2wav_spk[wav_file_idx]
                idx2text[wav_file_idx] = idx2text_spk[text_file_idx]
                idx2spk[wav_file_idx] = f"{spk}_{mic_idx}"
                idx2mic[wav_file_idx] = mic_idx

        idx2age, idx2gender, idx2accent = {}, {}, {}
        for idx, spk_mic in idx2spk.items():
            spk = spk_mic.split('_')[0]
            idx2age[idx], idx2gender[idx], idx2accent[idx] = spk2age[spk], spk2gender[spk], spk2accent[spk]

        # Update meta_dict for each sentence based on whether the speaker belongs to validation, test or training set.
        # loop each sentence
        for idx in idx2wav.keys():
            spk, mic = idx2spk[idx].split('_')
            if spk in valid_speakers:
                meta_dict[f'valid-{mic}']['idx2wav'][idx] = idx2wav[idx]
                meta_dict[f'valid-{mic}'][f'idx2{txt_format}_text'][idx] = idx2text[idx]
                meta_dict[f'valid-{mic}']['idx2spk'][idx] = idx2spk[idx]
                meta_dict[f'valid-{mic}']['idx2gen'][idx] = idx2gender[idx]
                meta_dict[f'valid-{mic}']['idx2age'][idx] = idx2age[idx]
                meta_dict[f'valid-{mic}']['idx2acc'][idx] = idx2accent[idx]

            elif spk in test_speakers:
                meta_dict[f'test-{mic}']['idx2wav'][idx] = idx2wav[idx]
                meta_dict[f'test-{mic}'][f'idx2{txt_format}_text'][idx] = idx2text[idx]
                meta_dict[f'test-{mic}']['idx2spk'][idx] = idx2spk[idx]
                meta_dict[f'test-{mic}']['idx2gen'][idx] = idx2gender[idx]
                meta_dict[f'test-{mic}']['idx2age'][idx] = idx2age[idx]
                meta_dict[f'test-{mic}']['idx2acc'][idx] = idx2accent[idx]

            else:
                meta_dict[f'train-{mic}']['idx2wav'][idx] = idx2wav[idx]
                meta_dict[f'train-{mic}'][f'idx2{txt_format}_text'][idx] = idx2text[idx]
                meta_dict[f'train-{mic}']['idx2spk'][idx] = idx2spk[idx]
                meta_dict[f'train-{mic}']['idx2gen'][idx] = idx2gender[idx]
                meta_dict[f'train-{mic}']['idx2age'][idx] = idx2age[idx]
                meta_dict[f'train-{mic}']['idx2acc'][idx] = idx2accent[idx]

        # --- 3. Sort Up Collected Statistical Information --- #
        # Sort data by indices and collect unique speaker list for each subset
        for subset in meta_dict.keys():
            # sort the existing data by the indices
            for data_name in meta_dict[subset].keys():
                meta_dict[subset][data_name] = dict(sorted(meta_dict[subset][data_name].items(), key=lambda x: x[0]))

            # collect the speaker list by the sorted idx2spk
            meta_dict[subset]['spk_list'] = sorted(set(meta_dict[subset]['idx2spk'].values()))

        return meta_dict


if __name__ == '__main__':
    # Execute the main method if this script is run as the main module
    VCTKMetaGenerator().main()
