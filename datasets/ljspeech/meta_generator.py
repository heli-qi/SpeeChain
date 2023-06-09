"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import argparse
from typing import Dict, List

import numpy as np
import os
import pandas as pd

from datasets.meta_generator import SpeechTextMetaGenerator
from speechain.utilbox.dump_util import en_text_process


class LJSpeechMetaGenerator(SpeechTextMetaGenerator):
    """
    Extract the statistical information from your specified subsets of the LJSpeech dataset.
    Statistical information contains speech and text data as well as metadata (speaker id and gender).

    Although LJSpeech is a single-speaker dataset, we also provide 'idx2spk', 'spk_list', and 'idx2gen' files which
    contain identical information for each speech-text pair.
    These files will be useful when you are doing experiments on LJSpeech along-with other datasets.
    """
    @staticmethod
    def add_parse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group("Specific Arguments for the LJSpeech Dataset.")
        group.add_argument('--valid_section', type=str, default='3',
                           help="Which section of LJSpeech you want to use as the validation set."
                                "Each digit represent a section in LJSpeech (i.e., LJ0XX).")
        group.add_argument('--test_section', type=str, default='1,2',
                           help="Which section of LJSpeech you want to use as the test set."
                                "Each digit represent a section in LJSpeech (i.e., LJ0XX).")
        group.add_argument('--separator', type=str, default=',',
                           help="How to separate different section numbers in 'valid_section' and 'test_section'.")
        return parser

    def generate_meta_dict(self, src_path: str, txt_format: str,
                           valid_section: str = '3', test_section: str = '1,2', separator: str = ',') \
            -> Dict[str, Dict[str, Dict[str, str] or List[str]]]:
        """

        Args:
            src_path:
            txt_format:
            valid_section:
            test_section:
            separator:

        Returns:

        """
        # --- 0. Argument Checking --- #
        def check_section(input_section: str):
            input_section = input_section.split(separator)
            assert len(set(input_section)) == len(input_section), \
                f"Your input argument {input_section} has repeated elements!"
            assert len([sec.isdigit() for sec in input_section]) == len(input_section), \
                f"Your input argument {input_section} has some non-integer elements!"
            return [int(sec) for sec in input_section]

        valid_section = check_section(valid_section)
        test_section = check_section(test_section)
        assert len(set(valid_section).difference(set(test_section))) == len(valid_section), \
            f"Your input valid_section and test_section have overlapping elements! " \
            f"Got valid_section={valid_section} & test_section={test_section}."

        # --- 1. Text Data Reading and Recording --- #
        meta_dict = dict()
        for subset in ['train', 'valid', 'test']:
            meta_dict[subset] = dict(idx2wav=dict(), idx2spk=dict(), idx2gen=dict())
            meta_dict[subset][f'idx2{txt_format}_text'] = dict()

        # metadata.csv contains a lot of '"', which is taken as the default quotechar by read_csv().
        # To correctly load those '"' for processing, we need to set quotechar=None and quoting=3.
        idx2text = pd.read_csv(os.path.join(src_path, "metadata.csv"),
                               delimiter='|', quotechar=None, quoting=3, header=None).to_numpy(dtype=str)
        # retain the normalized text and remove the original one
        idx2text = np.delete(idx2text, 1, axis=1)
        # loop each sentence
        for idx, text in idx2text:
            # record the wav file to different subsets by their section number
            sec_num = int(idx.split('-')[0].replace('LJ', ''))
            if sec_num in valid_section:
                meta_dict['valid'][f'idx2{txt_format}_text'][idx] = en_text_process(text, txt_format=txt_format)
            elif sec_num in test_section:
                meta_dict['test'][f'idx2{txt_format}_text'][idx] = en_text_process(text, txt_format=txt_format)
            else:
                meta_dict['train'][f'idx2{txt_format}_text'][idx] = en_text_process(text, txt_format=txt_format)

        # --- 2. Speech and Speaker Data Recording --- #
        wav_path = os.path.join(src_path, "wavs")
        # loop each wav file in the 'wavs'
        for wav_file in os.listdir(wav_path):
            file_path = os.path.join(wav_path, wav_file)
            if not file_path.endswith('.wav'):
                continue

            # record the wav file to different subsets by their section number
            file_name = wav_file.split('.')[0]
            sec_num = int(file_name.split('-')[0].replace('LJ', ''))
            if sec_num in valid_section:
                meta_dict['valid']['idx2wav'][file_name] = os.path.abspath(file_path)
                meta_dict['valid']['idx2spk'][file_name] = 'LJ'
                meta_dict['valid']['idx2gen'][file_name] = 'F'
            elif sec_num in test_section:
                meta_dict['test']['idx2wav'][file_name] = os.path.abspath(file_path)
                meta_dict['test']['idx2spk'][file_name] = 'LJ'
                meta_dict['test']['idx2gen'][file_name] = 'F'
            else:
                meta_dict['train']['idx2wav'][file_name] = os.path.abspath(file_path)
                meta_dict['train']['idx2spk'][file_name] = 'LJ'
                meta_dict['train']['idx2gen'][file_name] = 'F'

        # --- 3. Sort Up Collected Statistical Information --- #
        for subset in meta_dict.keys():
            # sort the existing data by the indices
            for data_name in meta_dict[subset].keys():
                meta_dict[subset][data_name] = dict(sorted(meta_dict[subset][data_name].items(), key=lambda x: x[0]))

            # collect the speaker list by the sorted idx2spk
            meta_dict[subset]['spk_list'] = ['LJ']

        return meta_dict


if __name__ == '__main__':
    LJSpeechMetaGenerator().main()
