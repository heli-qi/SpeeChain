"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.10
"""
import argparse
import os
from multiprocessing import Pool
from functools import partial
from typing import Dict, List

from datasets.meta_generator import SpeechTextMetaGenerator
from speechain.utilbox.dump_util import tts_text_process


class LibriTTSMetaGenerator(SpeechTextMetaGenerator):
    """
    Extract the statistical information from your specified subsets of the LibriTTS dataset.
    Statistical information contains speech and text data as well as metadata (speaker id and gender).

    """
    @staticmethod
    def add_parse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group("Specific Arguments for the LibriTTS Dataset.")
        group.add_argument('--subsets', type=str,
                           help="A comma-separated string that indicates the subsets you want to extract.",
                           default='train-clean-100,train-clean-360,train-other-500,dev-clean,dev-other,test-clean,test-other')
        group.add_argument('--separator', type=str, default=',',
                           help="The separator used to separate the input 'subsets' from a string to a list of string.")
        group.add_argument('--ncpu', type=int, default=8,
                           help="The number of threads you want to use to collect the statistical information of the "
                                "dataset.")
        return parser

    @staticmethod
    def collect_data(spk: str, subset_path: str, txt_format: str):
        # info initialization
        idx2data = dict(idx2wav=dict(), idx2spk=dict())
        idx2data[f'idx2{txt_format}_text'] = dict()
        spk_path = os.path.join(subset_path, spk)

        # looping for each chapter made by the specific speaker
        for chp in os.listdir(spk_path):
            if not chp.isdigit():
                continue
            chp_path = os.path.join(spk_path, chp)

            # pandas package can also be used to read the text data as follow:
            # ---------------------------------------------------------------------------------------------------
            # import pandas as pd
            # # use '\t' as the separator may cause some unexpected problems
            # transcripts = pd.read_csv(os.path.join(chp_path, f"{spk}_{chp}.trans.tsv"), sep="\r", header=None)
            # transcripts = pd.DataFrame([row.split('\t') for row in transcripts[0]])
            #
            # # In the cases that there are only two columns,
            # # there is only one row where original text and normalized text are not separated
            # if len(transcripts.keys()) < 3:
            #     transcripts[2] = pd.Series([sent.split('\t')[1] for sent in transcripts.pop(1)])
            # # In the case that original text and normalized text cannot be separated by '\t',
            # # there will be several float('nan') in the third column
            # else:
            #     flags = [isinstance(sent, float) for sent in transcripts[2]]
            #     for i in range(len(flags)):
            #         # manually get the normalized text if necessary
            #         if flags[i]:
            #             transcripts[2][i] = transcripts[1][i].split('\t')[1]
            #     transcripts.pop(1)
            #
            # # record each idx-wav and idx-sent pairs
            # for idx, sent in zip(transcripts[0], transcripts[2]):
            #     wav_path = os.path.join(chp_path, f"{idx}.wav")
            #     assert os.path.exists(wav_path), f"{wav_path} doesn't exist!"
            #     # record the waveform file path
            #     idx2data['wav'][idx] = wav_path
            #     # record the speaker ID
            #     idx2data['spk'][idx] = spk
            #     # record the processed text
            #     idx2data['sent'][idx] = tts_text_process(sent, txt_format)
            # ------------------------------------------------------------------
            # But there is a risk that some users may meet the error related to '\r' separator.
            # Therefore, it's a better idea to read the text data by Python built-in functions as below
            # since these .tsv files are not very large.
            with open(os.path.join(chp_path, f"{spk}_{chp}.trans.tsv"), mode='r') as f:
                idx2sent = f.readlines()
            idx2sent = dict(zip(
                [row.replace('\n', '').split("\t")[0] for row in idx2sent],
                [row.replace('\n', '').split("\t")[2] for row in idx2sent]
            ))

            # record each idx-wav and idx-sent pairs
            for idx, sent in idx2sent.items():
                wav_path = os.path.join(chp_path, f"{idx}.wav")
                assert os.path.exists(wav_path), f"{wav_path} doesn't exist!"
                # record the waveform file path
                idx2data['idx2wav'][idx] = wav_path
                # record the speaker ID
                idx2data['idx2spk'][idx] = spk
                # record the processed text
                idx2data[f'idx2{txt_format}_text'][idx] = tts_text_process(sent, txt_format)

        return idx2data

    def generate_meta_dict(self, src_path: str, txt_format: str,
                           subsets: str = 'train-clean-100,train-clean-360,train-other-500,dev-clean,dev-other,'
                                          'test-clean,test-other',
                           separator: str = ',', ncpu: int = 8) -> Dict[str, Dict[str, Dict[str, str] or List[str]]]:
        """

        Args:
            src_path: str
                The path where the original dataset is placed.
            txt_format: str
                The text format you want to use to process the raw text
            subsets: str
                A comma-separated string that indicates the subsets you want to extract.
            separator: str
                The separator used to separate the input 'subsets' from a string to a list of string
            ncpu: int
                The number of CPUs you want to use to collect the statistic information of the dataset.

        Returns:

        """
        # --- Statistical Dict Initialization --- #
        # separate the argument 'subsets' from a string to a list of string by the input 'separator'
        subsets = subsets.replace(' ', '').split(separator)
        if '' in subsets:
            subsets.remove('')
        meta_dict = dict()
        for subset in subsets:
            assert subset in ['train-clean-100', 'train-clean-360', 'train-other-500', 'dev-clean', 'dev-other',
                              'test-clean', 'test-other'], f"Unknown input subset {subset}."
            meta_dict[subset] = dict(idx2wav=dict(), idx2spk=dict(), idx2gen=dict())
            meta_dict[subset][f'idx2{txt_format}_text'] = dict()

        # --- Collect speech and text data --- #
        # looping for each subset
        for subset in meta_dict.keys():
            subset_path = os.path.join(src_path, subset)
            print(f"Collecting statistic information of subset {subset} in {subset_path}")

            # divide the workload with the data of a speaker as the unit
            spk_list = [file_name for file_name in os.listdir(subset_path) if file_name.isdigit()]
            with Pool(ncpu) as executor:
                collect_data_func = partial(self.collect_data, subset_path=subset_path, txt_format=txt_format)
                meta_dict_list = executor.map(collect_data_func, spk_list)

            # combine the data of all speakers together to form the data of a dataset
            for spk_meta_dict in meta_dict_list:
                for key in spk_meta_dict.keys():
                    assert key in meta_dict[subset].keys()
                    meta_dict[subset][key].update(spk_meta_dict[key].items())

        # --- Collect meta data (speaker id and gender) --- #
        spk2gen = dict()
        with open(os.path.join(src_path, 'speakers.tsv'), 'r') as f:
            spk_txt = f.readlines()[1:]
        for line in spk_txt:
            line = line.split("\t")
            spk2gen[line[0].replace(" ", "")] = line[1].replace(" ", "")

        # --- Sort Up Collected Statistical Information --- #
        for subset in meta_dict.keys():
            # sort the existing data by the indices
            for data_name in meta_dict[subset].keys():
                meta_dict[subset][data_name] = dict(sorted(meta_dict[subset][data_name].items(), key=lambda x: x[0]))

            # collect the gender information by the sorted idx2spk and spk2gen
            meta_dict[subset]['idx2gen'] = {idx: spk2gen[spk] for idx, spk in meta_dict[subset]['idx2spk'].items()}
            # collect the speaker list by the sorted idx2spk
            meta_dict[subset]['spk_list'] = sorted(set([i for i in meta_dict[subset]['idx2spk'].values()]))

        return meta_dict


if __name__ == '__main__':
    LibriTTSMetaGenerator().main()
