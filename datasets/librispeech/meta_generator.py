"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import argparse
import os

from functools import partial
from multiprocessing import Pool
from typing import Dict, List
from tqdm import tqdm

from datasets.meta_generator import SpeechTextMetaGenerator
from speechain.utilbox.dump_util import en_text_process


class LibriSpeechMetaGenerator(SpeechTextMetaGenerator):
    """
    Extract the statistical information from your specified subsets of the LibriSpeech dataset.
    Statistical information contains speech and text data as well as metadata (speaker id and gender).

    """
    @staticmethod
    def add_parse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group("Specific Arguments for the LibriSpeech Dataset.")
        group.add_argument('--subsets', type=str,
                           default='train-clean-100,train-clean-360,train-other-500,dev-clean,dev-other,test-clean,test-other',
                           help="A comma-separated string that indicates the subsets you want to extract.")
        group.add_argument('--separator', type=str, default=',',
                           help="The separator used to separate the input 'subsets' from a string to a list of string. "
                                "(default: ',')")
        group.add_argument('--ncpu', type=int, default=8,
                           help="The number of threads you want to use to collect the statistical information of the "
                                "dataset. (default: 8)")
        return parser

    @staticmethod
    def collect_data(spk: str, subset_path: str, txt_format: str):
        # idx2data Dict initialization
        idx2data = dict(idx2wav=dict(), idx2spk=dict())
        idx2data[f'idx2{txt_format}_text'] = dict()
        spk_path = os.path.join(subset_path, spk)

        # looping for each chapter made by the specific speaker
        for chp in os.listdir(spk_path):
            if not chp.isdigit():
                continue
            chp_path = os.path.join(spk_path, chp)

            # looping for each audio file of the specific chapter
            for file in os.listdir(chp_path):
                file_path = os.path.join(chp_path, file)

                # text data
                if file.endswith('.txt'):
                    # numpy package can also be used to read the text data as follow:
                    # ---------------------------------------------------------------
                    # chp_text = np.loadtxt(file_path, delimiter="\n", dtype=str)
                    # if len(chp_text.shape) == 0:
                    #     chp_text = np.array(chp_text.tolist().split(" ", 1)).reshape(1, -1)
                    # else:
                    #     chp_text = np.stack(np.chararray.split(chp_text, maxsplit=1))
                    # ------------------------------------------------------------------
                    # The codes above is faster but there is a risk that some users may meet the error below::
                    # TypeError: control character 'delimiter' cannot be a newline (`\r` or `\n`).
                    # A safer method is to read the text data by Python built-in functions as below,
                    # but it may consume more time.

                    # read the text data into a list
                    with open(file_path, mode='r') as f:
                        chp_text = f.readlines()
                    chp_text = [row.replace('\n', '').split(" ", 1) for row in chp_text]

                    # process each sentence and store them into idx2data
                    for i in range(len(chp_text)):
                        idx2data[f'idx2{txt_format}_text'][chp_text[i][0]] = \
                            en_text_process(chp_text[i][1], txt_format=txt_format)

                # speech & speaker data
                elif file.endswith('.flac'):
                    idx2data['idx2wav'][file.split('.')[0]] = os.path.abspath(file_path)
                    idx2data['idx2spk'][file.split('.')[0]] = spk
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

        # --- Collect speech, speaker, text, and gender data --- #
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

        # generate speaker-to-gender mapping Dict
        spk2gen = dict()
        with open(os.path.join(src_path, 'SPEAKERS.TXT'), 'r') as f:
            spk_txt = f.readlines()[12:]
        for line in spk_txt:
            line = line.split("|")
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

        # --- Process the Additional Text Data for Language Model --- #
        lm_text_path = os.path.join(os.path.dirname(src_path), 'lm_text')
        proc_text_path = os.path.join(lm_text_path, f'{txt_format}_lm_text')
        if not os.path.exists(proc_text_path):
            print("Start to process the additional LM text corpus......")
            with open(os.path.join(lm_text_path, 'librispeech-lm-norm.txt'), mode='r') as f:
                ori_lm_text = f.readlines()

            # make a specific row for each sentence and attach the sentence index at the beginning
            proc_lm_text, index = [], 0
            for _ori_text in tqdm(ori_lm_text):
                _proc_text = en_text_process(_ori_text, txt_format)
                if _proc_text != '':
                    proc_lm_text.append(f'{index} {_proc_text}\n')
                    index += 1

            with open(proc_text_path, mode='w+') as f:
                f.writelines(proc_lm_text)
            print(f"Processed LM text has been saved to {proc_text_path}!")
        else:
            print(f"Processed LM text has already existed in {proc_text_path}!")

        return meta_dict


if __name__ == '__main__':
    LibriSpeechMetaGenerator().main()
