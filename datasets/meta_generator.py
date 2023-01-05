"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.11
"""
import argparse
import os
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np


class SpeechTextMetaGenerator(ABC):
    """
    The base class for metadata generators of all the datasets. For contributing a new dataset dumping pipeline,
    please inherit this class in your meta_generator.py and implement the abstract functions generate_meta_dict().

    """

    def parse(self):
        """
        Declaration function for the general arguments shared by all dataset implementations.
        There are two shared general arguments here: 'src_path' and 'txt_format'.

        """
        parser = argparse.ArgumentParser(description='params')
        group = parser.add_argument_group("General Arguments for Statistical Information Generation.")
        group.add_argument('--src_path', type=str, required=True,
                           help="The path where the original dataset is placed.")
        group.add_argument('--txt_format', type=str, default='normal',
                           help="The text processing format controlling how to process the transcript sentence of "
                                "each utterance before saving them into 'idx2text' and 'text'. (default: normal)")
        # Add customized arguments if needed
        parser = self.add_parse(parser)
        return parser.parse_args()

    @staticmethod
    def add_parse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        The interface where users can add their own arguments.
        This function is not mandatory to be overridden.

        Args:
            parser: argparse.ArgumentParser
                The name space where you want to add your arguments.

        Returns:
            parser: argparse.ArgumentParser
                The name space containing your arguments.

        """
        return parser

    @abstractmethod
    def generate_meta_dict(self, src_path: str, txt_format: str, **kwargs) \
            -> Dict[str, Dict[str, Dict[str, str] or List[str]]]:
        """
        The abstract function that must be overridden to generate the metadata Dict for the specified dataset.

        Args:
            src_path: str
                The path where the original dataset is placed.
            txt_format: str = 'normal'
                The text processing format controlling how to process the transcript sentence of each utterance before
                saving them into 'idx2text' and 'text'.
            **kwargs:
                The newly-added arguments for your dataset implementation.

        Returns: Dict[str, Dict[str, Dict[str, str] or List[str]]]
            The metadata dictionary you want to save on the disk for later use.
            The first-level keys indicate the names of subsets in the dataset.
            The second-level keys indicate the names of metadata files you want to save.
                The third-level elements can be either Dict or List. Dict represents those 'idx2XXX' files where each
                line contains a file index and corresponding metadata value. List represents those 'XXX' files where
                each line only contains metadata value without any file indices.

        """
        raise NotImplementedError

    def main(self):
        """
        The entrance of SpeechTextMetaGenerator. There are twp steps in this function:
        1. obtain the metadata dictionary by self.generate_meta_dict()
        2. save the metadata in the dictionary in the given src_path where each subset will have a specific folder.

        """
        # --- 0. Argument Initialization --- #
        args = vars(self.parse())
        src_path = args.pop('src_path')
        txt_format = args['txt_format']

        # --- 1. Meta Data Generation (meta data dict) --- #
        meta_dict = self.generate_meta_dict(src_path, **args)

        # --- 2. Save all the statistical information to the disk --- #
        for subset in meta_dict.keys():
            assert 'idx2wav' in meta_dict[subset].keys() and f'idx2{txt_format}_text' in meta_dict[subset].keys(), \
                f"Your generate_meta_dict() implementation must return at least idx2wav, idx2{txt_format}_text " \
                f"as the file names."

            subset_path = os.path.join(src_path, subset)
            os.makedirs(subset_path, exist_ok=True)
            print(f"Saving statistic data files {list(meta_dict[subset].keys())} of subset {subset} to {subset_path}/")

            # each key acts as the file name while the corresponding value is the content of the file
            for meta_name, meta_content in meta_dict[subset].items():
                file_path = os.path.join(src_path, subset, meta_name)
                np.savetxt(file_path, meta_content if isinstance(meta_content, List) else list(meta_content.items()),
                           fmt='%s')


# For your personal stat_info_generator.py, please call the main function of your SpeechTextStatGenerator in the main
# branch like the way below. (Note: don't forget to change SpeechTextStatGenerator() to YourStatGenerator())
if __name__ == '__main__':
    SpeechTextMetaGenerator().main()
