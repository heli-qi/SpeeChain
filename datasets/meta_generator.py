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

from speechain.utilbox.import_util import parse_path_args


class SpeechTextMetaGenerator(ABC):
    """
        The base class for all metadata generators of datasets. To contribute a new dataset dumping pipeline,
        inherit this class in your `meta_generator.py` and implement the `generate_meta_dict` abstract method.
    """

    def parse(self):
        """
            Parse and declare common arguments shared by all dataset implementations.
            Current common arguments include: 'src_path', 'tgt_path', and 'txt_format'.

            Returns:
                argparse.Namespace: The namespace containing both the general and user-defined arguments.
        """
        parser = argparse.ArgumentParser(description='Parameters for Statistical Information Generation')
        group = parser.add_argument_group("General Arguments")
        group.add_argument('--src_path', type=str, default=None, help="Path to the original dataset.")
        group.add_argument('--tgt_path', type=str, required=True, help="Destination path for metadata files.")
        group.add_argument('--txt_format', type=str, default='no-punc',
                           help="Text processing format, defines the processing of transcript sentences "
                                "before saving into 'idx2text'. Default is 'normal'")
        # Add custom arguments if needed
        parser = self.add_parse(parser)
        return parser.parse_args()

    @staticmethod
    def add_parse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
            Interface for users to add custom arguments. This method can be overridden, but it's not mandatory.

            Args:
                parser: The argparse parser to which you want to add your arguments.

            Returns:
                argparse.ArgumentParser: The parser containing the custom arguments.
        """
        return parser

    @abstractmethod
    def generate_meta_dict(self, src_path: str, txt_format: str, **kwargs) \
            -> Dict[str, Dict[str, Dict[str, str] or List[str]]]:
        """
        Generate a metadata dictionary for the specified dataset.
        Must be overridden in subclasses.

        Args:
            src_path: Path to the original dataset.
            txt_format: Text processing format.
            **kwargs: Custom arguments for the dataset implementation.

        Returns: Dict[str, Dict[str, Dict[str, str] or List[str]]]
            The metadata dictionary you want to save on the disk.
            The first-level keys indicate the names of subsets in the dataset.
                The second-level keys indicate the names of metadata files you want to save.
                    The third-level elements can be either Dict or List. Dict represents those 'idx2XXX' files where each
                    line contains a file index and corresponding metadata value. List represents those 'XXX' files where
                    each line only contains metadata value without any file indices.

        """
        raise NotImplementedError

    def main(self):
        """
            Main entry point for `SpeechTextMetaGenerator`.

            Steps:
            1. Obtain metadata dictionary via `self.generate_meta_dict`.
            2. Save the metadata in the given source path, creating a specific folder for each subset.

        """
        # Argument Initialization
        args = vars(self.parse())
        tgt_path = args.pop('tgt_path')
        if args['src_path'] is None:
            args['src_path'] = tgt_path
        txt_format = args['txt_format']

        # Metadata Generation
        args['src_path'] = parse_path_args(args['src_path'])
        meta_dict = self.generate_meta_dict(**args)

        # Save statistical information to disk
        tgt_path = parse_path_args(tgt_path)
        for subset in meta_dict.keys():
            assert 'idx2wav' in meta_dict[subset].keys() and f'idx2{txt_format}_text' in meta_dict[subset].keys(), \
                f"'generate_meta_dict' must return at least idx2wav, idx2{txt_format}_text in the file names."

            subset_path = os.path.join(tgt_path, subset)
            os.makedirs(subset_path, exist_ok=True)
            print(f"Saving metadata files {list(meta_dict[subset].keys())} of subset {subset} to {subset_path}/")

            # each key acts as the file name while the corresponding value is the content of the file
            for meta_name, meta_content in meta_dict[subset].items():
                file_path = os.path.join(tgt_path, subset, meta_name)
                np.savetxt(file_path, meta_content if isinstance(meta_content, List) else list(meta_content.items()),
                           fmt='%s')


# For your personal stat_info_generator.py, please call the main function of your SpeechTextStatGenerator in the main
# branch like the way below. (Note: don't forget to change SpeechTextStatGenerator() to YourStatGenerator())
if __name__ == '__main__':
    SpeechTextMetaGenerator().main()
