"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.11
"""
import argparse
from abc import ABC, abstractmethod


class SpeechTextMetaPostProcessor(ABC):
    """
    The base class for metadata post-processing of all the datasets. This class is not mandatory to be overridden.
    For contributing a new dataset dumping pipeline, please inherit this class in your meta_post_processor.py if needed
    and implement the abstract functions meta_post_process().

    """

    def parse(self):
        """
        Declaration function for the general arguments shared by all dataset implementations.
        There is one shared general argument here: 'src_path'.

        """
        parser = argparse.ArgumentParser(description='params')
        group = parser.add_argument_group("General Arguments for Statistical Information Generation.")
        group.add_argument('--src_path', type=str, required=True,
                           help="The path where the original dataset is placed.")
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
    def meta_post_process(self, src_path: str, **kwargs):
        """
        The abstract function that must be overridden to post-process the metadata files in the given src_path for your
        target dataset.

        Args:
            src_path: str
                The path where the original dataset is placed.
            **kwargs:
                The newly-added arguments for your dataset implementation.

        """
        raise NotImplementedError

    def main(self):
        """
        The entrance of SpeechTextMetaPostProcessor.

        """
        # --- 0. Argument Initialization --- #
        args = vars(self.parse())
        src_path = args.pop('src_path')

        # --- 1. Metadata Post-processing --- #
        self.meta_post_process(src_path, **args)


# For your personal stat_post_processor.py, please call the main function of your SpeechTextPostProcessor in the main
# branch like the way below. (Note: don't forget to change SpeechTextPostProcessor() to YourPostProcessor())
if __name__ == '__main__':
    SpeechTextMetaPostProcessor().main()
