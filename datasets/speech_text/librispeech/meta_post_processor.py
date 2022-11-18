"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.11
"""
import os
import numpy as np
from typing import Dict, List

from datasets.speech_text.meta_post_processor import SpeechTextMetaPostProcessor


class LibriSpeechMetaPostProcessor(SpeechTextMetaPostProcessor):
    """
    
    """

    @staticmethod
    def contain_meta_data(path: str):
        for file in os.listdir(path):
            # meta data files are not directory and don't have any suffix
            if not os.path.isdir(os.path.join(path, file)) and len(file.split('.')) == 1:
                return True
        return False

    @staticmethod
    def get_meta_data_files(path: str):
        return [file for file in os.listdir(path)
                if not os.path.isdir(os.path.join(path, file)) and '.' not in file]

    @staticmethod
    def get_list_union(input_list: List[List]):
        # get the intersection
        union = set(input_list[0])
        for i in range(1, len(input_list)):
            union = union.intersection(set(input_list[i]))
        return union

    @staticmethod
    def save_meta_data_files(meta_data: Dict, subset_path: str):
        # loop each data dict in the current subset of meta_data
        for key, value in meta_data.items():
            np.savetxt(os.path.join(subset_path, key), value, fmt='%s')

    def meta_post_process(self, src_path: str, **kwargs):
        """
            Gather all the meta data files for manually-created subsets 'train-clean-460', 'train-960', and 'dev'.

            Args:
                src_path: str
                    The path where the original dataset is placed.

        """
        # looping for each target subset
        for subset in ['train-clean-460', 'train-960', 'dev']:
            member_path_list = None
            if subset == 'train-clean-460':
                # 'train-clean-460' = 'train-clean-100' + 'train-clean-360'
                member_path_list = [os.path.join(src_path, 'train-clean-100'),
                                    os.path.join(src_path, 'train-clean-360')]
            elif subset == 'train-960':
                # 'train-clean-460' = 'train-clean-100' + 'train-clean-360' + 'train-other-500'
                member_path_list = [os.path.join(src_path, 'train-clean-100'),
                                    os.path.join(src_path, 'train-clean-360'),
                                    os.path.join(src_path, 'train-other-500')]
            elif subset == 'dev':
                # 'dev' = 'dev-clean' + 'dev-other'
                member_path_list = [os.path.join(src_path, 'dev-clean'), os.path.join(src_path, 'dev-other')]

            # skip if one of 'train-clean-100' and 'train-clean-360' doesn't exist or doesn't have meta files
            if (sum([os.path.exists(i) for i in member_path_list]) != len(member_path_list)) or \
                    (sum([self.contain_meta_data(i) for i in member_path_list]) != len(member_path_list)):
                continue

            subset_path = os.path.join(src_path, subset)
            os.makedirs(subset_path, exist_ok=True)
            print(f"Collecting meta data of subset {subset} in {subset_path}")

            meta_data_file_list = self.get_list_union([self.get_meta_data_files(i) for i in member_path_list])
            for meta_data_file in meta_data_file_list:
                meta_data = []

                for member_path in member_path_list:
                    with open(os.path.join(member_path, meta_data_file), mode='r') as f:
                        _member_data = f.readlines()
                    meta_data.extend([row.replace('\n', '') for row in _member_data])

                np.savetxt(os.path.join(subset_path, meta_data_file), meta_data, fmt='%s')


if __name__ == '__main__':
    LibriSpeechMetaPostProcessor().main()
