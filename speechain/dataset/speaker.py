"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
from typing import Dict, Any, List

import numpy as np
import torch

from speechain.dataset.abs import Dataset


class SpeakerDataset(Dataset):
    """

    """
    def dataset_init(self, **dataset_conf):
        pass

    def read_data_file(self, data_file: str) -> Dict[str, str]:
        """

        Args:
            data_file:

        Returns:

        """
        raise NotImplementedError


    def read_label_file(self, label_file: str) -> Dict[str, str]:
        """

        Args:
            label_file:

        Returns:

        """
        # read idx2spk file that contains the speaker string id (raw data needed to be processed)
        # str -> np.ndarray
        spk_ids = np.loadtxt(label_file, dtype=str, delimiter=" ")
        # np.ndarray -> Dict[str, str]
        return dict(zip(spk_ids[:, 0], spk_ids[:, 1]))


    def read_meta_file(self, meta_file: str, meta_type: str) -> Dict[str, str]:
        """

        Args:
            meta_file:
            meta_type:

        Returns:

        """
        raise NotImplementedError


    def __getitem__(self, index) -> Dict[str, Any]:
        """

        Args:
            index:

        Returns:

        """
        outputs = dict()
        # extract the speaker feature from the disk by the index

        # extract the speaker string id by the index
        spk_ids = self.tgt_label[index]

        outputs.update(
            spk_ids=spk_ids
        )
        return outputs


    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        outputs = dict()
        if sum(['spk_feat' in ele.keys() for ele in batch]):
            outputs.update(
                spk_feat=torch.stack([ele['spk_feat'] for ele in batch])
            )

        if sum(['spk_ids' in ele.keys() for ele in batch]):
            outputs.update(
                spk_ids=[ele['spk_ids'] for ele in batch]
            )

        return outputs
