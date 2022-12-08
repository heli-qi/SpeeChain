"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""

import numpy as np
import torch

from typing import Dict, Any, List

from speechain.dataset.abs import Dataset
from speechain.utilbox.data_loading_util import read_data_by_path


class SpeechTextDataset(Dataset):
    """
    This Dataset subclass is mainly used by ASR and TTS models.
    In this subclass, each data instance is made up of an utterance and a sentence as well as the speaker information
    (speaker ID + speaker embedding feature).

    """

    def collate_main_data_fn(self, batch_dict: Dict[str, List]) -> Dict[str, torch.Tensor or List]:
        """
        The utterances used for training ASR and TTS models may have different lengths, so we need to do the
        padding operations to make them equal in length.

        The loaded speech feature vectors will be arranged into a single matrix with 0 padding at the end of short
        vectors. Text data remains unprocessed strings and the tokenization will be done later in the model.

        Args:
            batch_dict: Dict[str, List]
            The keys of the input `batch_dict` dictionary should be one of the following:
                1. `feat`: a List of 2d `torch.Tensor` with different lengths.
                2. `text`: a List of text strings.
                3. `spk_ids`: a List of speaker ID strings.
                4. `spk_feat`: a List of 2d `torch.Tensor` with equal lengths.

        Returns: Dict[str, torch.Tensor or List]
            `feat` and `spk_feat` are in the form of three-dimensional `torch.Tensor`;
            `text` and `spk_ids` are in the form of List of raw strings whose discretization is done in the Model object.

        """
        # get all the key names in advance
        key_list = list(batch_dict.keys())
        # loop each key in the Dict
        for key in key_list:
            # --- 1. Pad Speech Data and Stack them together --- #
            if key == 'feat':
                # para init
                batch_size, feat_dim = len(batch_dict[key]), batch_dict[key][0].shape[-1]

                # acoustic feature padding, feat.dtype needs to match the type of model parameters (torch.float32)
                feat_len = torch.LongTensor([ele.shape[0] for ele in batch_dict[key]])
                feat = torch.zeros((batch_size, feat_len.max().item(), feat_dim), dtype=torch.float32)

                # overwrite the padding matrix with each feat vector
                for i in range(batch_size):
                    # process feat data based on data type
                    if isinstance(batch_dict[key][i], np.ndarray):
                        feat[i][:feat_len[i]] = torch.tensor(batch_dict[key][i])
                    elif isinstance(batch_dict[key][i], torch.Tensor):
                        feat[i][:feat_len[i]] = batch_dict[key][i]
                    # only support np.ndarray and torch.Tensor now
                    else:
                        raise TypeError

                # update 'feat' and attach 'feat_len' for later model forward
                batch_dict[key] = feat
                batch_dict['feat_len'] = feat_len

            # --- 2. Stack Speaker Embedding Feature together --- #
            elif key == 'spk_feat':
                batch_dict[key] = torch.stack(batch_dict[key])

            # text and speaker ID data remains List[str] which will be processed by the model later
            elif key not in ['text', 'spk_ids']:
                raise RuntimeError(f"Unknown data name {key}.")

        return batch_dict

    def extract_main_data_fn(self, main_data: Dict[str, str]) -> Dict[str, Any]:
        """
        The function that loads speech-text data from the disk. If the speech is in the form of raw waveforms,
        the last dimension should be expanded to 1 of raw speech for compatibility with acoustic feature.

        Args:
            main_data: Dict[str, str]
                The keys of the input main_data dictionary should be one of the following:
                    1. 'feat': speech features, can be either raw waveforms or acoustic features like log-mel or MFCC.
                    2. 'text': transcript text, in the form of raw string. The tokenization will be done in the ASR and
                    TTS models.
                    3. 'spk_ids': speaker ID, in the form of raw string. The speaker discretization will be done in the
                    ASR and TTS models.
                    4. 'spk_feat': speaker embedding features.
                `spk_ids` and `spk_feat` are designed for multi-speaker TTS model and are not mandatory to be included
                in `main_data; 'feat' and 'text' are mandatory to be included for ASR and TTS training.
                However, during model testing, we can choose to only include one of 'feat' and 'text' here to reduce the
                CPU burden.

        Returns:
            `feat` and `spk_feat` are in the form of two-dimensional `torch.Tensor`;
            `text` and `spk_ids` are in the form of raw strings whose discretization is done in the Model object.

        """
        assert 'feat' in main_data.keys() or 'text' in main_data.keys(), \
            "Please at least include one of 'feat' and 'text' in a single batch."

        for key in main_data.keys():
            # --- 1. Speech Data Extraction --- #
            if key == 'feat':
                # read the selected data speech feature as a tensor by its path
                main_data[key] = read_data_by_path(main_data[key], return_tensor=True)

            # --- 2. Transcript Text Extraction --- #
            elif key == 'text':
                # text length is not returned because the text here is just a raw string
                assert isinstance(main_data[key], str),\
                    f"The 'text' data should be given as a string, but got {main_data[key]}"

            # --- 3. Speaker ID Extraction --- #
            elif key == 'spk_ids':
                # the speaker ID here is just a raw string
                assert isinstance(main_data['spk_ids'], str),\
                    f"The 'spk_ids' data should be given as a string, but got {main_data[key]}"

            # --- 4. Speaker Embedding Feature --- #
            elif key == 'spk_feat':
                # read the selected data speech feature as a tensor by its path
                main_data[key] = read_data_by_path(main_data[key], return_tensor=True)

            else:
                raise RuntimeError(f"Unknown data name {key}. For {self.__class__}, the key name must be one of "
                                   "'feat' (for paths of raw waveforms or acoustic features), "
                                   "'text' (for transcript text data), "
                                   "'spk_ids' (for speaker IDs), "
                                   "'spk_feat' (for speaker embedding features).")

        return main_data
