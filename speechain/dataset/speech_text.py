"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import os.path

import numpy as np
import torch
import random

from g2p_en import G2p
from typing import Dict, Any, List
from functools import partial
from speechain.tokenizer.g2p import abnormal_phns

from speechain.dataset.abs import Dataset
from speechain.utilbox.data_loading_util import read_data_by_path, load_idx2data_file
from speechain.utilbox.feat_util import convert_wav_to_pitch


class SpeechTextDataset(Dataset):
    """
    This Dataset subclass is mainly used by ASR and TTS models.
    In this subclass, each data instance is made up of an utterance and a sentence as well as the speaker information
    (speaker ID + speaker embedding feature).

    """
    def dataset_init_fn(self,
                        use_g2p: bool = False,
                        pitch_conf: Dict = None):
        """

        Args:
            # phoneme-related
            use_g2p: bool = False
                Whether to process the raw string by G2P. We don't recommend you to turn it on because on-the-fly
                transformer from string to phoneme list consumes a lot of CPU resources.
            # pitch-related
            pitch_conf: Dict = None
                The configuration given to convert_wav_to_pitch() for pitch extraction.
                If not given, pitch extraction will not be done on-the-fly.

        """
        # phoneme extraction
        if use_g2p:
            self.g2p = G2p()

        # pitch extraction
        if pitch_conf is not None:
            self.pitch_extract_fn = partial(convert_wav_to_pitch, return_tensor=True, **pitch_conf)

    @staticmethod
    def data_len_register_fn(main_data: Dict[str, Dict[str, str]]) -> Dict[str, int or float] or None:
        """

        Returns:
            If 'text' is given in main_data, return the number of characters in each sentence.
            Otherwise, return None

        """
        if 'text' in main_data.keys():
            return {key: len(value) for key, value in main_data['text'].items()}
        else:
            return None

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
                2. `pitch`: a List of 1d `torch.Tensor` with different lengths.
                3. `text`: a List of text strings.
                4. `spk_ids`: a List of speaker ID strings.
                5. `spk_feat`: a List of 2d `torch.Tensor` with equal lengths.

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
                feat_len = torch.LongTensor([ele.shape[0] for ele in batch_dict[key]])
                batch_size, feat_maxlen, feat_dim = \
                    len(batch_dict[key]), feat_len.max().item(), batch_dict[key][0].shape[-1]

                # acoustic feature padding, feat.dtype needs to match the type of model parameters (torch.float32)
                feat = torch.zeros((batch_size, feat_maxlen, feat_dim), dtype=torch.float32)

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

            # --- 2. Pad Pitch Data and Stack them together --- #
            if key == 'pitch':
                # para init
                pitch_len = torch.LongTensor([ele.shape[0] for ele in batch_dict[key]])
                batch_size, pitch_maxlen = len(batch_dict[key]), pitch_len.max().item()

                # pitch padding, pitch.dtype needs to match the type of model parameters (torch.float32)
                pitch = torch.zeros((batch_size, pitch_maxlen), dtype=torch.float32)

                # overwrite the padding matrix with each pitch vector
                for i in range(batch_size):
                    # process feat data based on data type
                    if isinstance(batch_dict[key][i], np.ndarray):
                        pitch[i][:pitch_len[i]] = torch.tensor(batch_dict[key][i])
                    elif isinstance(batch_dict[key][i], torch.Tensor):
                        pitch[i][:pitch_len[i]] = batch_dict[key][i]
                    # only support np.ndarray and torch.Tensor now
                    else:
                        raise TypeError

                batch_dict[key] = pitch
                batch_dict['pitch_len'] = pitch_len

            # --- 3. Separate Phoneme Duration Data into Text Data and Duration Data --- #
            elif key == 'duration':
                # para init
                batch_size, duration_len = len(batch_dict[key]), torch.LongTensor([len(ele) for ele in batch_dict[key]])

                # duration padding, feat.dtype needs to match the type of model parameters (torch.float32)
                duration = torch.zeros((batch_size, duration_len.max().item()), dtype=torch.float32)

                # overwrite the padding matrix with each duration vector
                for i in range(batch_size):
                    # process duration data based on data type
                    if isinstance(batch_dict[key][i], (np.ndarray, List)):
                        duration[i][:duration_len[i]] = torch.tensor(batch_dict[key][i])
                    elif isinstance(batch_dict[key][i], torch.Tensor):
                        duration[i][:duration_len[i]] = batch_dict[key][i]
                    else:
                        raise TypeError(f"{self.__class__.name} only supports np.ndarray and torch.Tensor now!")

                # attach 'duration' and 'duration_len' for model forward
                batch_dict[key] = duration
                batch_dict['duration_len'] = duration_len

            # --- 4. Stack Speaker Embedding Feature together --- #
            elif key == 'spk_feat':
                batch_dict[key] = torch.stack(batch_dict[key])

            # --- 5. For the pure-string data like 'text' and 'spk_ids', their tokenization will be done later in --- #
            # --- Model.batch_preprocess_fn() --- #

        return batch_dict

    def extract_main_data_fn(self, main_data: Dict) -> Dict[str, Any]:
        """
        The function that loads speech-text data from the disk. If the speech is in the form of raw waveforms,
        the last dimension should be expanded to 1 of raw speech for compatibility with acoustic feature.

        Args:
            main_data: Dict[str, str]
                The keys of the input main_data dictionary should be one of the following:
                    1. 'feat': speech features, can be either raw waveforms or acoustic features like log-mel or MFCC.
                    2. 'text': transcript text, in the form of raw string. The tokenization will be done in the ASR and
                    TTS models.
                    3. 'duration': phoneme durations. used for training fastspeech2 model.
                    4. 'spk_ids': speaker ID, in the form of raw string. The speaker discretization will be done in the
                    ASR and TTS models.
                    5. 'spk_feat': speaker embedding features.
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

        main_data_keys = list(main_data.keys())
        for key in main_data_keys:
            # --- 1. Speech Data Extraction --- #
            if key == 'feat':
                # read the selected data speech feature as a tensor by its path
                main_data[key] = read_data_by_path(main_data[key], return_tensor=True)

                # extract the pitch from the speech on-the-fly
                if hasattr(self, 'pitch_extract_fn'):
                    main_data['pitch'] = self.pitch_extract_fn(main_data[key])

            # --- 2. Transcript Text Extraction --- #
            elif key == 'text':
                # text length is not returned because the text here is just a raw string
                assert isinstance(main_data[key], str),\
                    f"The 'text' data should be given as a string, but got {main_data[key]}"
                # for the text data in the format of a list
                if main_data[key].startswith('[') and main_data[key].endswith(']'):
                    main_data[key] = main_data[key][1:-1]
                    # split the text into individual tokens by a comma followed a blank
                    if ', ' in main_data[key]:
                        main_data[key] = main_data[key].split(', ')
                        # remove the single quote marks surrounding each token if needed
                        main_data[key] = [token[1:-1] if token.startswith('\'') and token.endswith('\'') else token
                                          for token in main_data[key]]
                # process the raw string by G2P if specified
                elif hasattr(self, 'g2p'):
                    phn_list = self.g2p(main_data[key])
                    main_data[key] = [phn if phn != ' ' else '<space>' for phn in phn_list if phn not in abnormal_phns]

            # --- 3. Phoneme Duration Extraction --- #
            elif key == 'duration':
                # text length is not returned because the text here is just a raw string
                assert isinstance(main_data[key], str), \
                    f"The 'duration' data should be given as a string, but got {main_data[key]}"
                # for the text data in the format of a list
                if main_data[key].startswith('[') and main_data[key].endswith(']'):
                    main_data[key] = main_data[key][1:-1]
                    # split the text into individual tokens by a comma followed a blank
                    main_data[key] = main_data[key].split(', ')
                    # remove the single quote marks surrounding each token if needed
                    main_data[key] = [float(duration[1:-1]) if duration.startswith('\'') and duration.endswith('\'')
                                      else float(duration) for duration in main_data[key]]

            # --- 4. Speaker ID Extraction --- #
            elif key == 'spk_ids':
                # the speaker ID here is just a raw string
                assert isinstance(main_data['spk_ids'], str),\
                    f"The 'spk_ids' data should be given as a string, but got {main_data[key]}"

            # --- 5. Speaker Embedding Feature --- #
            elif key == 'spk_feat':
                # read the selected data speech feature as a tensor by its path
                main_data[key] = read_data_by_path(main_data[key], return_tensor=True)

            else:
                raise RuntimeError(f"Unknown data name {key}! "
                                   f"For {self.__class__.__name__}, the key in 'main_data' must be one of "
                                   "'feat' (for paths of raw waveforms or acoustic features), "
                                   "'text' (for transcript text data), "
                                   "'duration' (for phoneme duration data), "
                                   "'spk_ids' (for speaker IDs), "
                                   "'spk_feat' (for speaker embedding features).")
        return main_data


class RandomSpkFeatDataset(SpeechTextDataset):
    """
    This Dataset subclass inherits SpeechTextDataset and is mainly used for multi-speaker TTS evaluation.
    Random speaker embedding feature will be picked up as the reference for TTS synthesis.

    `collate_main_data_fn` of the parent class will be reused to collate a batch of data instances.

    """
    def dataset_init_fn(self, spk_feat: List[str] or str = None,
                        min_ref_len: int = None, ref_len: List[str] or str = None,
                        mixup_number: int = 1, same_gender: bool = True,
                        tgt_gender: str = None, gender_info: List[str] or str = None,
                        **super_conf):
        """

        Args:
            spk_feat: List[str] or str
                The address of the idx2spk_feat that contains the speaker embedding feature files you want to use.
            min_ref_len: int = None
                The minimal length for the reference speech to extract speaker embedding
            ref_len: List[str] or str = None
                The address of the idx2wav_len or idx2feat-len that contains the length of your reference speech
            mixup_number: int = 1
                The number of randomly-chosen speaker embedding vectors used for feature mixup.
            same_gender: bool = True
                Whether to conduct embedding mixup for the speakers with the same gender.
            tgt_gender: str = None
                The target gender used for the reference speech filtering.
            gender_info: List[str] or str = None
                The metadata file used for gender filtering. If not given, the file named 'idx2gen' in the directory of
                spk_feat will be used.
            super_conf: Dict
                The argument used for initialization of the super class

        """
        super(RandomSpkFeatDataset, self).dataset_init_fn(**super_conf)

        assert spk_feat is not None, f"spk_feat cannot be None. Please specify it in {self.__class__.__name__}!"
        assert isinstance(mixup_number, int) and mixup_number >= 1, \
            f"mixup_number must be a positive integer, but got {mixup_number}!"
        self.mixup_number = mixup_number
        self.same_gender = same_gender

        # List[str] or str -> List[str]
        if not isinstance(spk_feat, List):
            spk_feat = [spk_feat]
        metadata_dir = [os.path.dirname(s_f) for s_f in spk_feat]
        # speaker embedding file reading, List[str] -> Dict[str, str]
        self.spk_feat_dict = load_idx2data_file(spk_feat)

        # load the gender information
        if gender_info is None:
            gender_info = [os.path.join(m_d, 'idx2gen') for m_d in metadata_dir]
        self.gender_info_dict = load_idx2data_file(gender_info)

        # filter out the reference speech with the opposite gender
        if tgt_gender is not None:
            assert tgt_gender in ['M', 'F'], f"Your input tgt_gender must be one of 'M' or 'F', but got {tgt_gender}!"

            # check whether the keys of spk_feat and ref_len match each other
            spk_feat_keys, gender_info_keys = set(self.spk_feat_dict.keys()), set(self.gender_info_dict.keys())
            redundant_keys = spk_feat_keys.difference(gender_info_keys)
            assert len(redundant_keys) == 0, \
                f"There are {len(redundant_keys)} keys that exist in spk_feat but not in gender_info! " \
                f"Please check your data_cfg."

            self.gender_info_dict = {key: value for key, value in self.gender_info_dict.items() if value == tgt_gender}
            self.spk_feat_dict = {key: value for key, value in self.spk_feat_dict.items() if key in self.gender_info_dict.keys()}

        # filter out the short reference speech
        if min_ref_len is not None:
            if isinstance(min_ref_len, float):
                min_ref_len = int(min_ref_len)
            assert isinstance(min_ref_len, int) and min_ref_len > 0,\
                f"min_ref_len must be given as a positive integer, but got {min_ref_len}"

            # reference length file reading, List[str] or str -> Dict[str, str]
            if ref_len is None:
                ref_len = [os.path.join(m_d, 'idx2wav_len') for m_d in metadata_dir]
            self.ref_len_dict = load_idx2data_file(ref_len, data_type=int)

            # check whether the keys of spk_feat and ref_len match each other
            spk_feat_keys, ref_len_keys = set(self.spk_feat_dict.keys()), set(self.ref_len_dict.keys())
            redundant_keys = spk_feat_keys.difference(ref_len_keys)
            assert len(redundant_keys) == 0, \
                f"There are {len(redundant_keys)} keys that exist in spk_feat but not in ref_len! " \
                f"Please check your data_cfg."

            self.ref_len_dict = {key: value for key, value in self.ref_len_dict.items() if value > min_ref_len}
            self.spk_feat_dict = {key: value for key, value in self.spk_feat_dict.items() if key in self.ref_len_dict.keys()}

        # register the list of available speaker embedding features
        self.spk_feat_list = list(self.spk_feat_dict.keys())
        self.spk_feat_num = len(self.spk_feat_list)

    def extract_main_data_fn(self, main_data: Dict[str, str]) -> Dict[str, Any]:
        """
        This hook function randomly pick up a speaker embedding feature from the given spk_feat file as the reference.
        The randomness is controlled by the `seed` you give in the exp_cfg.

        """
        assert 'spk_ids' not in main_data.keys(), \
            f"Please don't give spk_ids to main_data of {self.__class__.__name__}. " \
            "This Dataset is used to evaluate open-set multi-speaker TTS that uses external speaker embedding."
        assert 'spk_feat' not in main_data.keys(), \
            f"Please don't give spk_feat to main_data of {self.__class__.__name__}. " \
            f"Your spk_feat should be given outside the main_data."

        # process 'feat' and 'text' by the parent class
        main_data = super(RandomSpkFeatDataset, self).extract_main_data_fn(main_data)

        # used for the same gender mixup
        ref_gender = None
        chosen_spk_feat_ids = []
        while len(chosen_spk_feat_ids) < self.mixup_number:
            # randomly pick up a speaker embedding feature vector
            spk_feat_idx = self.spk_feat_list[random.randint(0, self.spk_feat_num - 1)]
            if self.same_gender:
                curr_gender = self.gender_info_dict[spk_feat_idx]
                # record the current gender as the reference gender
                if ref_gender is None:
                    ref_gender = curr_gender
                # go to the next one if the current gender is different from the reference gender
                elif curr_gender != ref_gender:
                    continue

            if 'spk_feat' not in main_data.keys():
                main_data['spk_feat'] = read_data_by_path(self.spk_feat_dict[spk_feat_idx], return_tensor=True)
            else:
                main_data['spk_feat'] += read_data_by_path(self.spk_feat_dict[spk_feat_idx], return_tensor=True)
            chosen_spk_feat_ids.append(spk_feat_idx)

        # take the average of the chose speaker embedding features
        main_data['spk_feat'] /= self.mixup_number
        main_data['spk_feat_ids'] = '+'.join(chosen_spk_feat_ids)
        return main_data
