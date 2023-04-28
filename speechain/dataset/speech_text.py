"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import os.path
import warnings

import numpy as np
import torch
import random

import torchaudio
from g2p_en import G2p
from typing import Dict, Any, List
from functools import partial
from speechain.tokenizer.g2p import abnormal_phns

from speechain.dataset.abs import Dataset
from speechain.utilbox.data_loading_util import read_data_by_path, load_idx2data_file
from speechain.utilbox.feat_util import convert_wav_to_pitch
from speechain.utilbox.train_util import get_min_indices_by_freq


class SpeechTextDataset(Dataset):
    """
    This Dataset subclass is mainly used by ASR and TTS models.
    In this subclass, each data instance is made up of an utterance and a sentence as well as the speaker information
    (speaker ID + speaker embedding feature).

    """
    def dataset_init_fn(self,
                        use_g2p: bool = False,
                        unk_mask_prob: float = 0.0,
                        use_speed_perturb: bool = False,
                        sample_rate: int = 16000,
                        perturb_range: List[float] = [0.9, 1.0, 1.1],
                        pitch_conf: Dict = None):
        """

        Args:
            # phoneme-related
            use_g2p: bool = False
                Whether to process the raw string by G2P. We don't recommend you to turn it on because on-the-fly
                transformer from string to phoneme list consumes a lot of CPU resources.
            # waveform-related
            use_speed_perturb: bool = False
                Whether to perturb the speed of the waveforms
            sample_rate: int = 16000
            perturb_range: List[float] = [0.9, 1.0, 1.1]
            # pitch-related
            pitch_conf: Dict = None
                The configuration given to convert_wav_to_pitch() for pitch extraction.
                If not given, pitch extraction will not be done on-the-fly.

        """
        # register sampling rate for later check
        self.sample_rate = sample_rate
        warnings.warn(f"The waveform sampling rate of {self.__class__.__name__} is set to {sample_rate}. "
                      f"All the extracted waveforms will be downsampled into {sample_rate} if needed. "
                      f"Please make sure that {sample_rate} is the same with your model! "
                      f"If this is not your target sampling rate, "
                      f"please change it by the key 'sample_rate' in the item 'dataset_conf' under 'data_cfg'. "
                      f"If you are training Language Model, you can ignore this warning.")

        assert 0 <= unk_mask_prob <= 1, f"unk_mask_prob should be a float number in [0, 1], but got {unk_mask_prob}!"
        self.unk_mask_prob = unk_mask_prob

        # phoneme extraction
        if use_g2p:
            self.g2p = G2p()

        if use_speed_perturb:
            self.perturb_range = perturb_range
            self.speed_resampler_list = \
                [torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=int(sample_rate * factor))
                 for factor in perturb_range]

        # pitch extraction
        if pitch_conf is not None:
            if 'sr' in pitch_conf.keys():
                assert pitch_conf['sr'] == self.sample_rate, \
                    f"The sampling rate in your given 'pitch_conf' ({pitch_conf['sr']}) is different from your " \
                    f"given sample_rate ({self.sample_rate})!"
            pitch_conf['sr'] = self.sample_rate
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

        # --- 1. Pad Speech Data and Stack them together --- #
        if 'feat' in batch_dict.keys():
            # para init
            feat_len = torch.LongTensor([ele.shape[0] for ele in batch_dict['feat']])
            batch_size, feat_maxlen, feat_dim = \
                len(batch_dict['feat']), feat_len.max().item(), batch_dict['feat'][0].shape[-1]

            # acoustic feature padding, feat.dtype needs to match the type of model parameters (torch.float32)
            feat = torch.zeros((batch_size, feat_maxlen, feat_dim), dtype=torch.float32)
            # overwrite the padding matrix with each feat vector
            for i in range(batch_size):
                # process feat data based on data type
                if isinstance(batch_dict['feat'][i], np.ndarray):
                    feat[i][:feat_len[i]] = torch.tensor(batch_dict['feat'][i])
                elif isinstance(batch_dict['feat'][i], torch.Tensor):
                    feat[i][:feat_len[i]] = batch_dict['feat'][i]
                # only support np.ndarray and torch.Tensor now
                else:
                    raise TypeError

            # update 'feat' and attach 'feat_len' for later model forward
            batch_dict['feat'] = feat
            batch_dict['feat_len'] = feat_len

        # --- 2. Pad Pitch Data and Stack them together --- #
        if 'pitch' in batch_dict.keys():
            # para init
            pitch_len = torch.LongTensor([ele.shape[0] for ele in batch_dict['pitch']])
            batch_size, pitch_maxlen = len(batch_dict['pitch']), pitch_len.max().item()

            # pitch padding, pitch.dtype needs to match the type of model parameters (torch.float32)
            pitch = torch.zeros((batch_size, pitch_maxlen), dtype=torch.float32)
            # overwrite the padding matrix with each pitch vector
            for i in range(batch_size):
                # process feat data based on data type
                if isinstance(batch_dict['pitch'][i], np.ndarray):
                    pitch[i][:pitch_len[i]] = torch.tensor(batch_dict['pitch'][i])
                elif isinstance(batch_dict['pitch'][i], torch.Tensor):
                    pitch[i][:pitch_len[i]] = batch_dict['pitch'][i]
                # only support np.ndarray and torch.Tensor now
                else:
                    raise TypeError

            batch_dict['pitch'] = pitch
            batch_dict['pitch_len'] = pitch_len

        # --- 3. Separate Phoneme Duration Data into Text Data and Duration Data --- #
        if 'duration' in batch_dict.keys():
            # para init
            batch_size, duration_len = \
                len(batch_dict['duration']), torch.LongTensor([len(ele) for ele in batch_dict['duration']])

            # duration padding, feat.dtype needs to match the type of model parameters (torch.float32)
            duration = torch.zeros((batch_size, duration_len.max().item()), dtype=torch.float32)
            # overwrite the padding matrix with each duration vector
            for i in range(batch_size):
                # process duration data based on data type
                if isinstance(batch_dict['duration'][i], (np.ndarray, List)):
                    duration[i][:duration_len[i]] = torch.tensor(batch_dict['duration'][i])
                elif isinstance(batch_dict['duration'][i], torch.Tensor):
                    duration[i][:duration_len[i]] = batch_dict['duration'][i]
                else:
                    raise TypeError(f"{self.__class__.name} only supports np.ndarray and torch.Tensor now!")

            # attach 'duration' and 'duration_len' for model forward
            batch_dict['duration'] = duration
            batch_dict['duration_len'] = duration_len

        # --- 4. Stack Speaker Embedding Feature together --- #
        if 'spk_feat' in batch_dict.keys():
            batch_dict['spk_feat'] = torch.stack(batch_dict['spk_feat'])

        return batch_dict

    def extract_main_data_fn(self, main_data: Dict) -> Dict[str, Any] or None:
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
        for key in main_data.keys():
            if key not in ['feat', 'text', 'duration', 'spk_ids', 'spk_feat']:
                raise RuntimeError(f"Unknown data name {key}! "
                                   f"For {self.__class__.__name__}, the key in 'main_data' must be one of "
                                   "'feat' (for paths of raw waveforms or acoustic features), "
                                   "'text' (for transcript text data), "
                                   "'duration' (for phoneme duration data), "
                                   "'spk_ids' (for speaker IDs), "
                                   "'spk_feat' (for speaker embedding features).")

        # --- 1. Speech Data Extraction --- #
        if 'feat' in main_data.keys():
            # read the selected data speech feature as a tensor by its path
            main_data['feat'], sample_rate = \
                read_data_by_path(main_data['feat'], return_sample_rate=True, return_tensor=True)

            # on-the-fly downsampling if extracted sampling rate is larger than the built-in one
            if sample_rate > self.sample_rate:
                if not hasattr(self, 'wav_resampler_dict'):
                    self.wav_resampler_dict = \
                        {sample_rate: torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)}
                main_data['feat'] = self.wav_resampler_dict[sample_rate](main_data['feat'].squeeze(-1)).unsqueeze(-1)
            # extracted waveforms could not have lower sampling rate than the built-in one
            elif sample_rate < self.sample_rate:
                raise RuntimeError(f'The current waveform has the lower sampling rate than {self.sample_rate}!')

            # perturb the speed of the extracted speech if specified
            if hasattr(self, 'speed_resampler_list'):
                assert sample_rate == self.sample_rate, \
                    f"Your given sample rate ({self.sample_rate}) is different from the real one gotten from the " \
                    f"waveform ({sample_rate})!"
                resampler_index = torch.randint(len(self.speed_resampler_list), (1,))[0]
                main_data['feat'] = self.speed_resampler_list[resampler_index](main_data['feat'].squeeze(-1)).unsqueeze(-1)

            # extract the pitch from the speech on-the-fly
            if hasattr(self, 'pitch_extract_fn'):
                try:
                    main_data['pitch'] = self.pitch_extract_fn(main_data['feat'])
                # IndexError means all the pitch values are unvoiced (=0.0)
                # return None to remove this utterance from the current batch
                except IndexError:
                    return None

        # --- 2. Transcript Text Extraction --- #
        if 'text' in main_data.keys():
            # text length is not returned because the text here is just a raw string
            assert isinstance(main_data['text'], str),\
                f"The 'text' data should be given as a string, but got {main_data['text']}"
            # for the text data in the format of a list
            if main_data['text'].startswith('[') and main_data['text'].endswith(']'):
                main_data['text'] = main_data['text'][1:-1]
                # split the text into individual tokens by a comma followed a blank
                main_data['text'] = main_data['text'].split(', ')
                # remove the single quote marks surrounding each token if needed
                main_data['text'] = [token[1:-1] if token.startswith('\'') and token.endswith('\'') else token
                                  for token in main_data['text']]
            # process the raw string by G2P if specified
            elif hasattr(self, 'g2p'):
                phn_list = self.g2p(main_data['text'])
                main_data['text'] = [phn if phn != ' ' else '<space>' for phn in phn_list if phn not in abnormal_phns]

        # --- 3. Phoneme Duration Extraction --- #
        if 'duration' in main_data.keys():
            # text length is not returned because the text here is just a raw string
            assert isinstance(main_data['duration'], str), \
                f"The 'duration' data should be given as a string, but got {main_data['duration']}"
            # for the text data in the format of a list
            if main_data['duration'].startswith('[') and main_data['duration'].endswith(']'):
                main_data['duration'] = main_data['duration'][1:-1]
                # split the text into individual tokens by a comma followed a blank
                main_data['duration'] = main_data['duration'].split(', ')
                # remove the single quote marks surrounding each token if needed
                main_data['duration'] = [float(duration[1:-1]) if duration.startswith('\'') and duration.endswith('\'')
                                         else float(duration) for duration in main_data['duration']]
            else:
                raise RuntimeError("The 'duration' string should be surrounded by a pair of square brackets!")

        # --- 4. Silence Trimming at the two ends --- #
        # trim the silence at two ends of the waveforms if the phoneme sequence starts or ends with spaces
        if ('text' in main_data.keys() and isinstance(main_data['text'], List)) and \
                (main_data['text'][0] == '<space>' or main_data['text'][-1] == '<space>'):
            # trim both feat and text
            if 'feat' in main_data.keys():
                assert 'duration' in main_data.keys(), \
                    "If you want to trim the silence at two ends of speech, " \
                    "please give 'duration' in 'main_data' of the item 'dataset_conf' under 'data_cfg'."
                front_trim_len, tail_trim_len, total_duration = 0, 0, sum(main_data['duration'])
                try:
                    # sum up all the silence tokens at the beginning
                    while main_data['text'][0] == '<space>':
                        front_trim_len += main_data['duration'][0]
                        main_data['text'], main_data['duration'] = main_data['text'][1:], main_data['duration'][1:]
                    # sum up all the silence tokens at the end
                    while main_data['text'][-1] == '<space>':
                        tail_trim_len += main_data['duration'][-1]
                        main_data['text'], main_data['duration'] = main_data['text'][:-1], main_data['duration'][:-1]
                # IndexError means the text is full of '<space>'
                # return None to remove this utterance from the current batch
                except IndexError:
                    return None

                # normalize the trimming lengths by the total duration length
                front_trim_len, tail_trim_len = front_trim_len / total_duration, tail_trim_len / total_duration
                # trim the extra silence in feat (waveforms or acoustic features)
                feat_start, feat_end = \
                    int(front_trim_len * len(main_data['feat'])), int(tail_trim_len * len(main_data['feat']))
                main_data['feat'] = main_data['feat'][feat_start:]
                if feat_end > 0:
                    main_data['feat'] = main_data['feat'][:-feat_end]

                # also trim the two ends of pitch values if extracted
                if 'pitch' in main_data.keys():
                    pitch_start, pitch_end = \
                        int(front_trim_len * len(main_data['pitch'])), int(tail_trim_len * len(main_data['pitch']))
                    main_data['pitch'] = main_data['pitch'][pitch_start:]
                    if pitch_end > 0:
                        main_data['pitch'] = main_data['pitch'][:-pitch_end]

            # only trim text if feat is not given
            else:
                try:
                    # sum up all the <space> tokens at the beginning
                    while main_data['text'][0] == '<space>':
                        main_data['text'] = main_data['text'][1:]
                        if 'duration' in main_data.keys():
                            main_data['duration'] = main_data['duration'][1:]
                    # sum up all the <space> tokens at the end
                    while main_data['text'][-1] == '<space>':
                        main_data['text'] = main_data['text'][:-1]
                        if 'duration' in main_data.keys():
                            main_data['duration'] = main_data['duration'][:-1]
                # IndexError means the text is full of '<space>'
                # return None to remove this utterance from the current batch
                except IndexError:
                    return None

        # --- 5. Randomly Masking the text data by unknown tokens (After silence trimming for data safety) --- #
        if self.unk_mask_prob > 0:
            assert 'text' in main_data.keys() and isinstance(main_data['text'], List), \
                "If you want to activate unk_mask_prob, text must be given in the 'main_date' tag as a token sequence."

            # Get the start and end indices of words based on the positions of space tokens
            space_indices = [i for i, token in enumerate(main_data['text']) if token == '<space>']
            word_start_indices, word_end_indices = \
                [0] + [s_i + 1 for s_i in space_indices], space_indices + [len(main_data['text'])]

            # Determine which words to mask
            word_mask_flags = np.random.rand(len(word_start_indices)) < self.unk_mask_prob

            _tmp_text, _tmp_duration = [], []
            for i in range(len(word_mask_flags)):
                # If the word should be masked, add an '<unk>' token
                if word_mask_flags[i]:
                    _tmp_text.append('<unk>')
                    if 'duration' in main_data.keys():
                        _sum_duration = sum(main_data['duration'][word_start_indices[i]: word_end_indices[i]])
                        _tmp_duration.append(round(_sum_duration, 2))

                # If the word shouldn't be masked, add the original tokens of the word
                else:
                    _tmp_text += main_data['text'][word_start_indices[i]: word_end_indices[i]]
                    if 'duration' in main_data.keys():
                        _tmp_duration += main_data['duration'][word_start_indices[i]: word_end_indices[i]]

                # Add space tokens and their durations between words, except for the last word
                if i != len(word_mask_flags) - 1:
                    _tmp_text.append(main_data['text'][word_end_indices[i]])
                    if 'duration' in main_data.keys():
                        _tmp_duration.append(main_data['duration'][word_end_indices[i]])

            # Update main_data with the new text and duration information
            main_data['text'] = _tmp_text
            if 'duration' in main_data.keys():
                main_data['duration'] = _tmp_duration

        # --- 6. Speaker ID Extraction --- #
        if 'spk_ids' in main_data.keys():
            # the speaker ID here is just a raw string
            assert isinstance(main_data['spk_ids'], str),\
                f"The 'spk_ids' data should be given as a string, but got {main_data['spk_ids']}"

        # --- 7. Speaker Embedding Feature --- #
        if 'spk_feat' in main_data.keys():
            # read the selected data speech feature as a tensor by its path
            main_data['spk_feat'] = read_data_by_path(main_data['spk_feat'], return_tensor=True)

        return main_data

    def __repr__(self):
        outputs = f'{self.__class__.__name__}(sample_rate={self.sample_rate}'
        if hasattr(self, 'g2p'):
            outputs += ', use_g2p=True'
        if hasattr(self, 'speed_resampler_list'):
            outputs += f', speed_perturb_range={self.perturb_range}'
        if hasattr(self, 'pitch_extract_fn'):
            outputs += ', pitch_extract=True'
        if self.unk_mask_prob > 0:
            outputs += f', unk_mask_prob={self.unk_mask_prob}'
        return outputs + ')'


class RandomSpkFeatDataset(SpeechTextDataset):
    """

    """
    def dataset_init_fn(self,
                        spk_feat: List[str] or str = None,
                        use_aver_feat: bool = False,
                        mixup_number: int = 1,
                        same_gender_mixup: bool = True,
                        tgt_gender: str = None,
                        **super_conf):

        super(RandomSpkFeatDataset, self).dataset_init_fn(**super_conf)

        assert spk_feat is not None, f"spk_feat cannot be None. Please specify it in {self.__class__.__name__}!"
        assert isinstance(mixup_number, int) and mixup_number >= 1, \
            f"mixup_number must be a positive integer, but got {mixup_number}!"
        self.mixup_number = mixup_number
        self.same_gender_mixup = same_gender_mixup

        # List[str] or str -> List[str]
        if not isinstance(spk_feat, List):
            spk_feat = [spk_feat]
        metadata_dir = [os.path.dirname(s_f) for s_f in spk_feat]
        spk_emb_model = [os.path.basename(s_f).split('2')[-1].split('_')[0] for s_f in spk_feat]

        # register the list of available speaker IDs
        self.idx2spk = load_idx2data_file([os.path.join(m_d, 'idx2spk') for m_d in metadata_dir])
        self.spk_ids_list = sorted(set(self.idx2spk.values()))
        self.spk_num = len(self.spk_ids_list)
        self.spk2freq = {spk_id: 0 for spk_id in self.spk_ids_list}

        # speaker embedding file reading, List[str] -> Dict[str, str]
        idx2spk_feat = load_idx2data_file(spk_feat)
        self.spk2spk_feat = \
            {spk_id: {spk_feat_id: idx2spk_feat[spk_feat_id] for spk_feat_id in idx2spk_feat.keys()
                      if self.idx2spk[spk_feat_id] == spk_id} for spk_id in self.spk_ids_list}
        if use_aver_feat:
            self.spk2aver_spk_feat = load_idx2data_file(
                [os.path.join(m_d, f'spk2aver_{s_e_m}_spk_feat') for m_d, s_e_m in zip(metadata_dir, spk_emb_model)])

        # load the gender information
        idx2gender = load_idx2data_file([os.path.join(m_d, 'idx2gen') for m_d in metadata_dir])
        self.spk2gender = {
            spk_id: [idx2gender[spk_feat_id] for spk_feat_id in idx2gender.keys()
                     if self.idx2spk[spk_feat_id] == spk_id][0] for spk_id in self.spk_ids_list}

        # filter out the reference speech with the opposite gender
        if tgt_gender is not None:
            assert tgt_gender in ['M', 'F'], f"Your input tgt_gender must be one of 'M' or 'F', but got {tgt_gender}!"
            self.spk2gender = {spk: gender for spk, gender in self.spk2gender.items() if gender == tgt_gender}
            self.spk2spk_feat = {spk: spk_feat for spk, spk_feat in self.spk2spk_feat.items() if spk in self.spk2gender.keys()}
            if hasattr(self, 'spk2aver_spk_feat'):
                self.spk2aver_spk_feat = {spk: aver_spk_feat for spk, aver_spk_feat in self.spk2aver_spk_feat.items()
                                          if spk in self.spk2gender.keys()}

    def extract_main_data_fn(self, main_data: Dict[str, str]) -> Dict[str, Any] or None:
        """
        This hook function randomly pick up a speaker embedding feature from the given spk_feat file as the reference.
        The randomness is controlled by the `seed` you give in the exp_cfg.

        """
        assert 'spk_ids' not in main_data.keys(), \
            f"Please don't give spk_ids to main_data of {self.__class__.__name__}. " \
            f"This Dataset is used to evaluate open-set multi-speaker TTS that uses external speaker embedding."
        assert 'spk_feat' not in main_data.keys(), \
            f"Please don't give spk_feat to main_data of {self.__class__.__name__}. " \
            f"Your spk_feat should be given outside the main_data."

        # process 'feat' and 'text' by the parent class
        main_data = super(RandomSpkFeatDataset, self).extract_main_data_fn(main_data)
        # None means empty batch received from the parent class
        if main_data is None:
            return main_data

        # used for the same gender mixup
        ref_gender = None
        chosen_spk_feat_ids, chosen_spk_ids = [], []
        while len(chosen_spk_feat_ids) < self.mixup_number:
            random_spk_id, self.spk2freq = get_min_indices_by_freq(
                self.spk2freq,  freq_weights=len(main_data['text']) if 'text' in main_data.keys() else None)
            random_spk_id = random_spk_id[0]
            if self.same_gender_mixup:
                curr_gender = self.spk2gender[random_spk_id]
                # record the current gender as the reference gender
                if ref_gender is None:
                    ref_gender = curr_gender
                # go to the next one if the current gender is different from the reference gender
                elif curr_gender != ref_gender:
                    continue

            # randomly pick up a speaker embedding feature vector
            spk_feat = self.spk2spk_feat[random_spk_id]
            spk_feat_id_list = list(spk_feat.keys())
            random_spk_feat_id = spk_feat_id_list[random.randint(0, len(spk_feat_id_list) - 1)]
            if not hasattr(self, 'spk2aver_spk_feat'):
                spk_feat = read_data_by_path(spk_feat[random_spk_feat_id], return_tensor=True)
            else:
                # randomly pick up a useless spk_feat_id for the same randomness results
                random_spk_feat_id = 'aver_spk_feat'
                spk_feat = read_data_by_path(self.spk2aver_spk_feat[random_spk_id], return_tensor=True)

            if 'spk_feat' not in main_data.keys():
                main_data['spk_feat'] = spk_feat
            else:
                main_data['spk_feat'] += spk_feat

            chosen_spk_feat_ids.append(random_spk_feat_id)
            chosen_spk_ids.append(random_spk_id)

        # take the average of the chose speaker embedding features
        if self.mixup_number > 1:
            main_data['spk_feat'] /= self.mixup_number
            # sort all the IDs of spk_feat and spk to make sure the naming uniqueness
            main_data['spk_feat_ids'] = '+'.join(sorted(chosen_spk_feat_ids))
            main_data['spk_ids'] = '+'.join(sorted(chosen_spk_ids))
        else:
            main_data['spk_feat_ids'] = chosen_spk_feat_ids[0]
            main_data['spk_ids'] = chosen_spk_ids[0]

        return main_data
