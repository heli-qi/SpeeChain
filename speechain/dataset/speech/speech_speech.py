"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
from typing import Dict, Any, List

import torchaudio
import numpy as np
import torch

from speechain.dataset.abs import Dataset


class SpeechSpeechDataset(Dataset):
    """
    The dataset that gives speech-text paired data. Each pair has an utterance and a sentence.
    This dataset implementation is mainly used by ASR and TTS models.

    """
    def dataset_init(self, feat_type: str):
        """

        Args:
            feat_type: str
                The type of raw data on the disk. Can be either 'raw' (audio waveforms) or 'feat' (acoustic feature).
                If feat_type = 'feat', the acoustic file must be stored as .npz files on the disk.

        """
        # input feature type recording
        self.feat_type = feat_type


    def read_data_file(self, data_file: str) -> Dict[str, str]:
        """
        self.src_data is the dictionary of the speech data. Each key-value pair corresponds to an utterance where
        the key is its index and the value is the absolution path of the audio file.

        Args:
            data_file:

        Returns:

        """
        # read feat_scp file that contains the absolute paths of audio files
        # str -> np.ndarray
        feat_scp = np.loadtxt(data_file, dtype=str, delimiter=" ")
        # ToDo(heli-qi): add duplicated keys checking codes
        # np.ndarray -> Dict[str, str]
        return dict(zip(feat_scp[:, 0], feat_scp[:, 1]))


    def read_label_file(self, label_file: str) -> Dict[str, str]:
        """
        self.tgt_label is the dictionary of the text data. Each key-value pair corresponds to an sentence where the key
        is its index and the value is the string of the sentence.

        Args:
            label_file:

        Returns:

        """
        feat_scp = np.loadtxt(label_file, dtype=str, delimiter=" ")
        return dict(zip(feat_scp[:, 0], feat_scp[:, 1]))


    def read_meta_file(self, meta_file: str, meta_type: str) -> Dict[str, str]:
        """

        Args:
            meta_file:
            meta_type:

        Returns:

        """
        if meta_type in ['speaker', 'gender']:
            # read feat_scp file that contains the absolute paths of audio files
            # str -> np.ndarray
            meta = np.loadtxt(meta_file, dtype=str, delimiter=" ")
            # ToDo(heli-qi): add duplicated keys checking codes
            # np.ndarray -> Dict[str, str]
            return dict(zip(meta[:, 0], meta[:, 1]))
        else:
            raise NotImplementedError


    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        The loaded feature vectors will be arranged into a single matrix with 0 padding at the end of short vectors.
        Text data remains unprocessed strings and the tokenization will be done later in the model.

        """
        #SRC
        # para init
        batch_size, feat_src_dim = len(batch), batch[0]['feat_src'].shape[-1]
        _, feat_tgt_dim = len(batch), batch[0]['feat_tgt'].shape[-1]


        # acoustic feature padding, feat.dtype needs to match the type of model parameters (torch.float32)
        feat_src_len = torch.LongTensor([ele['feat_src_len'] for ele in batch])
        feat_src = torch.zeros((batch_size, feat_src_len.max().item(), feat_src_dim), dtype=torch.float32)

        feat_tgt_len = torch.LongTensor([ele['feat_tgt_len'] for ele in batch])
        feat_tgt = torch.zeros((batch_size, feat_tgt_len.max().item(), feat_tgt_dim), dtype=torch.float32)


        # overwrite the padding matrix with each feat vector
        for i in range(batch_size):

            # process feat data based on data type
            #SRC
            if isinstance(batch[i]['feat_src'], np.ndarray):
                feat_src[i][:feat_src_len[i]] = torch.from_numpy(batch[i]['feat_src'])
            elif isinstance(batch[i]['feat_src'], torch.Tensor):
                feat_src[i][:feat_src_len[i]] = batch[i]['feat_src']
            # only support np.ndarray and torch.Tensor now
            else:
                raise TypeError

            #TGT
            if isinstance(batch[i]['feat_tgt'], np.ndarray):
                feat_tgt[i][:feat_tgt_len[i]] = torch.from_numpy(batch[i]['feat_tgt'])
            elif isinstance(batch[i]['feat_tgt'], torch.Tensor):
                feat_tgt[i][:feat_tgt_len[i]] = batch[i]['feat_tgt']
            # only support np.ndarray and torch.Tensor now
            else:
                raise TypeError


        # return the data as a dictionary, text data and meta data remains unprocessed
        return dict(
            feat_src=feat_src, feat_src_len=feat_src_len,
            feat_tgt=feat_tgt,feat_tgt_len=feat_tgt_len,
            **{key: [ele[key] for ele in batch] for key in batch[0].keys() if key not in ['feat_src', 'feat_src_len', 'feat_tgt', 'feat_tgt_len']}
        )


    def __getitem__(self, index) -> Dict[str, Any]:
        """
        The function that loads speech-text data from the disk. If the speech is in the form of raw waveforms,
        the last dimension should be expanded to 1 of raw speech for compatibility with acoustic feature

        There are 3 ways to extract waveforms from the disk, no large difference in loaded values.
        The no.2 method by librosa consumes a little more time than the others. Among them, torchaudio.load() directly
        gives torch.Tensor, so it is chosen in this function.
            1. feat = soundfile.read(self.feat_scp[index], always_2d=True, dtype='float32')[0]
            2. feat = librosa.core.load(self.feat_scp[index], sr=self.sample_rate)[0].reshape(-1, 1)
            3. feat = torchaudio.load(self.feat_scp[index], channels_first=False, normalize=False)[0]

        Args:
            index: int
                The index of the chosen speech-text pair.

        """
        # extract audio waveform from the disk
        if self.feat_type == 'raw':
            feat_src = torchaudio.load(self.src_data[index], channels_first=False, normalize=False)[0]
        # extract acoustic features from the disk
        elif self.feat_type == 'feat':
            feat_src = np.load(self.src_data[index])['feat']
        # currently only support 'raw' and 'feat' types
        else:
            raise ValueError

        # extract text from the memory
        feat_tgt = np.load(self.tgt_label[index])['feat']
        #text = self.tgt_label[index]

        # return the feat length for padding, the text length is not returned
        outputs = dict(
            feat_src=feat_src,
            feat_src_len=feat_src.shape[0],
            feat_tgt=feat_tgt,
            feat_tgt_len=feat_tgt.shape[0],
        )

        # return the meta information if given
        if self.meta_info is not None:
            outputs.update(
                {key: value[index] for key, value in self.meta_info.items()}
            )
        return outputs