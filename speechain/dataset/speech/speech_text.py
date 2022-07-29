"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import torchaudio
import numpy as np
import torch

from speechain.dataset.abs import Dataset


class SpeechTextDataset(Dataset):
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


    def read_data_label_files(self, data_file: str, label_file: str):
        """
        In SpeechTextDataset:

        1. self.src_data is the dictionary of the speech data. Each key-value pair corresponds to an utterance where
        the key is its index and the value is the absolution path of the audio file.

        2. self.tgt_label is the dictionary of the text data. Each key-value pair corresponds to an sentence where the key
        is its index and the value is the string of the sentence.

        Here, numpy.loadtxt() is used to read the file contents.

        Args:
            data_file:
            label_file:

        Returns:

        """
        # read feat_scp file that contains the absolute paths of audio files
        # str -> np.ndarray
        feat_scp = np.loadtxt(data_file, dtype=str, delimiter=" ")
        # np.ndarray -> Dict
        feat_scp = dict(zip(feat_scp[:, 0], feat_scp[:, 1]))

        # read text file that contains strings of transcripts
        # str -> np.ndarray. Since there are blanks between words, the segmentation is first done by '\n'.
        text = np.loadtxt(label_file, dtype=str, delimiter="\n")
        # Then, the index and sentence are separated by the first blank
        text = np.stack(np.chararray.split(text, maxsplit=1))
        # np.ndarray -> Dict
        text = dict(zip(text[:, 0], text[:, 1]))

        return feat_scp, text


    def collate_fn(self, batch):
        """
        The loaded feature vectors will be arranged into a single matrix with 0 padding at the end of short vectors.
        Text data remains unprocessed strings and the tokenization will be done later in the model.

        """
        # para init
        batch_size, feat_dim = len(batch), batch[0][0].shape[-1]

        # acoustic feature padding, feat.dtype needs to match the type of model parameters (torch.float32)
        feat_len = torch.LongTensor([ele[1] for ele in batch])
        feat = torch.zeros((batch_size, feat_len.max().item(), feat_dim), dtype=torch.float32)

        # overwrite the padding matrix with each feat vector
        for i in range(batch_size):
            # process feat data based on data type
            if isinstance(batch[i][0], np.ndarray):
                feat[i][:feat_len[i]] = torch.from_numpy(batch[i][0])
            elif isinstance(batch[i][0], torch.Tensor):
                feat[i][:feat_len[i]] = batch[i][0]
            # only support np.ndarray and torch.Tensor now
            else:
                raise TypeError

        # return the data as a dictionary, text data remains unprocessed strings
        return dict(
            feat=feat, feat_len=feat_len,
            text=[ele[2] for ele in batch]
        )


    def __getitem__(self, index):
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
            feat = torchaudio.load(self.src_data[index], channels_first=False, normalize=False)[0]
        # extract acoustic features from the disk
        elif self.feat_type == 'feat':
            feat = np.load(self.src_data[index])['feat']
        # currently only support 'raw' and 'feat' types
        else:
            raise ValueError

        # extract text from the memory
        text = self.tgt_label[index]

        # return the feat length for padding, the text length is not returned
        return feat, feat.shape[0], text