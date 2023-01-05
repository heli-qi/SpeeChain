import os
import torch
import numpy as np
import soundfile as sf

from typing import List

from speechain.utilbox.tensor_util import to_cpu


def save_data_by_format(file_format: str, save_path: str, file_name_list: List[str], file_content_list: List,
                        sample_rate: int = None):
    """

    Args:
        file_format:
        save_path:
        file_name_list:
        file_content_list:
        sample_rate:

    Returns:

    """
    # make sure the saving folder exists
    os.makedirs(save_path, exist_ok=True)

    # record all the data file paths with their names
    name2file_path = {}
    # save the feature vectors as .npy files (feature-only without the sampling rate)
    if file_format == 'npy':
        for name, feat in zip(file_name_list, file_content_list):
            if isinstance(feat, torch.Tensor):
                feat = to_cpu(feat, tgt='numpy')
            file_path = os.path.join(save_path, f'{name}.npy')
            np.save(file_path, feat.astype(np.float32))
            name2file_path[name] = file_path

    # save the feature vectors as .npz files (coupled with the sampling rate)
    elif file_format == 'npz':
        assert sample_rate is not None
        for name, feat in zip(file_name_list, file_content_list):
            if isinstance(feat, torch.Tensor):
                feat = to_cpu(feat, tgt='numpy')
            file_path = os.path.join(save_path, f'{name}.npz')
            np.savez(file_path, feat=feat.astype(np.float32), sample_rate=sample_rate)
            name2file_path[name] = file_path

    # save the waveforms as .wav files, sampling rate needs to be given in the result Dict as 'sample_rate'
    elif file_format == 'wav':
        assert sample_rate is not None
        for name, wav in zip(file_name_list, file_content_list):
            if isinstance(wav, torch.Tensor):
                wav = to_cpu(wav, tgt='numpy')
            file_path = os.path.join(save_path, f'{name}.wav')
            sf.write(file=file_path, data=wav, samplerate=sample_rate, format='WAV', subtype=sf.default_subtype('WAV'))
            name2file_path[name] = file_path

    # save the waveforms as .flac files, sampling rate needs to be given in the result Dict as 'sample_rate'
    elif file_format == 'flac':
        assert sample_rate is not None
        for name, wav in zip(file_name_list, file_content_list):
            if isinstance(wav, torch.Tensor):
                wav = to_cpu(wav, tgt='numpy')
            file_path = os.path.join(save_path, f'{name}.flac')
            sf.write(file=file_path, data=wav, samplerate=sample_rate, format='FLAC', subtype=sf.default_subtype('FLAC'))
            name2file_path[name] = file_path

    # other file formats are not supported now
    else:
        raise NotImplementedError

    return name2file_path
