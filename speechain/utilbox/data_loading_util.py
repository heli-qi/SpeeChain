"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.11
"""
import numpy as np
import h5py
import os
import torch
import soundfile as sf


def read_data_by_path(data_path: str, return_tensor: bool = False):
    """

    Args:
        data_path:
        return_tensor:

    Returns:

    """
    # get the folder directory and data file name
    folder_path, data_file = os.path.dirname(data_path), os.path.basename(data_path)

    # ':' means that the data is stored in a compressed chunk file
    if ':' in data_file:
        assert len(data_file.split(':')) == 2
        chunk_file, data_idx = data_file.split(':')
        chunk_path = os.path.join(folder_path, chunk_file)

        # read data by its extension
        chunk_ext = chunk_file.split('.')[-1]
        if chunk_ext == 'npz':
            data = np.load(chunk_path)[data_idx]
        elif chunk_ext == 'hdf5':
            with h5py.File(chunk_path, 'r') as reader:
                data = np.array(reader[data_idx])
        else:
            raise NotImplementedError

    # without ':' means that the data is stored in an individual file
    else:
        # read data by its extension
        data_ext = data_file.split('.')[-1]
        if data_ext == 'npy':
            data = np.load(data_path)
        elif data_ext in ['wav', 'flac']:
            data = sf.read(data_path)[0]
        else:
            raise NotImplementedError

    if return_tensor:
        return torch.tensor(data)
    else:
        return data
