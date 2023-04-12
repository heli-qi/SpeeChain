"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.11
"""
import warnings
import json
import numpy as np
import h5py
import os
import torch
import soundfile as sf

from typing import Dict, List, Any

from speechain.utilbox.import_util import parse_path_args


def read_data_by_path(data_path: str, return_tensor: bool = False, return_sample_rate: bool = False) \
        -> np.ndarray or torch.Tensor:
    """
    This function reads data from the file in the specified path, considering the file format and extension.

    Args:
        data_path (str):
            The path where the data file to be read is placed.
        return_tensor (bool, optional):
            Whether to return data as torch.Tensor. Defaults to False.
        return_sample_rate (bool, optional):
            Whether to return the sample rate. Defaults to False.

    Returns:
        np.ndarray or torch.Tensor:
            Array-like data. If return_tensor is False, the data type will be numpy.ndarray;
            otherwise, the data type will be torch.Tensor.

    Raises:
        NotImplementedError: If the file format is not supported, this error is raised.

    """
    # get the folder directory and data file name
    folder_path, data_file = os.path.dirname(data_path), os.path.basename(data_path)
    sample_rate = None
    # ':' means that the data is stored in a compressed chunk file
    if ':' in data_file:
        assert len(data_file.split(':')) == 2
        chunk_file, data_idx = data_file.split(':')
        chunk_path = os.path.join(folder_path, chunk_file)

        # read data by its extension
        chunk_ext = chunk_file.split('.')[-1].lower()
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
        data_ext = data_file.split('.')[-1].lower()
        if data_ext == 'npy':
            data = np.load(data_path)
        elif data_ext == 'npz':
            npz_dict = np.load(data_path)
            data, sample_rate = npz_dict['feat'], npz_dict['sample_rate']
        elif data_ext in ['wav', 'flac']:
            # There are 3 ways to extract waveforms from the disk, no large difference in loaded values.
            # The no.2 method by librosa consumes a little more time than the others.
            # Among them, torchaudio.load() directly gives torch.Tensor.
            # 1. soundfile.read(self.src_data[index], always_2d=True, dtype='float32')[0]
            # 2. librosa.core.load(self.src_data[index], sr=self.sample_rate)[0].reshape(-1, 1)
            # 3. torchaudio.load(self.src_data[index], channels_first=False, normalize=False)[0]
            data, sample_rate = sf.read(data_path, always_2d=True, dtype='float32')
        else:
            raise NotImplementedError

    if return_tensor:
        data = torch.tensor(data)

    if return_sample_rate:
        return data, sample_rate
    else:
        return data


def load_idx2data_file(file_path: str or List[str], data_type: type = str, separator: str = ' ',
                       do_separate: bool = True) -> Dict[str, Any]:
    """
    Load a dictionary from a file or a list of files containing key-value pairs,
    where the key is the index of a data instance and the value is the target data.

    Args:
        file_path (str or List[str]):
            Absolute path of the file(s) to be loaded.
        data_type (type, optional):
            The data type of the values in the returned dictionary.
            It should be a Python built-in data type. Defaults to str.
        separator (str, optional):
            The separator between the data instance index and the data value in each line of the file.
            Defaults to ' '.
        do_separate (bool, optional):
            Whether to separate each row by the given separator. Defaults to True.

    Returns:
        Dict[str, Any]: A dictionary containing key-value pairs, where the key is the index of a data instance and
        the value is the target data.

    Raises:
        AssertionError: If the given file does not exist, this error is raised.
    """

    def load_single_file(_file_path: str):
        assert os.path.exists(_file_path), f"{_file_path} doesn't exist!"

        # non-json metadata file
        if not _file_path.endswith('.json'):
            # str -> (n,) List. First read the content of the given file one line a time.
            with open(_file_path, mode='r') as f:
                data = f.readlines()
            # (n,) List -> (n, 2) List or (n,) List. Then, the index and sentence are separated by the first blank
            data = [row.replace('\n', '').split(separator, 1) if do_separate else row.replace('\n', '') for row in data]

            if isinstance(data[0], List) and len(data[0]) > 1:
                # (n, 2) List -> Dict[str, data_type]
                return {key: data_type(value) for key, value in data}
            else:
                # (n,) List -> Dict[str, data_type]
                return {key: data_type(value[0] if isinstance(value, List) else value) for key, value in enumerate(data)}
        # .json metadata file
        else:
            # str -> (n,) np.ndarray. First read the content of the .json file into a string.
            with open(_file_path, mode='r') as f:
                json_data = f.read()
                data = json.loads(json_data)

            for key, value in data.items():
                if isinstance(value, List):
                    data[key] = [data_type(i) for i in value]
                elif isinstance(value, Dict):
                    data[key] = {k: data_type(v) for k, v in value}
                else:
                    data[key] = data_type(value)
            return data

    if not isinstance(file_path, List):
        file_path = [file_path]
    else:
        assert isinstance(file_path, List)

    # data file reading, List[str] -> List[Dict[str, str]]
    idx2data_dict = [load_single_file(parse_path_args(f_p)) for f_p in file_path]
    # data Dict combination, List[Dict[str, str]] -> Dict[str, str]
    idx2data_dict = {key: value for _idx2data_dict in idx2data_dict for key, value in _idx2data_dict.items()}
    # sort the key-value items in the dict by their key names
    idx2data_dict = dict(sorted(idx2data_dict.items(), key=lambda x: x[0]))
    return idx2data_dict


def read_idx2data_file_to_dict(path_dict: Dict[str, str or List[str]]) -> (Dict[str, str], List[str]):
    """

    Args:
        path_dict: Dict[str, str or List[str]
            The path dictionary of the 'idx2XXX' files to be read. In each key-value item, the key is the data name and
            the value is the path of the target 'idx2XXX' files. Multiple file paths can be given in a list.

    Returns: (Dict[str, str], List[str])
        Both the result dictionary and the data index list will be returned.

    """
    # --- 1. Transformation from path to Dict --- #
    # preprocess Dict[str, str] into Dict[str, List[str]]
    path_dict = {key: [value] if isinstance(value, str) else value for key, value in path_dict.items()}

    # loop each kind of information
    output_dict = {key: load_idx2data_file(value) for key, value in path_dict.items()}

    # for data_name in path_dict.keys():
    #     # data file reading, List[str] -> List[Dict[str, str]]
    #     output_dict[data_name] = [load_idx2data_file(_data_file) for _data_file in path_dict[data_name]]
    #     # data Dict combination, List[Dict[str, str]] -> Dict[str, str]
    #     output_dict[data_name] = {key: value for _data_dict in output_dict[data_name]
    #                               for key, value in _data_dict.items()}
    #     # sort the key-value items in the dict by their key names
    #     output_dict[data_name] = dict(sorted(output_dict[data_name].items(), key=lambda x: x[0]))

    # --- 2. Dict Key Mismatch Checking --- #
    # combine the key lists of all data sources
    dict_keys = [set(data_dict.keys()) for data_dict in output_dict.values()]

    # get the intersection of the list of key sets
    key_intsec = dict_keys[0]
    for i in range(1, len(dict_keys)):
        key_intsec &= dict_keys[i]

    # remove the redundant key-value items from self.main_data
    for data_name in output_dict.keys():
        key_set = set(output_dict[data_name].keys())

        # delete the redundant key-value pairs
        redundant_keys = key_set.difference(key_intsec)
        if len(redundant_keys) > 0:
            warnings.warn(
                f"There are {len(redundant_keys)} redundant keys that exist in main_data[{data_name}] but others! "
                f"Please check your data_cfg to examine whether there is a problem.")
            for redund_key in redundant_keys:
                output_dict[data_name].pop(redund_key)

    return output_dict, sorted(key_intsec)


def search_file_in_subfolder(curr_query: str, tgt_match_fn: callable = None,
                             return_name: bool = False, return_sorted: bool = True):
    """
    Recursively searches for files in subdirectories of a given path, optionally filtering by name.

    Args:
        curr_query (str):
            The path of the directory to start the search from.
        tgt_match_fn (callable, optional):
            A function to filter files by name. It should take a string as input and return a boolean.
            If not provided, all files will be returned. Defaults to None.
        return_name (bool, optional):
            Whether to return the full file paths (default) or just the file names. Defaults to False.
        return_sorted:

    Returns:
        List[str]: A list of file paths (default) or file names that match the given criteria.

    Raises:
        RuntimeError: If the input query is the path of a file and it doesn't match the target.

    """
    candidates = []
    curr_query = parse_path_args(curr_query)

    # input query is the path of a file
    if os.path.isfile(curr_query):
        dir_name, node_name = os.path.dirname(curr_query), os.path.basename(curr_query)
        if tgt_match_fn is None or tgt_match_fn(node_name):
            # skip the soft link, only return the existing file
            if not os.path.islink(curr_query):
                return candidates + [curr_query] if not return_name else candidates + [node_name]
        else:
            raise RuntimeError(f"Your input query is the path of a file {curr_query} and it doesn't match your target!")

    # input query is the path of a folder
    for node_name in os.listdir(curr_query):
        node_path = os.path.join(curr_query, node_name)
        if os.path.isdir(node_path):
            # recursively search subdirectories
            candidates = candidates + search_file_in_subfolder(node_path, tgt_match_fn,
                                                               return_name=return_name, return_sorted=return_sorted)
        elif tgt_match_fn is None or tgt_match_fn(node_name):
            # skip the soft link, only return the existing file
            if not os.path.islink(node_path):
                candidates = candidates + [node_path] if not return_name else candidates + [node_name]

    return sorted(candidates) if return_sorted else candidates
