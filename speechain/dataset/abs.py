"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
from typing import List, Dict, Any, Tuple

import torch
from abc import ABC, abstractmethod


class Dataset(torch.utils.data.Dataset, ABC):
    """
    Dataset is the base class for all dataset classes in this toolkit. The main job of the dataset is loading
    the raw data from the disk into the pipeline.

    If you want to inherit this class to make your own implementations, you need to override all abstract methods of this
    class.

    """
    def __init__(self,
                 src_data: str or List[str] = None,
                 tgt_label: str or List[str] = None,
                 meta_info: Dict[str, str or List[str]] = None,
                 **dataset_conf):
        """
        If we want to initialize some member variables in our implementations, it's mandatory for us to first write
        the code 'super(Dataset, self).__init__()'.

        It's very troublesome to type this code every time when we make a new implementation, so we include this
        command into the initialization function fo the base class and make a new interface function to do the job of
        dataset initialization.

        Args:
            src_data: str or List[str]
                The name of the source data files that contain the input data for model training. This argument can be
                given in either a single string or a list of strings. If a list is given, all the given feat_scp files
                will be mixed into a single dictionary.
            tgt_label: str or List[str]
                The name of the target label files that contain the ourput labels for model training. This argument can
                be given in either a single string or a list of strings. If a list is given, all the given text files
                will be mixed into a single dictionary.
            meta_info: Dict[str, str or List[str]]

            **dataset_conf:
                The configuration for initializing your dataset implementation.
        """
        super(Dataset, self).__init__()

        # --- Source Data and Target Label Reading and Processing --- #
        assert (src_data is not None) or (tgt_label is not None), \
            "A Dataset must contain at least one of src_data and tgt_label, or both of them!"

        # source data init, make sure that self.src_data is a list of str where each str corresponds to a file
        if src_data is not None:
            assert isinstance(src_data, (str, list)), \
                f"src_data must be given in str or List[str], but got type(src_data)={type(src_data)}"
            self.src_data = [src_data] if isinstance(src_data, str) else src_data

            # transform self.src_data into a single dictionary
            # data file reading, List[str] -> List[Dict[str, str]]
            self.src_data = [self.read_data_file(_data_file) for _data_file in self.src_data]
            # data Dict combination, List[Dict[str, str]] -> Dict[str, str]
            self.src_data = {key: value for _data_dict in self.src_data for key, value in _data_dict.items()}
            # sort the key-value items in the dict by their key names for the scenario of multiple data sources
            self.src_data = dict(sorted(self.src_data.items(), key=lambda x: x[0]))
        else:
            self.src_data = None

        # target label init, make sure that self.tgt_label is a list of str where each str corresponds to a file
        if tgt_label is not None:
            assert isinstance(tgt_label, (str, list)), \
                f"tgt_label must be given in str or List[str], but got type(tgt_label)={type(tgt_label)}"
            self.tgt_label = [tgt_label] if isinstance(tgt_label, str) else tgt_label

            # transform self.tgt_label into a single dictionary
            # label file reading, List[str] -> List[Dict[str, str]]
            self.tgt_label = [self.read_label_file(_label_file) for _label_file in self.tgt_label]
            # label Dict combination, List[Dict[str, str]] -> Dict[str, str]
            self.tgt_label = {key: value for _label_dict in self.tgt_label for key, value in _label_dict.items()}
            # sort the key-value items in the dict by their key names for the scenario of multiple data sources
            self.tgt_label = dict(sorted(self.tgt_label.items(), key=lambda x: x[0]))
        else:
            self.tgt_label = None


        # --- Extra Information Reading and Processing --- #
        self.meta_info = None
        if meta_info is not None:
            # extra information initialization
            # make sure that self.meta_info is a Dict of List and each List corresponds to a kind of information
            assert isinstance(meta_info, Dict), \
                f"meta_info must be given in Dict, but got type(meta_info)={type(meta_info)}"
            self.meta_info = {key: [value] if isinstance(value, str) else value for key, value in meta_info.items()}

            # loop each kind of information
            for meta_type in self.meta_info.keys():
                # information file reading, List[str] -> List[Dict[str, str]]
                self.meta_info[meta_type] = [self.read_meta_file(_meta_file, meta_type=meta_type)
                                             for _meta_file in self.meta_info[meta_type]]
                # information Dict combination, List[Dict[str, str]] -> Dict[str, str]
                self.meta_info[meta_type] = {key: value for _meta_dict in self.meta_info[meta_type]
                                             for key, value in _meta_dict.items()}
                # sort the key-value items in the dict by their key names for the scenario of multiple data sources
                self.meta_info[meta_type] = dict(sorted(self.meta_info[meta_type].items(), key=lambda x: x[0]))

        # --- Dict keys mismatch checking of self.src_data, self.tgt_label, and self.meta_info --- #
        # combine the key lists of all data sources
        dict_keys = []
        # collect the data index keys from the source data
        if self.src_data is not None:
            src_data_keys = set(self.src_data.keys())
            dict_keys.append(src_data_keys)
        else:
            src_data_keys = None
        # collect the data index keys from the target labels
        if self.tgt_label is not None:
            tgt_label_keys = set(self.tgt_label.keys())
            dict_keys.append(tgt_label_keys)
        else:
            tgt_label_keys = None
        # collect the data index keys from the metadata information
        if self.meta_info is not None:
            meta_info_keys = {key: set(value.keys()) for key, value in self.meta_info.items()}
            dict_keys += list(meta_info_keys.values())
        else:
            meta_info_keys = None

        # get the intersection of the key lists of all data sources
        key_intersection = dict_keys[0]
        for i in range(1, len(dict_keys)):
            key_intersection &= dict_keys[i]

        # remove the redundant key-value pairs that are in self.src_data but not in the intersection
        if src_data_keys is not None:
            for redundant_key in src_data_keys.difference(key_intersection):
                self.src_data.pop(redundant_key)
        # remove the redundant key-value pairs that are in self.tgt_label but not in the intersection
        if tgt_label_keys is not None:
            for redundant_key in tgt_label_keys.difference(key_intersection):
                self.tgt_label.pop(redundant_key)
        # remove the redundant key-value pairs that are in self.meta_info but not in the intersection
        if self.meta_info is not None:
            # loop each type of extra information
            for meta_type in self.meta_info.keys():
                # remove the redundant key-value pairs that are in self.tgt_label but not in the intersection
                for redundant_key in meta_info_keys[meta_type].difference(key_intersection):
                    self.meta_info[meta_type].pop(redundant_key)

        # initialize the customized part of the dataset
        self.dataset_init(**dataset_conf)


    @abstractmethod
    def dataset_init(self, **dataset_conf):
        """
        This function initializes the customized part of your dataset implementations.

        Args:
            **dataset_conf:

        """
        raise NotImplementedError


    @abstractmethod
    def read_data_file(self, data_file: str) -> Dict[str, str]:
        """
        In this function, you need to read the data files you give in 'src_data' into memory.

        The input of this function is one string that indicate the absolute paths of a data file.
        You need to first read the contents of the file into memory, and then transform the contents into a Dict.

        Finally, this Dict will be returned.

        Here is an implementation example:
        >>> from speechain.dataset.abs import Dataset
        >>> class MyDataset(Dataset):
        ...     def read_data_file(self, data_file: str):
        ...         # read the content of the data file
        ...         data = read_func(data_file)
        ...         # transform the read information into a Dict
        ...         data = dict(transform(data))
        ...         return data

        For more details, please refer to the SpeechTextDataset in ./speechain/dataset/speech/speech_text.py as an example.

        Args:
            data_file: str
                The absolute path of a data file.

        Returns:
            The Dict of the contents of the input data file.

        """
        raise NotImplementedError


    @abstractmethod
    def read_label_file(self, label_file: str) -> Dict[str, str]:
        """
        In this function, you need to read the label files you give in 'tgt_label' into memory.

        The input of this function is one strings that indicate the absolute paths of a label file.
        You need to first read the contents of the file into memory, and then transform the contents into a Dict.

        Finally, this Dict will be returned.

        Here is an implementation example:
        >>> from speechain.dataset.abs import Dataset
        >>> class MyDataset(Dataset):
        ...     def read_label_file(self, label_file: str):
        ...         # read the content of the label file
        ...         label = read_func(label_file)
        ...         # transform the read information into a Dict
        ...         label = dict(transform(label))
        ...         return label

        For more details, please refer to the SpeechTextDataset in ./speechain/dataset/speech/speech_text.py as an example.

        Args:
            label_file: str
                The absolute path of a label file.

        Returns:
            The Dict of the contents of the input label file.

        """
        raise NotImplementedError


    def read_meta_file(self, meta_file: str, meta_type: str) -> Dict[str, str]:
        """
        In this function, you need to read the info  files you give in 'meta_info' into memory.

        The input of this function is one string that indicate the absolute paths of a info file.
        You need to first read the contents of the file into memory, and then transform the contents into a Dict.

        Finally, this Dict will be returned.

        This interface is not mandatory to be overridden unless you would like to introduce meta information in your batches.

        Here is an implementation example:
        >>> from speechain.dataset.abs import Dataset
        >>> class MyDataset(Dataset):
        ...     def read_meta_file(self, meta_file: str, meta_type: str):
        ...         if meta_type == 'type1':
        ...             # read the content of the meta file
        ...             meta = read_func1(meta_file)
        ...             # transform the read information into a Dict
        ...             meta = dict(transform1(meta))
        ...         elif meta_type == 'type2':
        ...             # read the content of the meta file
        ...             meta = read_func2(meta_file)
        ...             # transform the read information into a Dict
        ...             meta = dict(transform2(meta))
        ...         return meta

        For more details, please refer to the SpeechTextDataset in ./speechain/dataset/speech/speech_text.py as an example.

        Args:
            meta_file: str
                The absolute path of a info file.
            meta_type: str
                The type of the information in the current input info file.

        Returns:
            The Dict of the contents of the input info file.

        """
        raise NotImplementedError


    def get_data_index(self) -> List[str]:
        """

        Returns:
            returns the list of the indices of all data samples in this dataset.

        """
        if self.src_data is not None:
            return list(self.src_data.keys())
        return list(self.tgt_label.keys())


    def remove_data_by_index(self, index: str):
        """
        This function removes the corresponding data sample from the dataset by the given index. Mainly used for
        solving the index mismatch of data samples with the iterator during training.

        """
        if self.src_data is not None:
            self.src_data.pop(index)

        if self.tgt_label is not None:
            self.tgt_label.pop(index)

        if self.meta_info is not None:
            for meta_type in self.meta_info.keys():
                if index in self.meta_info[meta_type].keys():
                    self.meta_info[meta_type].pop(index)


    @abstractmethod
    def __getitem__(self, index) -> Dict[str, Any]:
        """
        This function decides how to load the raw data from the disk by the given index from the Dataloader.

        """
        raise NotImplementedError


    @abstractmethod
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        This function decides how to preprocess a batch of data generated by the Dataloader before giving it to the model.

        Sometimes, the data cannot be directly used by the model after reading the data from the disk. For example,
        the utterances and sentences used for training a ASR model may have different lengths, so we need to do the
        padding operations to make them equal in length.

        This function should return the processed batch data in a Dict.

        Args:
            batch: The tuple of data samples loaded from the disk

        """
        raise NotImplementedError
