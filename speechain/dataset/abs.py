"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
from typing import List, Dict, Any

from torch.utils.data import Dataset
from abc import ABC, abstractmethod


class Dataset(Dataset, ABC):
    """
    Dataset is the base class for all dataset classes in this toolkit. The main job of the dataset is loading
    the raw data from the disk into the pipeline.

    If you want to inherit this class to make your own implementations, you need to override all abstract methods of this
    class.

    """
    def __init__(self, src_data: str or List[str], tgt_label: str or List[str], **dataset_conf):
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
            **dataset_conf:
                The configuration for initializing your dataset implementation.
        """
        super(Dataset, self).__init__()

        # source data init, make sure that self.src_data is a list of str where each str corresponds to a file
        assert isinstance(src_data, (str, list)), \
            f"src_data must be given in str or List[str], but got type(src_data)={type(src_data)}"
        self.src_data = [src_data] if isinstance(src_data, str) else src_data

        # target label init, make sure that self.tgt_label is a list of str where each str corresponds to a file
        assert isinstance(tgt_label, (str, list)), \
            f"tgt_label must be given in str or List[str], but got type(tgt_label)={type(tgt_label)}"
        self.tgt_label = [tgt_label] if isinstance(tgt_label, str) else tgt_label

        # the number of src_data files should equal to that of tgt_label files.
        assert len(self.src_data) == len(self.tgt_label), \
            f"The number of src_data files should equal to that of tgt_label files, " \
            f"but got len(self.src_data)={len(self.src_data)} and len(self.tgt_label)={len(self.tgt_label)}."

        # read the files in self.src_data and self.tgt_label and turn them into Dict
        for i, (data_file, label_file) in enumerate(zip(self.src_data, self.tgt_label)):
            self.src_data[i], self.tgt_label[i] = self.read_data_label_files(data_file, label_file)
            assert isinstance(self.src_data[i], Dict) and isinstance(self.tgt_label[i], Dict), \
                f"After self.read_data_and_label(), the elements of self.src_data and self.tgt_label must be Dict, " \
                f"but got type(self.src_data[{i}])={type(self.src_data[i])} and type(self.tgt_label[{i}])={type(self.tgt_label[i])}."

        # transform self.src_data and self.tgt_label into a single dictionary
        self.src_data = {key: value for _data_dict in self.src_data for key, value in _data_dict.items()}
        self.tgt_label = {key: value for _label_dict in self.tgt_label for key, value in _label_dict.items()}

        # check whether the keys of self.src_data and self.tgt_label match with each other
        src_data_keys, tgt_label_keys = set(self.src_data.keys()), set(self.tgt_label.keys())
        # remove the redundant key-value pairs that are in self.src_data but not in self.tgt_label
        for redundant_key in src_data_keys.difference(tgt_label_keys):
            self.src_data.pop(redundant_key)
        # remove the redundant key-value pairs that are in self.tgt_label but not in self.src_data
        for redundant_key in tgt_label_keys.difference(src_data_keys):
            self.tgt_label.pop(redundant_key)

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
    def read_data_label_files(self, data_file: str, label_file: str) -> (Dict[str, Any], Dict[str, Any]):
        """
        In this function, you need to read the data and label files you give in 'src_data' and 'tgt_label' into
        memory.

        The input of this function is two strings that indicate the absolute paths of a data file and a label file.
        You need to first read the contents of each file into memory, and then transform the contents into a Dict.

        Finally, these two Dict are returned. Note that the Dict of the data file is in the first place and the Dict
        of the label file is in the second place.

        Here is an implementation example:
        >>> from speechain.dataset.abs import Dataset
        >>> class MyDataset(Dataset):
        ...     def read_data_label_files(self, data_file: str, label_file: str):
        ...         # read the content of the data file
        ...         data = read_func(data_file)
        ...         # transform the read information into a Dict
        ...         data = dict(transform(data))
        ...         # read the content of the label file
        ...         label = read_func(label_file)
        ...         # transform the read information into a Dict
        ...         label = dict(transform(label))
        ...         # make sure the order of your return values are as below
        ...         return data, label

        For more details, please refer to the SpeechTextDataset in ./speechain/dataset/speech/speech_text.py as an example.

        Args:
            data_file: str
                The absolute path of a data file.
            label_file: str
                The absolute path of a label file.

        Returns:
            The Dict of the contents of the input data file, The Dict of the contents of the input label file.

        """
        raise NotImplementedError


    def get_data_index(self) -> List[str]:
        """

        Returns:
            returns the list of the indices of all data samples in this dataset.

        """
        return list(self.src_data.keys())


    def remove_data_by_index(self, index: str):
        """
        This function removes the corresponding data sample from the dataset by the given index. Mainly used for
        solving the index mismatch of data samples during training.

        """
        self.src_data.pop(index)
        self.tgt_label.pop(index)


    @abstractmethod
    def __getitem__(self, index):
        """
        This function decides how to load the raw data from the disk by the given index from the Dataloader.

        """
        raise NotImplementedError


    @abstractmethod
    def collate_fn(self, batch) -> Dict[str, Any]:
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
