"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import numpy as np
import torch

from torch.utils.data import DataLoader
from typing import Dict, List, Any
from abc import ABC, abstractmethod

from speechain.utilbox.import_util import import_class


class Iterator(ABC):
    """
    Iterator is the base class for all iterators in this toolkit. The main job of the iterator is giving a
    Dataloader in each epoch to provide the model with batch data. Each iterator has a built-in Dataset
    and its initialization is done automatically in the initialization function of this base class.

    """

    def __init__(self,
                 dataset_type: str,
                 dataset_conf: Dict,
                 batches_per_epoch: int = None,
                 is_descending: bool = None,
                 data_len: str or List[str] = None,
                 selection_mode: str = None,
                 selection_num: float or int = None,
                 shuffle: bool = True,
                 seed: int = 0,
                 num_workers: int = 1,
                 pin_memory: bool = True,
                 distributed: bool = False,
                 **iter_conf):
        """
        Dataset initialization is automatically done in this base class, so users only need to specify the dataset
        type and give its configuration here.

        Iterator initialization is implemented in the self.iter_init() of each child iterator. Each child iterator
        has its own strategy to generate batches.

        Args:
            dataset_type: str
                query string to pick up the target dataset object
            dataset_conf: Dict
                dataset configuration for its initialization
            batches_per_epoch: int
                The number of batches in each epoch. This number can be either smaller or larger than the real batch number.
                If not given (None), all batches will be used in each epoch.
            is_descending: bool
                Whether the batches are sorted in the descending order by the length (True) or in the ascending order (False).
                Note that if not given (None), the batches will remain the original order (no sorting is performed).
            data_len: str
                The absolute path of the data length file.
            selection_mode: str
                The selection mode that you would like to use for data selection.
                'random' mode randomly selects data samples from the built-in dataset.
                'order' mode selects data samples from the beginning of the built-in dataset.
                'rev_order' mode selects data samples from the end of the built-in dataset.
                If not given (None), no selection will be done.
            selection_num: float or int
                The number of data samples you would like to select from the built-in dataset.
                Positive float value (0.0 ~ 1.0) means the relative selection ratio of the built-in dataset size.
                Negative integer value (< 0) means the absolute selection number from the built-in dataset.
            shuffle: bool
                Whether the batches are shuffled.
            seed: int
                Random seed for the Dataloader.
                Given by the experiment pipeline (not the user responsibility).
            num_workers: int
                Number of workers for the Dataloader.
                Given by the experiment pipeline (not the user responsibility).
            pin_memory: bool
                Whether pin_memory trick is used in the Dataloader.
                Given by the experiment pipeline (not the user responsibility).
            distributed: bool
            **iter_conf:
                iterator configuration for its customized initialization
        """
        # initialize the built-in dataset of the iterator
        dataset_class = import_class('speechain.dataset.' + dataset_type)
        self.dataset = dataset_class(**dataset_conf)

        # initialize the general part of the iterator
        if batches_per_epoch is not None:
            assert batches_per_epoch > 0, f"batches_per_epoch must be a positive integer, but got {batches_per_epoch}."
        self.batches_per_epoch = batches_per_epoch
        self.is_descending = is_descending
        self.shuffle = shuffle
        self.seed = seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.distributed = distributed

        # remain the original order of the data indices if data_len not specified
        self.sorted_data = self.dataset.get_data_index()

        # initialize the data lengths if given
        if data_len is not None:
            data_len = [data_len] if isinstance(data_len, str) else data_len
            # remain the original order of the data indices if is_descending not specified
            self.data_len = [np.loadtxt(file, dtype=str, delimiter=" ") for file in data_len]
            self.data_len = [dict(zip(np_len[:, 0], np_len[:, 1].astype(int))) for np_len in self.data_len]
            self.data_len = {key: value for d_len in self.data_len for key, value in d_len.items()}

            # check the data index in data_len and self.dataset
            data_len_keys, dataset_keys = set(self.data_len.keys()), set(self.sorted_data)
            # delete the redundant key-value pairs in data_len
            for redundant_key in data_len_keys.difference(dataset_keys):
                self.data_len.pop(redundant_key)
            # delete the redundant key-value pairs in self.dataset
            for redundant_key in dataset_keys.difference(data_len_keys):
                self.dataset.remove_data_by_index(redundant_key)

        # select a portion of data samples to generate batches if specified
        if selection_mode is not None:
            mode_list = ['random', 'order', 'rev_order']
            assert selection_mode is not None and selection_mode in mode_list, \
                f"portion_mode should be one of {mode_list}, but got {selection_mode}."
            assert selection_num < 1 and selection_num != 0, \
                f"selection_num should be either a float number between 0 and 2 or a negative number. " \
                f"But got {selection_num:.2f}"
            if selection_num < 0:
                assert -selection_num < len(self.sorted_data), \
                    "The absolute value of selection_num cannot be larger than the total number of data samples. " \
                    f"You give {len(self.sorted_data)} samples but give selection_num={selection_num}."

            # portion the old self.sorted_data to get the new self.sorted_data
            self.sorted_data = self.data_selection(selection_num, selection_mode)

        # sorting the data indices by their lengths if specified
        if self.is_descending is not None and self.data_len is not None:
            # shrink the data_len by sorted_data if necessary
            self.data_len = {index: self.data_len[index] for index in self.sorted_data}
            self.data_len = dict(sorted(self.data_len.items(), key=lambda x: x[1], reverse=self.is_descending))

            # record the keys of the data samples for batch generation
            self.sorted_data = list(self.data_len.keys())

        # initialize the customized part of the iterator and get the batches of data indices
        self.batches = self.iter_init(**iter_conf)

        # clip the batches for distributed training
        if self.distributed:
            # set stride to the number of processes
            stride = torch.distributed.get_world_size()
            # set the start point to the global rank of the current process
            # make sure that the batches on GPU no.0 have the least data size (for more memory on no.0 GPU)
            start_point = stride - torch.distributed.get_rank() - 1 if self.is_descending \
                else torch.distributed.get_rank()
            self.batches = [batch[start_point::stride] for batch in self.batches]

            # delete all the empty elements in the multi-GPU distributed mode
            while [] in self.batches:
                self.batches.remove([])


    @abstractmethod
    def iter_init(self, **kwargs) -> List[List[Any]]:
        """
        This interface initializes the customized part of your iterator and returns the batches generated by the
        batching strategy of your iterator.

        This function should return the batches of sample indices as a list of list where each sub-list corresponds
        to a batch. Each element of a sub-list corresponds to a data sample.

        Here is an implementation example:
        >>> from speechain.iterator.abs import Iterator
        >>> class MyIterator(Iterator):
        >>>     def iter_init(self, my_conf):
        ...         # initialize of your configuration variables
        ...         self.my_conf = my_conf
        ...         batches = []
        ...         # generate batches by your batching strategy using self.my_conf and self.sorted_data
        ...         return batches

        For more detailed examples, please refer to the existing implementations in ./speechain/iterator/

        Returns:
            A list of batches generated by your batching strategy

        """
        raise NotImplementedError


    def data_selection(self, selection_num: float or int, selection_mode: str):
        """
        This function performs the data selection by the given selection_num and selection_mode

        Args:
            selection_num: float or int
                The number of samples selected from the built-in dataset.
            selection_mode: str
                The mode indicating how the samples are selected from the built-in dataset.

        Returns:
            A list of indices of the selected data samples

        """
        # 0 < selection_num < 1 means that we relatively select data samples by a percentage number
        # selection_num < 0 means that we absolutely select data samples by the given value
        selection_num = -selection_num if selection_num < 0 else len(self.sorted_data) * selection_num
        selection_num = int(selection_num)

        # turn into numpy.array for clipping operations
        sorted_data = np.array(self.sorted_data, dtype=str)
        # 'order' means we select the data samples from the beginning to the end
        if selection_mode == 'order':
            sorted_data = sorted_data[:selection_num]
        # 'rev_order' means we select the data samples from the end to the beginning
        elif selection_mode == 'rev_order':
            sorted_data = sorted_data[-selection_num:]
        # 'random' means we randomly select the data samples
        elif selection_mode == 'random':
            sorted_data = sorted_data[np.random.randint(0, len(sorted_data), selection_num)]
        else:
            raise ValueError

        # return the list of indices
        return sorted_data.tolist()


    def __len__(self):
        """

        Returns:
            The number of batches the model will receive during training. If batches_per_epoch is given, it will be
            returned; otherwise, the real number of batches will be returned.

        """
        if self.batches_per_epoch is not None:
            return self.batches_per_epoch
        else:
            return len(self.batches)


    def get_sample_indices(self):
        """

        Returns:

        """
        return self.batches


    def get_meta_info(self):
        """

        Returns:

        """
        return self.dataset.meta_info if hasattr(self.dataset, 'meta_info') else None


    def build_loader(self, epoch: int = 1):
        """
        cut a segment of batches off from self.batches and generate a DataLoader based on this segment of batches in
        each epoch.

        If batches_per_epoch is not given, all batches will be used to generate the Dataloader; If batches_per_epoch
        is given, 'batches_per_epoch' batches will be made from the existing batches according to the difference
        between batches_per_epoch and the number of existing batches.

        Simply speaking, 'batches_per_epoch' is used as a sliding window that cuts out a group of indices from
        self.batches in a non-overlapping way. 'batches_per_epoch' can be either larger or smaller than the real
        number of batches.

        Args:
            epoch: int
                The number of the current epoch. Used as the random seed to shuffle the batches.

        Returns:
            A DataLoader built on the chosen window in this epoch.

        """
        # no cut off when batches_per_epoch is not given
        if self.batches_per_epoch is None or len(self.batches) == self.batches_per_epoch:
            batches = self.batches

        # the amount of batches is larger than the given batches_per_epoch
        elif len(self.batches) > self.batches_per_epoch:
            # where to start cutting off the batches in this epoch
            cursor = (self.batches_per_epoch * (epoch - 1)) % len(self.batches)
            # the remaining part of existing batches is enough for this epoch
            if len(self.batches) - cursor >= self.batches_per_epoch:
                batches = self.batches[cursor: cursor + self.batches_per_epoch]
            # the remaining part is not enough, we need to go back to the beginning of existing batches
            else:
                batches = self.batches[cursor:] + \
                          self.batches[: self.batches_per_epoch - len(self.batches) + cursor]

        # the amount of batches is smaller than the given batches_per_epoch
        elif len(self.batches) < self.batches_per_epoch:
            # same way to get the starting point (cursor)
            cursor = (self.batches_per_epoch * (epoch - 1)) % len(self.batches)
            current_batch_size = 0
            batches = []
            # looping until we get enough batches
            while current_batch_size < self.batches_per_epoch:
                # the remaining part of existing batches is enough for us
                if current_batch_size + len(self.batches) - cursor >= self.batches_per_epoch:
                    last_remain = self.batches_per_epoch - current_batch_size
                    batches += self.batches[cursor: cursor + last_remain]
                    current_batch_size += last_remain
                # the remaining is not enough, we need to go to the beginning and do again
                else:
                    batches += self.batches[cursor:]
                    current_batch_size += len(self.batches) - cursor
                    cursor = 0

        if self.shuffle:
            np.random.RandomState(epoch + self.seed).shuffle(batches)

        return DataLoader(dataset=self.dataset,
                          batch_sampler=batches,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          collate_fn=self.dataset.collate_fn)


    def __repr__(self):
        batch_len = [len(batch) for batch in self.batches]
        return f"{self.__class__.__name__}(" \
               f"dataset={self.dataset.__class__.__name__}, " \
               f"num_workers={self.num_workers}, " \
               f"pin_memory={self.pin_memory}, " \
               f"is_descending={self.is_descending}, " \
               f"shuffle={self.shuffle}, " \
               f"total_batches={len(self.batches)}, " \
               f"batches_per_epoch={len(self)}, " \
               f"max_batch={max(batch_len)}, " \
               f"min_batch={min(batch_len)}, " \
               f"mean_batch={sum(batch_len) / len(batch_len):.1f})"