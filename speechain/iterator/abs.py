"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
from functools import partial

import numpy as np
import random
import torch

from torch.utils.data import DataLoader
from typing import Dict, List, Any
from abc import ABC, abstractmethod

from speechain.utilbox.import_util import import_class


def worker_init_fn(worker_id, base_seed: int = 0):
    """
    Set random seed for each worker in DataLoader.
    Borrowed from https://github.com/espnet/espnet/blob/master/espnet2/iterators/sequence_iter_factory.py#L13

    """
    seed = base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)


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
                 data_len: str or List[str] = None,
                 data_selection: List = None,
                 is_descending: bool = True,
                 shuffle: bool = True,
                 seed: int = 0,
                 ngpu: int = 1,
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
            assert batches_per_epoch > 0, f"batches_per_epoch must be a positive number, but got {batches_per_epoch}."
        self.batches_per_epoch = int(batches_per_epoch) if batches_per_epoch is not None else batches_per_epoch
        self.is_descending = is_descending
        self.shuffle = shuffle
        self.seed = seed
        self.ngpu = ngpu
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.distributed = distributed
        # remain the original order of the data indices if data_len not specified
        self.sorted_data = self.dataset.get_data_index()

        # --- 1. Loading the Data Length Information --- #
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

        # --- 2. Performing the Data Selection --- #
        # select a portion of data samples to generate batches if specified
        if data_selection is not None:
            nometa_modes, meta_modes = ['random', 'order', 'rev_order'], ['min', 'max', 'middle']
            if not isinstance(data_selection[0], List):
                data_selection = [data_selection]

            # loop each data selection strategy in order of the input list
            for i in data_selection:
                # non-meta selection
                if len(i) == 2:
                    selection_mode, selection_num, meta_info = i[0], i[1], None
                    assert selection_mode in nometa_modes
                # meta-required selection
                elif len(i) == 3:
                    selection_mode, selection_num, meta_info = i[0], i[1], i[2]
                    assert selection_mode in meta_modes
                else:
                    raise RuntimeError

                assert (isinstance(selection_num, float) and 0 < selection_num < 1) or \
                       (isinstance(selection_num, int) and selection_num < 0) or \
                       isinstance(selection_num, str), \
                    f"Data selection number should be either a float number between 0 and 1, a negative integer, " \
                    f"or a number in the string format. But got selection_num={selection_num}"

                if not isinstance(selection_num, str) and selection_num < 0:
                    assert -selection_num < len(self.sorted_data), \
                        "The data selection amount cannot be larger than the total number of data samples. " \
                        f"You have {len(self.sorted_data)} samples but give selection_num={-selection_num}."

                # portion the old self.sorted_data to get the new self.sorted_data
                self.sorted_data = self.data_selection(selection_mode, selection_num, meta_info)

        # --- 3. Sorting the Remaining Data Samples in order --- #
        # sorting the data indices by their lengths if specified
        if hasattr(self, 'data_len'):
            # shrink the data_len by sorted_data if necessary
            self.data_len = {index: self.data_len[index] for index in self.sorted_data}
            self.data_len = dict(sorted(self.data_len.items(), key=lambda x: x[1], reverse=self.is_descending))

            # record the keys of the data samples for batch generation
            self.sorted_data = list(self.data_len.keys())

        # --- 4. Initialize the Customized Part (batching strategy) of the Iterator --- #
        # initialize the customized part of the iterator and get the batches of data indices
        self.batches = self.iter_init(**iter_conf)
        # make sure that each batch has self.ngpu data indices for even workload on each GPU
        if self.ngpu > 1:
            _tmp_indices = None
            for i in range(len(self.batches)):
                # attach the redundant ones from the last batch to the beginning of the current batch
                if _tmp_indices is not None:
                    self.batches[i] = _tmp_indices + self.batches[i]
                    _tmp_indices = None
                # check whether there are some redundant ones in the current batch
                _remain = len(self.batches[i]) % self.ngpu
                if _remain != 0:
                    _tmp_indices = self.batches[i][-_remain:]
                    self.batches[i] = self.batches[i][:-_remain]
            # check whether there are extra ones not included
            if _tmp_indices is not None:
                self.batches.append(_tmp_indices)

        # --- 5. Separate the Dataset into Multiple Non-overlapping Sections in the DDP Mode --- #
        # clip the batch view for distributed training
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

    def data_selection(self, selection_mode: str, selection_num: float or int or str, meta_info: str = None):
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
        # turn into numpy.array for clipping operations
        sorted_data = np.array(self.sorted_data, dtype=str)

        # for non-meta selection strategies
        if meta_info is None:
            assert isinstance(selection_num, (int, float))
            # 0 < selection_num < 1 means that we relatively select data samples by a percentage number
            # selection_num < 0 means that we absolutely select data samples by the given value
            selection_num = int(-selection_num if selection_num < 0 else len(sorted_data) * selection_num)
            # 'order' means we select the data samples from the beginning to the end
            if selection_mode == 'order':
                sorted_data = sorted_data[:selection_num]
            # 'rev_order' means we select the data samples from the end to the beginning
            elif selection_mode == 'rev_order':
                sorted_data = sorted_data[-selection_num:]
            # 'random' means we randomly select the data samples
            elif selection_mode == 'random':
                sorted_data = sorted_data[np.random.randint(0, len(sorted_data), selection_num)]

        # for meta-required selection strategies
        else:
            # read the metadata information for data selection
            meta_info = np.loadtxt(meta_info, dtype=str, delimiter=" ")
            # initialize the sorted indices and metadata values of the data samples
            meta_sorted_data = meta_info[:, 0][np.argsort(meta_info[:, 1].astype(float))]
            meta_sorted_value = np.sort(meta_info[:, 1].astype(float))
            # retain only the intersection of data samples in case that there is a index mismatch
            intsec_indices = np.in1d(meta_sorted_data, sorted_data)
            meta_sorted_data, meta_sorted_value = meta_sorted_data[intsec_indices], meta_sorted_value[intsec_indices]

            # select a certain amount of data samples
            if isinstance(selection_num, (int, float)):
                # 0 < selection_num < 1 means that we relatively select data samples by a percentage number
                # selection_num < 0 means that we absolutely select data samples by the given value
                selection_num = int(-selection_num if selection_num < 0 else len(meta_sorted_data) * selection_num)
                # 'min' means the samples with the minimal meta values will be selected
                if selection_mode == 'min':
                    removed_sorted_data = meta_sorted_data[selection_num:]
                # 'max' means the samples with the maximal meta values will be selected
                elif selection_mode == 'max':
                    removed_sorted_data = meta_sorted_data[:-selection_num]
                # 'middle' means the samples with the minimal and maximal meta values will be excluded
                elif selection_mode == 'middle':
                    removed_sorted_data = meta_sorted_data[:int((meta_sorted_data.shape[0] - selection_num) / 2)] + \
                                          meta_sorted_data[-int((meta_sorted_data.shape[0] - selection_num) / 2):]
                else:
                    raise ValueError

            # select the data samples by a certain threshold
            elif isinstance(selection_num, str):
                selection_num = float(selection_num)
                # 'min' means the samples whose metadata is lower than the given threshold will be selected
                if selection_mode == 'min':
                    removed_sorted_data = meta_sorted_data[meta_sorted_value > selection_num]
                # 'max' means the samples whose metadata is larger than the given threshold will be selected
                elif selection_mode == 'max':
                    removed_sorted_data = meta_sorted_data[meta_sorted_value < selection_num]
                # 'middle' is not supported for the threshold selection
                else:
                    raise ValueError

            else:
                raise TypeError

            # remove the undesired samples from the accessible sample indices
            sorted_data = np.setdiff1d(sorted_data, removed_sorted_data)

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

    def build_loader(self, epoch: int = 1, start_step: int = 0):
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
            start_step: int
                The start point for the dataloader of the current epoch. Used for resuming from a checkpoint during testing.

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

        if start_step > 0:
            batches = batches[start_step:]

        return DataLoader(dataset=self.dataset,
                          batch_sampler=batches,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          collate_fn=self.dataset.collate_fn,
                          worker_init_fn=partial(worker_init_fn, base_seed=epoch + self.seed))

    def __repr__(self):
        batch_len = [len(batch) for batch in self.batches]
        return f"{self.__class__.__name__}(" \
               f"dataset={self.dataset.__class__.__name__}, " \
               f"seed={self.seed}, " \
               f"ngpu={self.ngpu}, " \
               f"num_workers={self.num_workers}, " \
               f"pin_memory={self.pin_memory}, " \
               f"is_descending={self.is_descending}, " \
               f"shuffle={self.shuffle}, " \
               f"total_batches={len(self.batches)}, " \
               f"batches_per_epoch={len(self)}, " \
               f"max_batch={max(batch_len)}, " \
               f"min_batch={min(batch_len)}, " \
               f"mean_batch={sum(batch_len) / len(batch_len):.1f})"
