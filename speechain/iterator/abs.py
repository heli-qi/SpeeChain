"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import warnings

import numpy as np
import random
import torch
import os

from functools import partial
from torch.utils.data import DataLoader
from typing import Dict, List
from abc import ABC

from speechain.utilbox.import_util import import_class
from speechain.utilbox.data_loading_util import load_idx2data_file, read_idx2data_file_to_dict


def worker_init_fn(worker_id: int, base_seed: int, same_worker_seed: bool):
    """
    Set random seed for each worker in DataLoader to ensure the reproducibility.

    """
    seed = base_seed if same_worker_seed else base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class Iterator(ABC):
    """
    Iterator is the base class that takes charge of grouping data instances into batches for training or testing models.
    Each iterator has a built-in speechain.dataset.Dataset object as one of its member variables. Actually, an Iterator
    object cannot directly access the data instances in the built-in Dataset object but maintains a batching view of
    the indices of the data instances used for model training or testing.

    The initialization of the built-in Dataset object is done automatically during the initialization of the iterator.
    At the beginning of each epoch, the iterator generates a `torch.utils.data.DataLoader` object to fetch the batches
    of data instances from the disk.

    The iterators are divided into 3 groups: train, valid, and test. In each group, 2 or more iterator objects can be
    constructed so that there could be multiple data-label pairs in a single batch.

    """

    def __init__(self,
                 dataset_type: str,
                 dataset_conf: Dict,
                 batches_per_epoch: int = None,
                 data_len: str or List[str] = None,
                 group_info: Dict[str, str or List[str]] = None,
                 is_descending: bool or None = True,
                 shuffle: bool = True,
                 seed: int = 0,
                 ngpu: int = 1,
                 num_workers: int = 1,
                 same_worker_seed: bool = False,
                 pin_memory: bool = True,
                 distributed: bool = False,
                 **iter_conf):
        """
        The general initialization function of all the Iterator classes. Dataset initialization is automatically done
        here by the given dataset_type and dataset_conf.

        In this initialization function, each iterator subclass should override a hook function batches_generate_fn()
        to generate the batching view of data instances in the built-in Dataset object based on their own data batching
        strategy.

        Args:
            dataset_type: str
                Query string to pick up the target Dataset subclass in `speechain/dataset/`
            dataset_conf: Dict
                Dataset configuration for its automatic initialization
            batches_per_epoch: int = None
                The number of batches in each epoch. This number can be either smaller or larger than the real batch
                number. If not given (None), all batches will be used in each epoch.
            is_descending: bool = True
                Whether the batches are sorted in the descending order by the length (True) or in the ascending order
                (False). If this argument is given as None, no sorting is done for involved data instances.
            data_len: str or List[str] = None
                The absolute path of the data length file. Multiple length files can be given in a list, but they
                should contain non-overlapping data instances.
            group_info: Dict[str, str or List[str]] = None
                The dictionary of paths for the 'idx2data' files used for group-wise evaluation results visualization.
            shuffle: bool = True
                Whether the batches are shuffled at the beginning of each epoch.
            seed: int = 0
                Random seed for iterator initialization.
                It will be used to
                    1. shuffle batches before giving to the Dataloader of each epoch.
                    2. initialize the workers of the Dataloader for the reproducibility.
                This argument is automatically given by the experiment environment configuration.
            ngpu: int = 1
                The number of GPUs used to train or test models. The GPU number is used to ensure that each GPU process
                in the DDP mode has the batches with the same number of data instances.
                This argument is automatically given by the experiment environment configuration.
            num_workers: int = 1
                Number of workers for the Dataloader.
                This argument is automatically given by the experiment environment configuration.
            pin_memory: bool = False
                Whether pin_memory trick is used in the Dataloader.
                This argument is automatically given by the experiment environment configuration.
            distributed: bool = False
                Whether DDP is used to distribute the model.
                This argument is automatically given by the experiment environment configuration.
            **iter_conf:
                iterator configuration for customized batch generation
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
        self.same_worker_seed = same_worker_seed
        self.pin_memory = pin_memory
        self.distributed = distributed

        # --- 1. Loading the Data Length Information --- #
        if data_len is None:
            data_len = self.dataset.data_len

        # initialize the data lengths if given
        if data_len is not None:
            # remain the original order of the data indices if is_descending not specified
            self.data_len = load_idx2data_file(data_len, int) if not isinstance(data_len, Dict) else data_len

            # check the data index in data_len and self.dataset
            data_len_keys, dataset_keys = set(self.data_len.keys()), set(self.dataset.get_data_index())
            # delete the redundant key-value pairs in data_len
            redundant_keys = data_len_keys.difference(dataset_keys)
            if len(redundant_keys) > 0:
                warnings.warn(
                    f"There are {len(redundant_keys)} redundant keys that exist in data_len but not in main_data! "
                    f"If you are using data_selection in data_cfg, this may not be a problem.")
                for redundant_key in redundant_keys:
                    self.data_len.pop(redundant_key)
            # delete the redundant key-value pairs in self.dataset
            redundant_keys = dataset_keys.difference(data_len_keys)
            if len(redundant_keys) > 0:
                warnings.warn(
                    f"There are {len(redundant_keys)} redundant keys that exist in main_data but not in data_len! "
                    f"If you are using data_selection in data_cfg, this may not be a problem.")
                for redundant_key in dataset_keys.difference(data_len_keys):
                    self.dataset.remove_data_by_index(redundant_key)
        else:
            self.data_len = None

        # remain the original order of the data indices if data_len not specified
        self.sorted_data = self.dataset.get_data_index()

        # --- 2. Sorting the Data instances in order --- #
        # sorting the data indices by their lengths if specified
        if self.data_len is not None and self.is_descending is not None:
            # shrink the data_len by sorted_data if necessary
            if len(self.data_len) > len(self.sorted_data):
                self.data_len = {index: self.data_len[index] for index in self.sorted_data}
            self.data_len = dict(sorted(self.data_len.items(), key=lambda x: x[1], reverse=self.is_descending))

            # record the keys of the data instances for batch generation
            self.sorted_data = list(self.data_len.keys())

        # --- 3. Initialize the Customized Part (batching strategy) of the Iterator --- #
        # initialize the customized part of the iterator and get the batches of data indices
        self.batches = self.batches_generate_fn(self.sorted_data, self.data_len, **iter_conf)
        assert len(self.batches) > 0, \
            f"There is no batch generated in {self.__class__.__name__}! " \
            f"It's probably because there is a index mismatch between you given main_data in the dataset."

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

        # --- 4. Separate the Dataset into Multiple Non-overlapping Sections in the DDP Mode --- #
        # clip the batch view for distributed training
        if self.distributed:
            # set stride to the number of processes
            stride = torch.distributed.get_world_size()
            # set the start point to the global rank of the current process
            # make sure that the batches on GPU no.0 have the least data size (for more memory on no.0 GPU)
            start_point = stride - torch.distributed.get_rank() - 1 if self.is_descending or self.is_descending is None \
                else torch.distributed.get_rank()
            self.batches = [batch[start_point::stride] for batch in self.batches]

            # delete all the empty elements in the multi-GPU distributed mode
            while [] in self.batches:
                self.batches.remove([])

        # --- 5. Extract the Metadata Information from the Disk to the Memory --- #
        if group_info is not None:
            # --- 6.1. Loading the Group Information of Data Instances from the Disk to the Memory --- #
            assert isinstance(group_info, Dict), \
                f"group_info must be given in Dict, but got type(main_data)={type(group_info)}"
            self.group_info, self.data_index = read_idx2data_file_to_dict(group_info)

            # --- 6.2. Data Instance Index Checking between self.group_info and self.dataset.main_data --- #
            # check the data index in self.group_info and self.dataset
            group_info_keys, dataset_keys = set(self.data_index), set(self.dataset.get_data_index())
            # delete the redundant key-value pairs in self.group_info
            for redundant_key in group_info_keys.difference(dataset_keys):
                for group_name in self.group_info.keys():
                    self.group_info[group_name].pop(redundant_key)
            # delete the redundant key-value pairs in self.dataset
            for redundant_key in dataset_keys.difference(group_info_keys):
                self.dataset.remove_data_by_index(redundant_key)
        else:
            self.group_info, self.data_index = None, self.dataset.get_data_index()

    def batches_generate_fn(self, data_index: List[str], data_len: Dict[str, int], batch_size: int = None) \
            -> List[List[str]]:
        """
        This hook interface function generates the batching view based on a specific batch generation strategy.

        Your overridden function should return the batches of instance indices as a List[List[str]] where each sub-list
        corresponds to a batch of data instances. Each element in the sub-list is the index of a data instance.

        In this original hook implementation, all the data instances in the built-in Dataset object will be grouped
        into batches with exactly the same amount of instances. data_len is not used in this hook function but used for
        sorting all the instances in the general initialization function of the iterator. The sorted data instances make
        sure that the instances in a single batch have similar lengths.

        Args:
            data_index: List[str]
                The list of indices of all the data instances available to generate the batching view.
            data_len: Dict[str, int]
                The dictionary that indicates the data length of each available data instance in data_index.
            batch_size: int = None
                How many data instances does a batch should have. If not given, it will be the number of GPUs (ngpu) to
                ensure that the model validation or testing is done one data instance at each step on a single GPU
                process.

        Returns:
            A list of batches generated by your batching strategy. This List[List[str]] is called the batching view of
            the iterator object. Each batch in the returned list is a sub-list whose elements are the indices of data
            instances in the corresponding batch.

        """
        # batch_size is default to be the number of used GPUs to ensure that the model validation or testing is done one
        # data instance at each step on a single GPU process
        if batch_size is None:
            batch_size = self.ngpu
        # argument checking
        if not isinstance(batch_size, int):
            batch_size = int(batch_size)
        assert batch_size > 0, f"batch_size must be a positive integer, but got {batch_size}."

        # divide the data into individual batches with equal amount of instances
        batches = [data_index[i: i + batch_size]
                   for i in range(0, len(data_index) - batch_size + 1, batch_size)]
        # in case that there are several uncovered instances at the end of self.sorted_data
        remaining = len(data_index) % batch_size
        if remaining != 0:
            batches.append(data_index[-remaining:])

        return batches

    def __len__(self):
        """

        Returns:
            The real number of batches the iterator will load.
            If batches_per_epoch is given, it will be returned; otherwise, the total number of all the batches in the
            built-in Dataset object will be returned.

        """
        if self.batches_per_epoch is not None:
            return self.batches_per_epoch
        else:
            return len(self.batches)

    def get_batch_indices(self) -> List[List[str]]:
        """
        This function return the current batching view of the iterator object.

        Returns: List[List[str]]
            The batching view generated by the customized hook interface batches_generate_fn(). Each element of the
            returned batching view list is a sub-list of data indices where each index corresponds to a data instance
            in the built-in Dataset object.

        """
        return self.batches

    def get_group_info(self) -> Dict[str, Dict[str, str]] or None:
        """
        This function returns the metadata information of the built-in Dataset object.
        The returned metadata is mainly used for group-wise testing results visualization.

        Returns:
            If metadata information is not initialized in the built-in Dataset object, None will be returned.
            Otherwise, the meta_info member of the built-in Dataset object will be returned which is a dictionary.

        """
        return self.group_info

    def build_loader(self, epoch: int = 1, start_step: int = 0):
        """
        This function generate a torch.util.data.DataLoader to load the batches of data instances for the input epoch.

        If batches_per_epoch is not given, all the batches in self.batches will be used to generate the Dataloader;
        If batches_per_epoch is given, 'batches_per_epoch' batches will be generated by self.batches according to the
        difference between batches_per_epoch and the number of existing batches.

        batches_per_epoch can be either larger or smaller than the total number of batches.
        For a smaller batches_per_epoch, a part of self.batches will be used as the batch clip;
        For a larger batches_per_epoch, self.batches will be supplemented by a part of itself to form the batch clip.

        Args:
            epoch: int = 1
                The number of the current epoch. Used as part of the random seed to shuffle the batches.
            start_step: int = 0
                The start point for the dataloader of the current epoch. Used for resuming from a checkpoint during
                testing.

        Returns:
            A DataLoader built on the batch clip of the current epoch.
            If batches_per_epoch is not given, the batch clip is self.batches.

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
        else:
            raise RuntimeError

        if self.shuffle:
            np.random.RandomState(epoch + self.seed).shuffle(batches)

        if start_step > 0:
            batches = batches[start_step:]

        return DataLoader(dataset=self.dataset,
                          batch_sampler=batches,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          collate_fn=self.dataset.collate_fn,
                          worker_init_fn=partial(worker_init_fn, base_seed=epoch + self.seed,
                                                 same_worker_seed=self.same_worker_seed))

    def __repr__(self):
        batch_len = [len(batch) for batch in self.batches]
        return f"{self.__class__.__name__}(" \
               f"dataset=({str(self.dataset)}), " \
               f"seed={self.seed}, " \
               f"ngpu={self.ngpu}, " \
               f"num_workers={self.num_workers}, " \
               f"same_worker_seed={self.same_worker_seed}, " \
               f"pin_memory={self.pin_memory}, " \
               f"is_descending={self.is_descending}, " \
               f"shuffle={self.shuffle}, " \
               f"total_batches={len(self.batches)}, " \
               f"batches_per_epoch={len(self)}, " \
               f"max_batch={max(batch_len)}, " \
               f"min_batch={min(batch_len)}, " \
               f"mean_batch={sum(batch_len) / len(batch_len):.1f})"
