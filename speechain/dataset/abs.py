"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""

import torch
import numpy as np

from typing import List, Dict, Any, Union, Optional
from abc import ABC

from speechain.utilbox.data_loading_util import load_idx2data_file, read_idx2data_file_to_dict


class Dataset(torch.utils.data.Dataset, ABC):
    """

    Base class for reading and packaging data instances from disk into memory for model training or testing.

    The Dataset receives indices of selected data instances from a Dataloader object, created by a high-level Iterator.
    Post-processing steps may need to be executed in the Model object later as the output batches might not be fully processed.

    """

    def __init__(self,
                 main_data: Dict[str, Union[str, List[str]]],
                 data_selection: Optional[List[Union[List[str], str]]] = None,
                 **dataset_conf):
        """
        This initialization function reads the main body of the data instances into the memory. The main body is used to
        extract individual data instances from the disk to form a batch during model training or testing.

        The hook dataset_init_fn() is executed here after reading the main body of data instances.

        Args:
            main_data (Dict[str, Union[str, List[str]]]):
                Dictionary containing data instances used in the Dataset object. Each key-value pair consists of a data
                variable name and an absolute path to the corresponding 'idx2data' file. The value can be a single path string
                or a list of multiple path strings.

            data_selection (Optional[List[Union[List[str], str]]]):
                Strategies for data selection to limit used data instances during iterator initialization. Multiple strategies
                can be specified in a list. Each data selection strategy should be either a bi-list (non-meta strategy)
                or tri-list (meta strategy). Refer to the function docstring of data_selection() for more details on
                the selection strategies.

            **dataset_conf: Additional configuration arguments for custom Dataset initialization.

            data_selection: List[List[str] or str] = None
                The strategies for data selection during the iterator initialization to shrink the used data instances.
                Multiple strategies can be specified in a list. Each data selection strategy must be either a bi-list
                (non-meta strategy) or tri-list (meta strategy).
                1. non-meta strategy:
                    The selection strategies that don't involve metadata. These strategies should be given as a bi-list,
                    i.e., ['selection mode', 'selection number']. 'selection mode' indicates the way to select data
                    instances while 'selection number' indicates how many data instances to be selected.
                    Currently, available non-meta selection modes include:
                        1. 'order': Select the given number of data instances from the beginning.
                        2. 'rev_order': Select the given number of data instances from the end.
                        3. 'random': Randomly select the given number of data instances.
                            Note: You should keep the same random seeds for all the GPU processes in the DDP mode to
                            ensure that the selected data instances are the same in each process. In this case, please
                            set the 'same_proc_seed' argument to True in your configuration given to speechain.runner
                2. meta strategy:
                    The selection strategies that involves metadata. These strategies should be given as a tri-list,
                    i.e., ['selection mode', 'selection threshold', 'metadata path']. 'selection mode' indicates the
                    way to select data instances, 'selection threshold' indicates the metadata threshold to select data
                    instances, and 'metadata path' indicates where is the metadata used for selection.
                    Currently, available meta selection modes include:
                        1. 'min': Select the data instances whose metadata is smaller than the threshold.
                        2. 'max': Select the data instances whose metadata is larger than the threshold.
                        3. 'middle': Remove the data instances whose metadata is the largest and smallest.
        """
        super(Dataset, self).__init__()

        # Validate main_data
        if not isinstance(main_data, Dict):
            raise TypeError(f"Expected main_data to be a Dict, but got {type(main_data)}")

        # Load main body of data instances
        self.main_data, self.data_index = read_idx2data_file_to_dict(main_data)

        # Apply data selection if specified
        if data_selection is not None:
            # Ensure data_selection is a list of lists
            if sum([isinstance(i, List) for i in data_selection]) != len(data_selection):
                data_selection = [data_selection]

            # Iterate through each selection strategy
            for i in data_selection:
                # Non-meta selection
                if len(i) == 2:
                    selection_mode, selection_num, meta_info = i[0], i[1], None
                    if selection_mode not in ['random', 'order', 'rev_order']:
                        raise ValueError(
                            f"For non-meta selection, mode must be 'random', 'order', or 'rev_order'. Got {selection_mode}")
                # Meta-required selection
                elif len(i) == 3:
                    selection_mode, selection_num, meta_info = i[0], i[1], i[2]
                    if selection_mode not in ['min', 'max', 'middle', 'group']:
                        raise ValueError(
                            f"For meta selection, mode must be 'min', 'max', 'middle', or 'group'. Got {selection_mode}")
                else:
                    raise ValueError("Each element of data_selection should be either a 2-element or 3-element list")

                # Validate selection_num
                if isinstance(selection_num, str):
                    # Non-numerical contents are turned into a list for identification
                    if not selection_num.isdigit() and not selection_num.replace('.', '').isdigit():
                        assert selection_mode == 'group'
                        selection_num = [selection_num]

                valid_selection_num = (
                        (isinstance(selection_num, float) and 0 < selection_num < 1) or
                        (isinstance(selection_num, int) and -len(self.data_index) < selection_num < 0) or
                        isinstance(selection_num, (str, List))
                )
                if not valid_selection_num:
                    raise ValueError(
                        "Data selection number should be a float number between 0 and 1, a negative integer, "
                        "a string, or a list of strings")

                if (isinstance(selection_num, (int, float)) and selection_num < 0) and \
                        (-selection_num >= len(self.data_index)):
                        raise ValueError("Data selection amount cannot be larger than total number of data instances")

                # Apply the data selection
                self.data_index = self.data_selection(self.data_index, selection_mode, selection_num, meta_info)

        # Custom initialization for subclasses
        self.data_len = self.data_len_register_fn(self.main_data)
        self.dataset_init_fn(**dataset_conf)

    @staticmethod
    def data_len_register_fn(main_data: Dict[str, Dict[str, str]]) -> Union[Dict[str, Union[int, float]], None]:
        """
            Static hook function that registers default information about the length of each data instance.

            By default, this function does nothing. If you need to decide the data length on-the-fly, override this function
            with your own implementation.

            Args:
                main_data (Dict[str, Dict[str, str]]): Dictionary of main data from which length information is derived.

            Returns:
                Dict[str, Union[int, float]] or None: Dictionary mapping data instances to their lengths, or None if not implemented.
        """
        return None

    def dataset_init_fn(self, **dataset_conf):
        """
            Hook function that initializes the custom parts of dataset implementations.

            By default, this function does nothing. If your Dataset subclass has custom parts, override this function
            with your own implementation.

            Args:
                **dataset_conf: Arguments for the custom initialization of the Dataset subclass.
        """
        pass

    @staticmethod
    def data_selection(data_index: List[str], selection_mode: str, selection_num: Union[float, int, str],
                       meta_info: Union[List[str], str, None] = None) -> List:
        """
        Selects data instances based on the provided selection strategy.

        Returns a new list of selected data instances.

        Args:
            data_index (List[str]):
                List of data instance indices prior to data selection.
            selection_num (Union[float, int, str]):
                Indicates how many data instances to select, varying with its data type.
                float: Represents the relative number of data instances to select (between 0 and 1).
                int: Represents the absolute number of data instances to select. If negative, its absolute value is taken.
                str: Represents the metadata threshold for data selection. Only 'min' and 'max' modes support this.
                    You can use the !-suffixed representer `!str` to convert a float or integer number to a string in your .yaml file.
            selection_mode: str
                Defines the selection strategy:
                1. non-meta strategy:
                   Rule-based selection strategies that do not involve metadata. Includes:
                     1. 'order': Selects the given number of data instances from the beginning.
                     2. 'rev_order': Selects the given number of data instances from the end.
                     3. 'random': Selects the given number of data instances randomly.
                         Note: You should keep the same random seeds for all the GPU processes in the DDP mode to ensure
                         that the selected data instances are the same in each process. In this case, please set the
                         'same_proc_seed' argument to True in your configuration given to speechain.runner.py.
                2. meta strategy:
                   Selection strategies that involve metadata. Includes:
                     1. 'min': Selects the data instances whose metadata is smaller than the threshold.
                     2. 'max': Selects the data instances whose metadata is larger than the threshold.
                     3. 'middle': Removes the data instances whose metadata is the largest and smallest.
            meta_info (Union[List[str], str, None], optional):
                Path to metadata information used for selection. Defaults to None.

        Returns: List[str]
            List[str]: A list of selected data instance indices.

        """
        # Convert data_index to numpy.array for easier manipulation
        sorted_data = np.array(data_index, dtype=str)

        # Non-metadata selection strategies
        if meta_info is None:
            assert isinstance(selection_num, (int, float))
            # Determine absolute or relative number of instances to select
            selection_num = int(-selection_num if selection_num < 0 else len(sorted_data) * selection_num)
            # Selection from the beginning
            if selection_mode == 'order':
                sorted_data = sorted_data[:selection_num]
            # Selection from the end
            elif selection_mode == 'rev_order':
                sorted_data = sorted_data[-selection_num:]
            # Random selection
            elif selection_mode == 'random':
                sorted_data = sorted_data[np.random.randint(0, len(sorted_data), selection_num)]

        # Metadata-based selection strategies
        else:
            # Load metadata information
            meta_info = load_idx2data_file(meta_info)
            meta_info = np.array([[key, value] for key, value in meta_info.items()])
            # Initialize sorted indices and metadata values
            try:
                meta_sorted_data = meta_info[:, 0][np.argsort(meta_info[:, 1].astype(float))]
                meta_sorted_value = np.sort(meta_info[:, 1].astype(float))
            # Catch conversion errors
            except ValueError:
                meta_sorted_data = meta_info[:, 0]
                meta_sorted_value = meta_info[:, 1]

            # Only retain data instances present in both datasets
            retain_flags = np.in1d(meta_sorted_data, sorted_data)
            meta_sorted_data, meta_sorted_value = meta_sorted_data[retain_flags], meta_sorted_value[retain_flags]

            # Process selection based on provided selection_num
            if isinstance(selection_num, (int, float)):
                # 0 < selection_num < 1 means that we relatively select data instances by a percentage number
                # selection_num < 0 means that we absolutely select data instances by the given value
                selection_num = int(-selection_num if selection_num < 0 else len(meta_sorted_data) * selection_num)
                # 'min' means the instances with the minimal meta values will be selected
                if selection_mode == 'min':
                    removed_sorted_data = meta_sorted_data[selection_num:]
                # 'max' means the instances with the maximal meta values will be selected
                elif selection_mode == 'max':
                    removed_sorted_data = meta_sorted_data[:-selection_num]
                # 'middle' means the instances with the minimal and maximal meta values will be excluded
                elif selection_mode == 'middle':
                    removed_sorted_data = np.concatenate(
                        (meta_sorted_data[:int((meta_sorted_data.shape[0] - selection_num) / 2)],
                         meta_sorted_data[-int((meta_sorted_data.shape[0] - selection_num) / 2):]), axis=0)
                else:
                    raise RuntimeError(f"If selection_num is given in a integer or float number ({selection_num}), "
                                       f"selection_mode must be one of ['min', 'max', 'middle']. "
                                       f"But got {selection_mode}.")

            # select the data instances by a given threshold
            elif isinstance(selection_num, str):
                selection_num = float(selection_num)
                # 'min' means the instances whose metadata is lower than the given threshold will be selected
                if selection_mode == 'min':
                    removed_sorted_data = meta_sorted_data[meta_sorted_value > selection_num]
                # 'max' means the instances whose metadata is larger than the given threshold will be selected
                elif selection_mode == 'max':
                    removed_sorted_data = meta_sorted_data[meta_sorted_value < selection_num]
                # 'middle' is not supported for the threshold selection
                else:
                    raise RuntimeError(f"If selection_num is given in a string ({selection_num}), selection_mode must "
                                       f"be one of ['min', 'max']. But got {selection_mode}.")

            # other strings mean the target groups of data instances
            elif isinstance(selection_num, List):
                removed_sorted_data = meta_sorted_data[
                    [True if value not in selection_num else False for value in meta_sorted_value]]

            else:
                raise ValueError("Invalid type for selection_num.")

            # Remove undesired instances from sorted_data
            sorted_data = np.setdiff1d(sorted_data, removed_sorted_data)

        # Return selected indices as list
        return sorted_data.tolist()

    def get_data_index(self) -> List[str]:
        """
        This function is designed to make users know the data indices of this Dataset object without accessing its
        members for lower coupling.

        Returns: List[str]
            The list of the indices of all data instances in this dataset.

        """
        return self.data_index

    def remove_data_by_index(self, index: str):
        """
        This function removes the corresponding data instance from this Dataset object by the given index. It's mainly
        used for solving the index mismatch of data instances with the high-level Iterator object.

        """
        # remove the data instances with the given index from self.main_data
        for data_type in self.main_data.keys():
            if index in self.main_data[data_type].keys():
                self.main_data[data_type].pop(index)

    def __getitem__(self, index: str) -> Dict[str, Any]:
        """
        This function is the implementation of the one in the parent class `torch.utils.data.Dataset`.  This function
        is activated by the _Dataloader_ object one data instance a time. In each time, this function receives an index
        and returns the selected data instance.

        The hook `proc_main_data_fn()` is executed here after extracting the main body of the selected data instance.

        Args:
            index: str
                The index of the selected data instance given by the Dataloader object.

        Returns: Dict[str, Any]
            A dictionary containing a data instance.
        """
        # pre-extract the data instances from self.main_data dictionary by the given index
        outputs = {key: value[index] for key, value in self.main_data.items()}

        # process the main body of data instances by the hook interface implementation
        outputs = self.extract_main_data_fn(outputs)
        return outputs

    def extract_main_data_fn(self, main_data: Dict) -> Dict[str, Any] or None:
        """
        This hook function extracts the selected data instance from the disk to the memory. If you want to implement
        your own data instance extraction, please override this hook function and give your logic here.

        Args:
            main_data: Dict[str, str]
                The dictionary containing necessary information for extracting the data instance from the disk to the
                memory. For example, the audio file path for the waveform data and the feature file path for the speaker
                embedding.

        Returns: Dict[str, Any]
            The dictionary containing the extracted data instance.

        """
        return main_data

    def collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """
        This hook function decides how to preprocess a list of extracted data instance dictionary before giving them to
        the model. This hook function is used as the value of the argument collate_fn for initializing Dataloader object
        at the beginning of each epoch.

        If you have your own batch collating strategy, we don't recommend you to override this hook but another hook
        named `collate_main_data_fn()`.

        This function should return the processed batch data in the form of a dictionary.

        Args:
            batch: List[Dict[str, Any]]
                The tuple of data instance dictionaries extracted by `extract_main_data_fn()`.

        Returns: Dict[str, Any]
            The batch dictionary that will be passed to the model.

        """
        # preprocess List[Dict[str, Any]] to Dict[str, List[Any]]
        outputs = dict()
        while len(batch) != 0:
            ele_dict = batch[0]
            if ele_dict is not None:
                for key in ele_dict.keys():
                    if key not in outputs.keys():
                        outputs[key] = []
                    outputs[key].append(ele_dict[key])
            # remove the redundant data for memory safety
            batch.remove(ele_dict)

        # postprocess Dict[str, List[Any]] by the hook implementation
        return self.collate_main_data_fn(outputs)

    def collate_main_data_fn(self, batch_dict: Dict[str, List]) -> Dict[str, torch.Tensor or List]:
        """
        This hook function decides how to preprocess a dictionary of the extracted batch of data instances before giving
        them to the model. The original hook in the base class packages all the elements other than strings of the batch
        into a `torch.Tensor`. Therefore, the `torch.Tensor` elements must have the same shape. The string elements will
        remain a list.

        If you have your own batch collating strategy, please override this hook function and give your logic here.

        Args:
            batch_dict: Dict[str, List]
                The reshaped dictionary of the extracted batch. In each key-value item, the key is the name of the data
                variable that will be passed to the model and the value is the list of unorganized data from all the
                elements in the batch.

        Returns: Dict[str, torch.Tensor or List]
            The dictionary containing the collated batch of data instances.

        """
        # extract the main body of data instances by the hook interface implementation
        for key in batch_dict.keys():
            # List[torch.Tensor] -> torch.Tensor
            if isinstance(batch_dict[key][0], torch.Tensor):
                batch_dict[key] = torch.stack([ele for ele in batch_dict[key]])
            # List[numpy.ndarry] -> List[torch.Tensor] -> torch.Tensor
            elif isinstance(batch_dict[key][0], np.ndarray):
                batch_dict[key] = torch.stack([torch.tensor(ele) for ele in batch_dict[key]])
            # List[int] -> torch.LongTensor
            elif isinstance(batch_dict[key][0], int):
                batch_dict[key] = torch.LongTensor(batch_dict[key])
            # List[float] -> torch.FloatTensor
            elif isinstance(batch_dict[key][0], float):
                batch_dict[key] = torch.FloatTensor(batch_dict[key])
            # List[str] remains List[str]
            elif not isinstance(batch_dict[key][0], str):
                raise RuntimeError

        return batch_dict

    def __repr__(self):
        return self.__class__.__name__
