"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""

import torch
import numpy as np

from typing import List, Dict, Any
from abc import ABC

from speechain.utilbox.data_loading_util import load_idx2data_file, read_idx2data_file_to_dict, parse_path_args


class Dataset(torch.utils.data.Dataset, ABC):
    """
    Dataset is the base class that takes charge of reading the data instances from the disk into the memory and
    packaging them into a batch for model training or testing.

    This object receives indices of the selected data instances from the Dataloader object created by the high-level
    Iterator object. The output batches of packaged data instances may not be well-processed. Some post-processing steps
    need to be done in the Model object later.

    """

    def __init__(self,
                 main_data: Dict[str, str or List[str]],
                 data_selection: List[List[str] or str] = None,
                 **dataset_conf):
        """
        This initialization function reads the main body of the data instances into its memory. The main body is used to
        extract individual data instances from the disk to form a batch during model training or testing.

        The hook dataset_init_fn() is executed here after reading the main body of data instances.

        Args:
            main_data: Dict[str, str or List[str]]
                The main body dictionary of the data instances used in this Dataset object. In each key-value item, the
                key is the name of the data variable and the value is the absolute path of the target 'idx2data' files.
                The value can be given as a single path string or a list of multiple path strings.
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
            **dataset_conf:
                The configuration arguments for customized Dataset initialization.
        """
        # For declaring some member variables in the initialization function, it's mandatory to write the code
        # 'super(Dataset, self).__init__()' here.
        super(Dataset, self).__init__()

        # --- 1. Loading the Main Body of Data Instances from the Disk to the Memory --- #
        assert isinstance(main_data, Dict), \
            f"main_data must be given in Dict, but got type(main_data)={type(main_data)}"
        self.main_data, self.data_index = read_idx2data_file_to_dict(main_data)

        # --- 2. Executing the Data Selection --- #
        # select a part of data instances to generate batches if specified
        if data_selection is not None:
            if sum([isinstance(i, List) for i in data_selection]) != len(data_selection):
                data_selection = [data_selection]

            # loop each data selection strategy in order of the input list
            for i in data_selection:
                # non-meta selection
                if len(i) == 2:
                    selection_mode, selection_num, meta_info = i[0], i[1], None
                    assert selection_mode in ['random', 'order', 'rev_order']
                # meta-required selection
                elif len(i) == 3:
                    selection_mode, selection_num, meta_info = i[0], i[1], i[2]
                    assert selection_mode in ['min', 'max', 'middle']
                else:
                    raise ValueError("The elements of data_selection should be either a bi-list or tri-list, "
                                     f"but got {i}!")

                assert (isinstance(selection_num, float) and 0 < selection_num < 1) or \
                       (isinstance(selection_num, int) and -len(self.data_index) < selection_num < 0) or \
                       isinstance(selection_num, str), \
                    f"Data selection number should be either a float number between 0 and 1, a negative integer, " \
                    f"or a number in the string format. But got selection_num={selection_num}"

                if not isinstance(selection_num, str) and selection_num < 0:
                    assert -selection_num < len(self.data_index), \
                        "The data selection amount cannot be larger than the total number of data instances. " \
                        f"You have {len(self.data_index)} instances but give selection_num={-selection_num}."

                # portion the old self.sorted_data to get the new self.sorted_data
                self.data_index = self.data_selection(self.data_index, selection_mode, selection_num, meta_info)

        # --- 3. Customized Initialization for Individual Dataset Subclasses --- #
        self.dataset_init_fn(**dataset_conf)

    def dataset_init_fn(self, **dataset_conf):
        """
        This hook function initializes the customized part of your dataset implementations.
        This hook is not mandatory to be overridden and the original one in the base class does nothing.
        If your Dataset subclass has some customized part, please override this hook function and put your logic here.

        Args:
            **dataset_conf:
                The arguments used for the customized initialization of your Dataset subclass.

        """
        pass

    @staticmethod
    def data_selection(data_index: List[str], selection_mode: str, selection_num: float or int or str,
                       meta_info: List[str] or str = None) -> List:
        """
        This function performs the data selection by the input selection strategy arguments.
        A new batching view of the selected data instances will be returned.

        Args:
            data_index: List[str]
                The list of data instance indices before data selection.
            selection_num: float or int or str
                This argument has the different usage with different data types.
                1. float type:
                    Float value represents the relative number of data instances to be selected.
                    If selection_num is given as a float number, it must be between 0 and 1.
                2. int type:
                    Integer value represents the absolute number of data instances to be selected. If selection_num is
                    given as an interger number, it must be negative (its absolute value will be taken).
                3. str type:
                    String value represents the metadata threshold used to select the data instances.
                    Only 'min' and 'max' modes support string _selection_num_.
                    Note: You can use the !-suffixed representer `!str` to convert a float or integer number to a
                    string in your .yaml file.
            selection_mode: str
                The mode indicating how the data instances are selected.
                Selection modes are grouped by different types of data selection strategies.
                1. non-meta strategy:
                  The rule-based selection strategies that don't involve metadata.
                  Currently, available non-meta selection modes include:
                     1. 'order': Select the given number of data instances from the beginning.
                     2. 'rev_order': Select the given number of data instances from the end.
                     3. 'random': Randomly select the given number of data instances.
                     Note: You should keep the same
                     random seeds for all the GPU processes in the DDP mode to ensure that the selected data instances
                     are the same in each process. In this case, please set the 'same_proc_seed' argument to True in
                     your configuration given to speechain.runner.py.
                2. meta strategy:
                  The selection strategies that involves metadata.
                  Currently, available meta selection modes include:
                   1. 'min': Select the data instances whose metadata is smaller than the threshold.
                   2. 'max': Select the data instances whose metadata is larger than the threshold.
                   3. 'middle': Remove the data instances whose metadata is the largest and smallest.
            meta_info: str = None
                The path where the metadata information used for selection is placed.

        Returns: List[str]
            A list of indices of the selected data instances

        """
        # turn into numpy.array for clipping operations
        sorted_data = np.array(data_index, dtype=str)

        # for non-meta selection strategies
        if meta_info is None:
            assert isinstance(selection_num, (int, float))
            # 0 < selection_num < 1 means that we relatively select data instances by a percentage number
            # selection_num < 0 means that we absolutely select data instances by the given value
            selection_num = int(-selection_num if selection_num < 0 else len(sorted_data) * selection_num)
            # 'order' means we select the data instances from the beginning to the end
            if selection_mode == 'order':
                sorted_data = sorted_data[:selection_num]
            # 'rev_order' means we select the data instances from the end to the beginning
            elif selection_mode == 'rev_order':
                sorted_data = sorted_data[-selection_num:]
            # 'random' means we randomly select the data instances
            elif selection_mode == 'random':
                sorted_data = sorted_data[np.random.randint(0, len(sorted_data), selection_num)]

        # for meta-required selection strategies
        else:
            # read the metadata information for data selection
            meta_info = load_idx2data_file(meta_info)
            meta_info = np.array([[key, value] for key, value in meta_info.items()])
            # initialize the sorted indices and metadata values of the data instances
            meta_sorted_data = meta_info[:, 0][np.argsort(meta_info[:, 1].astype(float))]
            meta_sorted_value = np.sort(meta_info[:, 1].astype(float))
            # retain only the intersection of data instances in case that there is an index mismatch
            retain_flags = np.in1d(meta_sorted_data, sorted_data)
            meta_sorted_data, meta_sorted_value = meta_sorted_data[retain_flags], meta_sorted_value[retain_flags]

            # select a certain amount of data instances
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
                    raise ValueError

            # select the data instances by a certain threshold
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
                    raise ValueError

            else:
                raise TypeError

            # remove the undesired instances from the accessible instance indices
            sorted_data = np.setdiff1d(sorted_data, removed_sorted_data)

        # return the list of indices
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

    def extract_main_data_fn(self, main_data: Dict[str, str]) -> Dict[str, Any]:
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

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
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
