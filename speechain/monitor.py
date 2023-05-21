"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import copy
import os
import argparse
import time
import warnings

import GPUtil
import shutil
from contextlib import contextmanager

import torch
import numpy as np
import torch.multiprocessing as mp

try:
    import soundfile as sf
except OSError as e:
    warnings.warn(f"Monitor meets {e} when importing soundfile.")

from typing import Dict, List
from abc import ABC, abstractmethod
from multiprocessing import Process, Queue, Event

try:
    from speechain.snapshooter import snapshot_logs
except ImportError as e:
    warnings.warn(f"Monitor meets {e} when importing speechain.snapshooter.snapshot_logs.")
from speechain.model.abs import Model
from speechain.utilbox.md_util import get_table_strings, get_list_strings

from speechain.utilbox.import_util import parse_path_args
from speechain.utilbox.data_saving_util import save_data_by_format
from speechain.utilbox.data_loading_util import search_file_in_subfolder, get_file_birthtime


class Monitor(ABC):
    """
        Abstract Base Class (ABC) for a Monitor that keeps track of various aspects of training and evaluatio.
    """
    def __init__(self, logger, args: argparse.Namespace, result_path: str = None, **kwargs):
        """
            Initializes the monitor class.

            Args:
                logger:
                    A logger object for logging messages.
                args:
                    Command line arguments passed as a Namespace object.
                result_path:
                    Path to the directory where results should be saved.
                kwargs:
                    Additional keyword arguments.
        """
        # Initialize shared members
        self.logger = logger
        self.result_path = args.train_result_path if result_path is None else parse_path_args(result_path)
        self.gpus = args.gpus if isinstance(args.gpus, List) else [args.gpus]

        # Initialize shared record information
        self.epoch_records = dict(
            consumed_time=dict(
                data_load_time=[],
                model_forward_time=[]
            ),
            consumed_memory=dict(),
            criteria=dict()
        )
        # Add GPU ranks to consumed_memory if they don't exist
        for rank, gpu in enumerate(self.gpus):
            if f'Rank{rank}' not in self.epoch_records['consumed_memory'].keys():
                self.epoch_records['consumed_memory'][f'Rank{rank}'] = []

        # Initialize step-level records
        self.step_records = dict(
            consumed_time=dict(
                data_load_time=[],
                model_forward_time=[]
            ),
            criteria=dict()
        )
        self.mode = None
        self.monitor_init(args, **kwargs)

        # Initialize snapshooter for log snapshotting
        self.logs_queue = Queue()
        snapshot_conf = dict(
            result_path=self.result_path, snap_mode=self.mode, **args.monitor_snapshot_conf
        )
        # Initialize multiprocessing event to enable communication with the snapshooter process
        self.event = Event()
        self.event.clear()
        Process(target=snapshot_logs, args=(self.logs_queue, self.event, snapshot_conf), daemon=True).start()

    def enqueue(self, logs: Dict or List[Dict]):
        """
            Enqueues logs into the logs_queue.

            Args:
                logs:
                    A dictionary or list of dictionaries containing logs.
        """
        if isinstance(logs, Dict):
            self.logs_queue.put(logs)
        elif isinstance(logs, List):
            for log in logs:
                self.logs_queue.put(log)
        else:
            raise RuntimeError("Expected logs to be a Dict or a List[Dict].")

    def empty_queue(self):
        """
            Checks if the logs queue is empty.

            Returns:
                Boolean value indicating whether the logs queue is empty or not.
        """
        self.logs_queue.qsize()
        return self.logs_queue.empty()

    @contextmanager
    def measure_time(self, names: str or List[str]):
        """
            Context manager for measuring time of execution.

            Args:
                names:
                    Name or list of names for the operations to be timed.
        """
        start = time.perf_counter()
        yield
        t = time.perf_counter() - start

        names = ["consumed_time", names] if isinstance(names, str) else ["consumed_time"] + names
        dict_pointer = self.step_records
        for i, name in enumerate(names):
            if name not in dict_pointer.keys():
                dict_pointer[name] = [] if i == len(names) - 1 else dict()
            dict_pointer = dict_pointer[name]
        dict_pointer.append(t)

    def refresh_step_records(self, records: Dict = None):
        """
            Refreshes the step records by resetting the values.

            Args:
                records:
                    A dictionary containing the records to be refreshed.
        """
        if records is None:
            records = self.step_records
        if isinstance(records, Dict):
            for key in records.keys():
                if isinstance(records[key], Dict):
                    self.refresh_step_records(records[key])
                elif isinstance(records[key], List):
                    records[key] = []
                else:
                    raise RuntimeError(f"Unexpected type in records: {type(records[key])}.")
        else:
            raise RuntimeError("Expected records to be of type Dict.")

    def record_step_info(self, key: str, step_info: Dict):
        """
            Records information at each step during training or evaluation.

            Args:
                key:
                    Key under which information should be recorded.
                step_info:
                    Dictionary containing the information to be recorded.
        """
        for name, info in step_info.items():
            if name not in self.step_records[key].keys():
                self.step_records[key][name] = []
            # result is in the form of torch.Tensor, so it needs to be transformed by .item()
            if isinstance(info, (torch.Tensor, np.ndarray)):
                if len(info.shape) == 1:
                    info = info[0]
                info = info.item()
            elif isinstance(info, List):
                info = info[0]
            self.step_records[key][name].append(info)

    def record_consumed_time(self, epoch_message: str):
        """
            Records time consumed in each epoch during training or evaluation.

            Args:
                epoch_message:
                    String to be included in the epoch message.

            Returns:
                The updated epoch message.
        """
        epoch_message += " -- Consumed Time -- \n"

        # Record data loading time
        _total_time = sum(self.step_records['consumed_time']['data_load_time'])
        epoch_message += f"Total data load time: {_total_time:.2f}s -- "
        self.epoch_records['consumed_time']['data_load_time'].append(_total_time)

        # Record model forward time
        _total_time = sum(self.step_records['consumed_time']['model_forward_time'])
        epoch_message += f"Total model forward time: {_total_time:.2f}s -- "
        self.epoch_records['consumed_time']['model_forward_time'].append(_total_time)
        epoch_message += "\n"

        return epoch_message

    def record_consumed_memory(self, epoch_message: str):
        """
            Records memory consumed in each epoch during training or evaluation.

            Args:
                epoch_message:
                    String to be included in the epoch message.

            Returns:
                The updated epoch message.
        """
        epoch_message += " -- Consumed Memory -- \n"
        gpus = GPUtil.getGPUs()
        if len(gpus) == 0:
            self.logger.warn(f"GPUtil.getGPUs() returns nothing at the {self.mode} part of epoch no.{self.epoch}. ")

        # Record consumed memory for each GPU
        for rank, gpu in enumerate(self.gpus):
            # recover the GPU number from 'CUDA_VISIBLE_DEVICES'
            if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
                gpu = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[gpu]
                # if gpu is in the form of the local ID, it will be converted into an integer for List indexing
                if gpu.isdigit():
                    gpu = int(gpu)
                # if gpu is in the form of the UUID, it will be kept as a string

            # --- torch.cuda is only able to report the GPU used in the current rank --- #
            # --- but torch.cuda can report the precise allocated and reserved memory information of the model --- #
            # turn bytes into MB，
            # memory_allocated = int(torch.cuda.memory_allocated(gpu) * (10 ** (-6)))
            # memory_reserved = int(torch.cuda.memory_reserved(gpu) * (10 ** (-6)))
            # epoch_message += f"GPU rank no.{rank} (cuda:{gpu}): " \
            #                  f"allocated memory {memory_allocated} MB, " \
            #                  f"reserved memory {memory_reserved} MB -- "
            # self.epoch_records['consumed_memory'][f'Rank{rank}']['memory_allocated'].append(memory_allocated)
            # self.epoch_records['consumed_memory'][f'Rank{rank}']['memory_reserved'].append(memory_reserved)

            # --- GPUtil can load the information of all the GPUs no matter the rank of the current process --- #
            # --- but it sometimes fails to report anything because of some IO errors --- #
            # --- and the returned used memory is the overall memory consumption of the GPU, --- #
            # --- which means that if there are more than one jobs running on the same GPU, --- #
            # --- the used memory is not precise for the current job. --- #
            memory_used = 0
            # for local GPU ID
            if isinstance(gpu, int) and gpu < len(gpus):
                memory_used = gpus[gpu].memoryUsed
                gpu_name = f'cuda:{gpu}'
            # for GPU UUID
            elif isinstance(gpu, str):
                curr_gpu = None
                for g in gpus:
                    if g.uuid == gpu:
                        curr_gpu = g
                if curr_gpu is not None:
                    memory_used = curr_gpu.memoryUsed
                gpu_name = gpu
            else:
                raise RuntimeError(f'Unexpected errors happen when retrieving GPU IDs. (got {gpu})')
            # if some unexpected errors happen above, memory_used will remain 0
            epoch_message += f"GPU rank no.{rank} ({gpu_name}): {memory_used} MB -- "

            if f'Rank{rank}' not in self.epoch_records['consumed_memory'].keys():
                self.epoch_records['consumed_memory'][f'Rank{rank}'] = []
            self.epoch_records['consumed_memory'][f'Rank{rank}'].append(memory_used)
        epoch_message += "\n"

        return epoch_message

    def record_criteria(self, epoch_message: str):
        """
            Records criteria in each epoch during training or evaluation.

            Args:
                epoch_message:
                    String to be included in the epoch message.

            Returns:
                The updated epoch message.
        """
        epoch_message += " -- Criteria information -- \n"

        # Loop through all training criteria and record average and standard deviation
        for name, results in self.step_records['criteria'].items():
            if name not in self.epoch_records['criteria'].keys():
                self.epoch_records['criteria'][name] = []

            # calculate the average criterion value
            aver_result = np.mean(results).item()
            std_result = np.std(results).item()
            epoch_message += f"Average {name}: {aver_result:.2e} ± {std_result:.2f}\n"
            # record the average criterion value
            self.epoch_records['criteria'][name].append(aver_result)
        epoch_message += "\n"

        return epoch_message

    @abstractmethod
    def monitor_init(self, args: argparse.Namespace, **kwargs):
        """
            Abstract method for initializing the monitor.

            Args:
                args:
                    Command line arguments passed as a Namespace object.
                kwargs:
                    Additional keyword arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def start_epoch(self, **kwargs):
        """
            Abstract method to be called at the start of each epoch.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, **kwargs):
        """
            Abstract method to be called at each step in an epoch.
        """
        raise NotImplementedError

    @abstractmethod
    def finish_epoch(self, **kwargs):
        """
            Abstract method to be called at the end of each epoch.
        """
        raise NotImplementedError

    @abstractmethod
    def state_dict(self):
        """
            This method currently performs no operation and always returns None.

            The intention is for subclasses to override this method to return a dictionary
            containing the state of the Monitor. However, in the base Monitor class,
            it does not have any state to save, so it returns None.

            Returns:
                None
        """
        return None

    def load_state_dict(self, state_dict: Dict = None):
        """
            Loads the Monitor state.

            Args:
                state_dict:
                    Dictionary containing a whole state of the Monitor.
        """
        if state_dict is not None:
            for key, value in state_dict.items():
                self.__setattr__(key, value)


class TrainMonitor(Monitor):
    """
        The TrainMonitor class extends the Monitor class by adding functionality to track training progress and
        performance. It provides methods to initialize monitoring, record metrics at each step and epoch, and store the
        state of the training process.
    """

    def monitor_init(self, args: argparse.Namespace, **kwargs):
        """
            Initializes the training monitor with the given arguments.
            This method is responsible for setting up the general members and tracking optimizer information.

            Args:
                args (argparse.Namespace):
                    Arguments provided for monitoring.
        """
        # general members
        self.report_per_steps = args.report_per_steps
        self.dry_run = args.dry_run
        self.no_optim = args.no_optim
        self.mode = 'train'

        # training monitor needs to additionally track optimizer information
        # update epoch-level records
        self.epoch_records['consumed_time'].update(
            loss_backward_time=dict(),
            optim_time=dict()
        )
        self.epoch_records.update(
            optim_lr=dict()
        )
        # update step-level records
        self.step_records['consumed_time'].update(
            loss_backward_time=dict(),
            optim_time=dict()
        )
        self.step_records.update(
            optim_lr=dict()
        )

    def start_epoch(self, epoch: int):
        """
            Prepares the monitor for a new epoch of training.

            Args:
                epoch (int):
                    The current epoch number.
        """
        # epoch-level information
        self.epoch = epoch
        self.epoch_start_time = time.time()

        # refresh the step-level records at the beginning of each epoch
        self.refresh_step_records()

        # logging the beginning information
        self.logger.info(f"The training part of epoch no.{epoch} starts.")

    def step(self, step_num: int, optim_lr: Dict[str, float], train_metrics: Dict[str, torch.Tensor]):
        """
            Records information for a single training step.

            Args:
                step_num (int):
                    The current step number.
                optim_lr (Dict[str, float]):
                    The learning rates for each optimizer.
                train_metrics (Dict[str, torch.Tensor]):
                    The training metrics for the current step.
        """
        # accumulate the values of training criteria
        if train_metrics is not None:
            self.record_step_info('criteria', train_metrics)

        # accumulate the optimization times and learning rates of each OptimScheduler
        if optim_lr is not None:
            self.record_step_info('optim_lr', optim_lr)

        # report all the information for every 'report_per_steps' training steps
        if step_num % self.report_per_steps == 0:
            # calculate the accumulated time for reporting
            _data_load_time = sum(self.step_records['consumed_time']['data_load_time'][-self.report_per_steps:])
            _model_forward_time = sum(self.step_records['consumed_time']['model_forward_time'][-self.report_per_steps:])
            # initialize the returned message of the current step
            step_message = f"Training step no.{step_num - self.report_per_steps + 1:d}-{step_num:d} -- " \
                           f"data loading time: {_data_load_time:.2f}s -- " \
                           f"model forward time: {_model_forward_time:.2f}s -- "

            if not self.dry_run:
                # report the values of criteria in each training step
                step_message += "Training Criteria: "
                for name, result in self.step_records['criteria'].items():
                    _tmp_criteria = result[-self.report_per_steps:]
                    step_message += f"{name}: {np.mean(_tmp_criteria):.2e} -- "

                if not self.no_optim:
                    # report the information of optimizers after each training step
                    step_message += "OptimSchedulers: "
                    for optim_name in self.step_records['optim_lr'].keys():
                        # accumulate the backward and optimization times
                        _loss_backward_time = sum(
                            self.step_records['consumed_time']['loss_backward_time'][optim_name][
                            -self.report_per_steps:])
                        _optim_time = sum(
                            self.step_records['consumed_time']['optim_time'][optim_name][-self.report_per_steps:])
                        # average the learning rate
                        _lr = sum(
                            self.step_records['optim_lr'][optim_name][-self.report_per_steps:]) / self.report_per_steps

                        # accumulated optimization time and averaged learning_rates are reported
                        step_message += f"{optim_name}: " \
                                        f"loss backward time= {_loss_backward_time:.2f}s, " \
                                        f"optimization time={_optim_time:.2f}s, " \
                                        f"learning rate={_lr:.2e} -- "

            # logging the information of the current step
            self.logger.info(step_message)

    def finish_epoch(self):
        """
            Finishes monitoring for the current epoch, logging information and preparing for the next epoch.
        """
        # ---- The Information Logging Part ---- #
        # report the overall consuming time of the current epoch
        epoch_message = f"The training part of epoch no.{self.epoch} is finished in {time.time() - self.epoch_start_time:.2f}s.\n" \
                        f"Summary of all training steps:\n"

        # report the information of the consumed calculation time
        epoch_message = self.record_consumed_time(epoch_message)

        # report the information of the consumed GPU memory
        epoch_message = self.record_consumed_memory(epoch_message)

        # data loading only
        if not self.dry_run:
            # report the information of all the training criteria
            epoch_message = self.record_criteria(epoch_message)

            # no optimization
            if not self.no_optim:
                # report the information of all the OptimSchedulers
                epoch_message += " -- OptimScheduler information -- \n"
                # record the optimization information of the current epoch
                for optim_name in self.step_records['optim_lr'].keys():
                    if optim_name not in self.epoch_records['optim_lr'].keys():
                        self.epoch_records['optim_lr'][optim_name] = []
                    if optim_name not in self.epoch_records['consumed_time']['loss_backward_time'].keys():
                        self.epoch_records['consumed_time']['loss_backward_time'][optim_name] = []
                    if optim_name not in self.epoch_records['consumed_time']['optim_time'].keys():
                        self.epoch_records['consumed_time']['optim_time'][optim_name] = []

                    epoch_message += f"{optim_name} -- "
                    # accumulate the loss backward time
                    _total_time = sum(self.step_records['consumed_time']['loss_backward_time'][optim_name])
                    self.epoch_records['consumed_time']['loss_backward_time'][optim_name].append(_total_time)
                    epoch_message += f"Total loss backward time: {_total_time:.2f}s, "

                    # accumulate the optimization time
                    _total_time = sum(self.step_records['consumed_time']['optim_time'][optim_name])
                    self.epoch_records['consumed_time']['optim_time'][optim_name].append(_total_time)
                    epoch_message += f"Total optimization time: {_total_time:.2f}s, "

                    # average the learning rate
                    aver_lr = np.mean(self.step_records['optim_lr'][optim_name]).item()
                    self.epoch_records['optim_lr'][optim_name].append(aver_lr)
                    epoch_message += f"Average learning rate: {aver_lr:.2e}\n"
                epoch_message += "\n"

        # logging the information for the current epoch
        self.logger.info(epoch_message)

        # ---- The SnapShotting Part ---- #
        for key in self.epoch_records.keys():
            # only snapshot the time records in the dry running mode
            if self.dry_run and key != 'consumed_time':
                continue
            # skip the learning rate records in the no optimization mode
            if self.no_optim and key == 'optim_lr':
                continue
            # snapshot the epoch records so for to a curve figure
            self.enqueue(
                dict(
                    materials=copy.deepcopy(self.epoch_records[key]), plot_type='curve',
                    epoch=self.epoch, xlabel="epoch", sep_save=False, subfolder_names=key
                )
            )

        # notify the snapshooter process of the new queue elements
        self.event.set()

    def state_dict(self):
        """
            Returns the current state of the monitor as a dictionary.

            Returns:
                dict: The current state of the monitor.
        """
        return dict(
            epoch_records=self.epoch_records
        )


class ValidMonitor(Monitor):
    """
        The ValidMonitor class extends the Monitor class by adding functionality to track validation progress and
        performance. It provides methods to initialize monitoring, record metrics at each validation step, and snapshot
        the model at each epoch.
    """
    def monitor_init(self, args: argparse.Namespace, model: Model = None):
        """
            Initializes the validation monitor with the given arguments and the model.
            This method is responsible for setting up the general members, tracking best models, early stopping, and last models.

            Args:
                args (argparse.Namespace):
                    Arguments provided for monitoring.
                model (Model, optional):
                    The model being validated. This parameter must not be None.

            Raises:
                AssertionError: If the provided model is None.
        """
        assert model is not None, "Model must be provided and not None."
        # register a pointer of the model
        self.model = model

        # running mode
        self.dry_run = args.dry_run
        self.no_optim = args.no_optim
        self.mode = 'valid'

        # best models-related members
        self.best_model_selection = args.best_model_selection
        # receive a single metric as a standalone list or tuple
        if isinstance(self.best_model_selection, (List, tuple)) and \
                isinstance(self.best_model_selection[0], str):
            self.best_model_selection = [self.best_model_selection]
        else:
            assert isinstance(self.best_model_selection, List), \
                f"best_model_selection must be given as a list, " \
                f"but got type(best_model_selection)={self.best_model_selection}."

        for i in range(len(self.best_model_selection)):
            # checking the argument types
            assert isinstance(self.best_model_selection[i], (List, tuple)), \
                "Each element of best_model_selection must be either a list or a tuple, " \
                f"but got type={type(self.best_model_selection[i])}."
            assert len(self.best_model_selection[i]) == 4, \
                f"Each element of best_model_selection must be a quad-tuple or qual-list, " \
                f"but got length={len(self.best_model_selection[i])}."

            if isinstance(self.best_model_selection[i], tuple):
                self.best_model_selection[i] = list(self.best_model_selection[i])
            self.best_model_selection[i][2] = self.best_model_selection[i][2].lower()
            assert self.best_model_selection[i][2] in ['max', 'min'], \
                f"The best_model_mode must be either 'max' or 'min', but got {self.best_model_selection[i][2]}."

        # model saving-related members
        self.best_model_performance = dict()
        for metric in self.best_model_selection:
            self.best_model_performance['_'.join(metric[:2])] = dict()
        self.saved_model_epoch = []
        self.model_save_path = os.path.join(self.result_path, 'models')
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path, exist_ok=True)

        # early stopping-related members, the first metric in self.best_model_selection is used
        self.early_stopping_metric = '_'.join(self.best_model_selection[0][:2])
        self.early_stopping_mode = self.best_model_selection[0][2]
        self.early_stopping_patience = args.early_stopping_patience
        self.early_stopping_threshold = args.early_stopping_threshold
        self.early_stopping_epochs = 0
        self.last_best_performance = 0.0 if self.early_stopping_mode == 'max' else torch.inf

        # last models-related members
        self.last_model_number = args.last_model_number
        if self.last_model_number < 1:
            raise ValueError("last_model_number cannot be lower than 1, "
                             "otherwise the training will not be able to resume."
                             f"Got last_model_number={self.last_model_number}!")

        # initialize the snapshooter of this validation monitor
        self.visual_snapshot_interval = args.visual_snapshot_interval

    def start_epoch(self, epoch: int):
        """
            Prepares the monitor for a new epoch of validation.

            Args:
                epoch (int): The current epoch number.
        """
        # epoch-level information
        if epoch in self.best_model_performance.keys():
            self.logger.warning(f"The record of epoch no.{epoch} has already existed in the monitor! "
                                f"It will be overwritten by the new record obtained shortly thereafter.")
        self.epoch = epoch
        self.epoch_start_time = time.time()

        # refresh the step-level records at the beginning of each epoch
        self.refresh_step_records()

        # logging the beginning information
        self.logger.info(f"The validation part of epoch no.{epoch} starts.")

    def step(self, valid_metrics: Dict[str, torch.Tensor]):
        """
            Records information for a single validation step.

            Args:
                valid_metrics (Dict[str, torch.Tensor]):
                    The validation metrics for the current step.
        """
        # accumulate the values of validation criteria
        if valid_metrics is not None:
            self.record_step_info('criteria', valid_metrics)

    def model_snapshot(self, epoch: int, domain: str, sample_index: str, used_sample: Dict):
        """
            Takes a snapshot of the model at the given epoch for a given sample.

            Args:
                epoch (int):
                    The current epoch number.
                domain (str):
                    The domain of the current sample.
                sample_index (str):
                    The index of the current sample.
                used_sample (Dict):
                    The current sample being used for validation.
        """
        # initialize the sub-dict for each sample
        if sample_index not in self.epoch_records.keys():
            self.epoch_records[sample_index] = dict()

        # get the visualization logs for model snapshotting
        vis_logs = self.model(batch_data=used_sample, epoch=epoch, domain=domain,
                              epoch_records=self.epoch_records, sample_index=sample_index,
                              snapshot_interval=self.visual_snapshot_interval)

        # put all the visualization logs into the queue
        self.enqueue(vis_logs)

    @staticmethod
    def is_better(query: int or float, target: int or float, mode: str, threshold: float = 0.0):
        """
            Compares a query value with a target value under a specified mode, optionally considering a threshold.

            Parameters:
                query (int, float):
                    The value to be compared against the target.
                target (int, float):
                    The reference value for the comparison.
                mode (str):
                    The comparison mode - 'max' implies the query is considered better if it's larger,
                    'min' implies the query is considered better if it's smaller.
                threshold (float, optional):
                    A value that adjusts the target value before comparison. Default is 0.0.

            Returns:
                bool: True if the query value is considered better than the target, False otherwise.
        """
        _target = target
        # relative threshold if the argument value is positive
        if threshold > 0:
            _target *= 1 + threshold if mode == 'max' else 1 - threshold
        # absolute threshold if the argument value is negative
        elif threshold < 0:
            _target += -threshold if mode == 'max' else threshold

        # the threshold is applied to the better comparison
        return query > _target if mode == 'max' else query < _target

    def model_insert(self, train_records: Dict, valid_flag: bool):
        """
            Inserts the model into the ensemble if it's better than the existing models. The model is evaluated using
            the training records and the validation flag.

            Parameters:
                train_records (Dict):
                    A dictionary containing the training records, presumably including model performance metrics.
                valid_flag (bool):
                    A flag indicating whether the model has passed validation.

            Returns:
                None
        """
        # loop each metric for best model selection
        for metric in self.best_model_selection:
            if metric[0] == 'valid' and not valid_flag:
                continue

            _metric_name, _metric_mode, _model_num = '_'.join(metric[:2]), metric[2], metric[3]
            _criteria_dict = train_records['criteria'] if metric[0] == 'train' else self.epoch_records['criteria']
            curr_performance = _criteria_dict[metric[1]][-1]

            # controls whether to insert the current model into self.best_model_performance or not
            model_insert_flag = False

            # if there is no empty positions for the model of the current epoch
            if len(self.best_model_performance[_metric_name]) == _model_num:
                # as long as the current performance is better than one existing record, it should be inserted.
                for performance in self.best_model_performance[_metric_name].values():
                    if self.is_better(query=curr_performance, target=performance, mode=_metric_mode):
                        model_insert_flag = True
                        break
            # True if there are some empty positions for the best models
            else:
                model_insert_flag = True

            # record the current model performance
            if model_insert_flag:
                # record the performance of the current epoch
                self.best_model_performance[_metric_name][self.epoch] = curr_performance

        # save the model of the latest epoch onto the disk
        torch.save(self.model.state_dict(), os.path.join(self.model_save_path, f"epoch_{self.epoch}.pth"))
        self.saved_model_epoch.append(self.epoch)

    def update_best_and_pop_worst(self, epoch_message: str):
        """
            Updates the best model in the ensemble and removes the worst model. The best and worst are determined
            based on the performance metrics. The function also handles logging related to model performance.

            Parameters:
                epoch_message (str):
                    A string message related to the current epoch.

            Returns:
                Tuple[str, bool, Dict[str, bool]]:
                    Returns a tuple containing the updated epoch message, a flag indicating whether early stopping
                    conditions are met, and flags related to model performance metrics.
        """

        def whether_remove(remove_epoch: int):
            # retain the last several models within self.last_model_number
            if self.epoch - remove_epoch < self.last_model_number:
                return False
            else:
                remove_flag = True
                # access metric_epoch_records from the outer scope
                for _epoch_record in metric_epoch_records.values():
                    if remove_epoch in _epoch_record['sorted_epochs']:
                        remove_flag = False
                        break
                return remove_flag

        # --- Gather the epoch record information for each metric --- #
        metric_epoch_records = dict()
        # loop each metric for best model selection
        for metric in self.best_model_selection:
            _metric_name, _metric_mode, _model_num = '_'.join(metric[:2]), metric[2], metric[3]

            # find the best epoch and worst epoch in self.best_model_performance
            sorted_epochs = dict(sorted(self.best_model_performance[_metric_name].items(), key=lambda x: x[1],
                                        reverse=True if _metric_mode == 'max' else False))
            metric_epoch_records[_metric_name] = dict(
                sorted_epochs=list(sorted_epochs.keys()),
                metric_mode=_metric_mode,
                model_num=_model_num
            )

        # --- Pop out the worst model and Update the model symbol links --- #
        metric_pop_flags = dict()
        for metric_name, epoch_record in metric_epoch_records.items():
            # controls whether the worst model has been pooped out or not
            metric_pop_flags[metric_name] = False
            # pop out the worst model if there is a redundant one in self.best_model_performance
            if len(epoch_record['sorted_epochs']) > epoch_record['model_num']:
                # pick up the epoch number of the worst model
                worst_epoch = epoch_record['sorted_epochs'][-1]
                self.best_model_performance[metric_name].pop(worst_epoch)
                epoch_record['sorted_epochs'].remove(worst_epoch)
                metric_pop_flags[metric_name] = True

            # update the symbol links of all the best models so far
            for i, epoch in enumerate(epoch_record['sorted_epochs']):
                _best_model_pointer = f"{metric_name}_best.pth" if i == 0 else f"{metric_name}_best_{i + 1}.pth"
                # create a soft link from the best model pointer to the model fi le of the current epoch
                symlink_dst = os.path.join(self.model_save_path, _best_model_pointer)
                if os.path.islink(symlink_dst) or os.path.exists(symlink_dst):
                    os.unlink(symlink_dst)
                os.symlink(os.path.join(self.model_save_path, f"epoch_{epoch}.pth"), symlink_dst)

        # update the symbol links of the last several models
        for epoch in range(self.epoch, max(0, self.epoch - self.last_model_number), -1):
            _last_model_pointer = f"latest.pth" if epoch == self.epoch else f"last_{self.epoch - epoch + 1}.pth"
            # create a soft link from the best model pointer to the model file of the current epoch
            symlink_dst = os.path.join(self.model_save_path, _last_model_pointer)
            if os.path.islink(symlink_dst) or os.path.exists(symlink_dst):
                os.unlink(symlink_dst)
            os.symlink(os.path.join(self.model_save_path, f"epoch_{epoch}.pth"), symlink_dst)

        # remove the redundant model files
        saved_epochs = self.saved_model_epoch.copy()
        for epoch in saved_epochs:
            epoch_model_path = os.path.join(self.model_save_path, f"epoch_{epoch}.pth")
            if whether_remove(epoch):
                # remove the record of epoch in the memory
                self.saved_model_epoch.remove(epoch)

                # remove the model file if it exists
                if os.path.exists(epoch_model_path):
                    # ensure that the model to be removed is successfully removed
                    while os.path.exists(epoch_model_path):
                        os.remove(epoch_model_path)

        # --- Early-Stopping epoch number checking for the early-stopping metric --- #
        if len(metric_epoch_records[self.early_stopping_metric]['sorted_epochs']) != 0:
            best_epoch = metric_epoch_records[self.early_stopping_metric]['sorted_epochs'][0]
            # refresh to 0 or add 1 depending on the comparison the best one and the second best one
            if best_epoch == self.epoch:
                epoch_message += \
                    f"{self.early_stopping_metric} of the current epoch no.{self.epoch} is the best so far.\n"

                # compare the current performance and the last best performance
                best_performance = self.best_model_performance[self.early_stopping_metric][best_epoch]
                if self.is_better(best_performance, self.last_best_performance,
                                  mode=self.early_stopping_mode, threshold=self.early_stopping_threshold):
                    epoch_message += f"The early-stopping threshold {self.early_stopping_threshold} is reached, " \
                                     "so the early-stopping epoch number is refreshed.\n"
                    self.early_stopping_epochs = 0
                    self.last_best_performance = best_performance
                else:
                    epoch_message += f"The early-stopping threshold {self.early_stopping_threshold} is not reached, " \
                                     "so the early-stopping epoch number keeps increasing.\n"
                    self.early_stopping_epochs += 1
            # directly add 1 if the current epoch is not the best
            else:
                epoch_message += \
                    f"No improvement of {self.early_stopping_metric} in the current epoch no.{self.epoch}.\n"
                self.early_stopping_epochs += 1

            # report the updated early-stopping epoch number
            epoch_message += f"The early-stopping epoch number has been updated to {self.early_stopping_epochs}.\n"

        # early-stopping check by the patience
        early_stopping_flag = False
        if self.early_stopping_patience is not None and self.early_stopping_epochs > self.early_stopping_patience:
            epoch_message += f"The early-stopping patience {self.early_stopping_patience} is reached, " \
                             f"so the training process stops here.\n"
            early_stopping_flag = True
        elif self.early_stopping_patience is None:
            epoch_message += "The early-stopping patience is not set by your exp_cfg, " \
                             "so the training will continue until num_epochs is reached.\n"

        return epoch_message, early_stopping_flag, metric_pop_flags

    def save_aver_model(self, epoch_message: str, metric_pop_flags: Dict[str, bool]):
        """
            Stores the average model in the ensemble and updates the epoch message based on the model performance metrics.

            Parameters:
                epoch_message (str):
                    A string message related to the current epoch.
                metric_pop_flags (Dict[str, bool]):
                    A dictionary containing flags related to model performance metrics.

            Returns:
                str: The updated epoch message.
        """
        def save_aver_models(aver_epoch_list: List, aver_num: int, aver_model_name: str):
            # no average model is saved if there is only one candidate model
            if len(aver_epoch_list) == 1:
                return ""

            # sum up the parameters of all best models
            avg_model = None
            for epoch in aver_epoch_list:
                _tgt_model_path = os.path.join(self.model_save_path, f"epoch_{epoch}.pth")
                # skip if the model doesn't exist
                if not os.path.exists(_tgt_model_path):
                    continue

                _avg = None
                # access self.model_save_path from the outer scope
                if avg_model is not None:
                    _avg = torch.load(_tgt_model_path, map_location="cpu")
                else:
                    avg_model = torch.load(_tgt_model_path, map_location="cpu")

                if _avg is not None:
                    for key in avg_model.keys():
                        avg_model[key] += _avg[key]

            # if no average model, skip this function and return an empty string
            if avg_model is None:
                return ""
            else:
                # for the parameters whose dtype is int, averaging is not performed
                # reference: https://github.com/espnet/espnet/blob/5fa6dcc4e649dc66397c629d0030d09ecef36b80/espnet2/main_funcs/average_nbest_models.py#L90
                for key in avg_model.keys():
                    if not str(avg_model[key].dtype).startswith("torch.int"):
                        avg_model[key] /= aver_num

                # save the average model
                _aver_model_path = os.path.join(self.model_save_path, aver_model_name)
                torch.save(avg_model, _aver_model_path)

                return f"{aver_model_name} has been updated to the average of epochs {aver_epoch_list}.\n"

        # --- Save the average model for the best models of each metric --- #
        # loop each metric for best model selection
        for metric in self.best_model_selection:
            _metric_name, _metric_mode, _model_num = '_'.join(metric[:2]), metric[2], metric[3]

            # average the recorded best models so far
            if len(self.best_model_performance[_metric_name]) == _model_num and metric_pop_flags[_metric_name]:
                epoch_message += save_aver_models(
                    aver_epoch_list=list(self.best_model_performance[_metric_name].keys()),
                    aver_num=len(self.best_model_performance[_metric_name]),
                    aver_model_name=f"{_model_num}_{_metric_name}_average.pth"
                )

        # --- Save the average model of the last models --- #
        if self.epoch >= self.last_model_number:
            epoch_message += save_aver_models(
                aver_epoch_list=list(range(self.epoch, self.epoch - self.last_model_number, -1))[::-1],
                aver_num=self.last_model_number,
                aver_model_name=f"{self.last_model_number}_last_average.pth"
            )

        return epoch_message

    def finish_epoch(self, train_records: Dict, valid_flag: bool, valid_per_epochs: int):
        """
            Completes the current epoch by evaluating the model on the training records and potentially performing
            validation based on the validation flag and number of validation epochs.

            Parameters:
                train_records (Dict):
                    A dictionary containing the training records, presumably including model performance metrics.
                valid_flag (bool):
                    A flag indicating whether the model has passed validation.
                valid_per_epochs (int):
                    The number of epochs between validations.

            Returns:
                bool: A flag indicating whether early stopping conditions are met.
        """
        # ---- The Information Logging Part ---- #
        if valid_flag:
            # report the overall consuming time of the current validation epoch
            epoch_message = f"The validation part of epoch no.{self.epoch} is finished in " \
                            f"{time.time() - self.epoch_start_time:.2f}s.\n" \
                            f"Summary of all validation steps:\n"

            # report the information of the consumed calculation time
            epoch_message = self.record_consumed_time(epoch_message)

            # report the information of the consumed GPU memory
            epoch_message = self.record_consumed_memory(epoch_message)

            if not self.dry_run:
                # report the information of all the validation criteria
                epoch_message = self.record_criteria(epoch_message)

            # ---- The SnapShotting Part ---- #
            for key in self.epoch_records.keys():
                # only snapshot the time and memory info in the dry running mode
                if self.dry_run and key != 'consumed_time':
                    continue
                # skip the model visualization records
                elif key in ['consumed_time', 'consumed_memory', 'criteria']:
                    # snapshot the epoch records so for to a curve figure
                    self.enqueue(
                        dict(
                            materials=copy.deepcopy(self.epoch_records[key]), plot_type='curve',
                            epoch=self.epoch, xlabel="epoch", sep_save=False, subfolder_names=key,
                            x_stride=valid_per_epochs
                        )
                    )

            # notify the snapshooter process of the new queue elements
            self.event.set()

        else:
            epoch_message = f"The validation part of epoch no.{self.epoch} is skipped.\n"

        # ---- The Model Saving and Early Stopping Part ---- #
        early_stopping_flag = False
        if not self.dry_run:
            # insert the current model into self.best_model_performance if needed
            self.model_insert(train_records, valid_flag)

            # After inserting, deal with the worst model performance so far and check the early-stopping
            epoch_message, early_stopping_flag, metric_pop_flags = self.update_best_and_pop_worst(epoch_message)

            # save the average models of the best models so far if needed
            epoch_message = self.save_aver_model(epoch_message, metric_pop_flags)

        # log the information of the current validation epoch
        self.logger.info(epoch_message)
        return early_stopping_flag

    def state_dict(self):
        """
            Retrieves the state of the ensemble in dictionary form.

            Returns:
                Dict: The current state of the ensemble, typically including model parameters and other related information.
        """
        return dict(
            epoch_records=self.epoch_records,
            saved_model_epoch=self.saved_model_epoch,
            best_model_performance=self.best_model_performance,
            early_stopping_epochs=self.early_stopping_epochs,
            last_best_performance=self.last_best_performance
        )


class TrainValidMonitor(object):
    """
        A wrapper class for TrainMonitor and ValidMonitor.
        The motivations of wrapping TrainMonitor and ValidMonitor together are two-folds:
            1. enable multi-metric best model recording among training and validation metrics.
            2. decouple TrainMonitor and ValidMonitor from Runner to improve cohesion and code readability.

        Attributes:
            logger: Logger object for logging purposes.
            train_monitor: An instance of the TrainMonitor class.
            valid_monitor: An instance of the ValidMonitor class.
    """

    def __init__(self, logger, args: argparse.Namespace, model: Model):
        """
            Initializes the TrainValidMonitor with a logger, arguments, and a model.

            Args:
                logger:
                    Logger object for logging purposes.
                args:
                    argparse.Namespace object, contains command line arguments.
                model:
                    Model object, the model to be trained and validated.
        """
        self.logger = logger

        self.train_monitor = TrainMonitor(logger=logger, args=args)
        self.valid_monitor = ValidMonitor(logger=logger, args=args, model=model)

    def start_train_epoch(self, epoch: int):
        """
            Starts a new training epoch.

            Args:
                epoch: The epoch number.
        """
        self.train_monitor.start_epoch(epoch)

    def train_step(self, step_num: int, optim_lr: Dict[str, float], train_metrics: Dict[str, torch.Tensor]):
        """
            Executes a training step.

            Args:
                step_num: The step number.
                optim_lr: The learning rate(s) of the optimizer(s).
                train_metrics: The training metrics.
        """
        self.train_monitor.step(step_num=step_num, optim_lr=optim_lr, train_metrics=train_metrics)

    def finish_train_epoch(self):
        """
            Finishes a training epoch.
        """
        self.train_monitor.finish_epoch()

    def start_valid_epoch(self, epoch: int):
        """
            Starts a new validation epoch.

            Args:
                epoch: The epoch number.
        """
        self.valid_monitor.start_epoch(epoch)

    def valid_step(self, valid_metrics: Dict[str, torch.Tensor]):
        """
            Executes a validation step.

            Args:
                valid_metrics: The validation metrics.
        """
        self.valid_monitor.step(valid_metrics=valid_metrics)

    def valid_model_snapshot(self, epoch: int, domain: str, sample_index: str, used_sample: Dict):
        """
            Creates a snapshot of the validation model.

            Args:
                epoch: The epoch number.
                domain: The domain of validation.
                sample_index: The sample index.
                used_sample: The sample used for validation.
        """
        self.valid_monitor.model_snapshot(epoch=epoch, domain=domain, sample_index=sample_index, used_sample=used_sample)

    def finish_valid_epoch(self, valid_flag: bool, valid_per_epochs: int):
        """
            Finishes a validation epoch.

            Args:
                valid_flag: The validation flag.
                valid_per_epochs: The number of epochs per validation.

            Returns:
                The result from the finish_epoch method of the ValidMonitor object.
        """
        return self.valid_monitor.finish_epoch(train_records=self.train_monitor.epoch_records, valid_flag=valid_flag,
                                               valid_per_epochs=valid_per_epochs)

    def wait_empty_queues(self, sleep_time: int = 10, max_wait_round: int = 60):
        """
            Check whether the snapshot creator processes of train_monitor and valid_monitor are still working.
            Wait until the material queues of the snapshot creators become empty.

            Args:
                sleep_time: The sleep time in seconds between each check. Default is 10 seconds.
                max_wait_round: The maximum number of waiting rounds. Default is 60 rounds.
        """
        if not self.train_monitor.empty_queue() or not self.valid_monitor.empty_queue():
            for _ in range(max_wait_round):
                message = ""
                if not self.train_monitor.empty_queue():
                    message += "The training snapshooter is still snapshotting. "
                if not self.valid_monitor.empty_queue():
                    message += "The validation snapshooter is still snapshotting. "
                self.logger.info(message + f"Waiting for {sleep_time} seconds......")
                time.sleep(sleep_time)
            self.logger.info(f"The maximal waiting time {max_wait_round * sleep_time} seconds is reached, "
                             f"so the snapshooters will be shut down......")

    def state_dict(self):
        """
            Retrieves the state of the TrainMonitor and ValidMonitor in dictionary form.

            Returns:
                Dict:
                    The current state of the TrainMonitor and ValidMonitor, typically including model parameters and
                    other related information.
        """
        return dict(
            train_monitor=self.train_monitor.state_dict(),
            valid_monitor=self.valid_monitor.state_dict()
        )

    def load_state_dict(self, state_dict):
        """
            Loads the state of the TrainMonitor and ValidMonitor from a dictionary.

            Args:
                state_dict:
                    The dictionary containing the state of the TrainMonitor and ValidMonitor.
        """
        self.train_monitor.load_state_dict(state_dict['train_monitor'])
        self.valid_monitor.load_state_dict(state_dict['valid_monitor'])


def data_saving_logs(proc_id: int, logs_queue: Queue, wait_time: int = 1):
    """
        Continuously monitors a queue for log data to save. If logs are available,
        they are retrieved from the queue and saved in the appropriate format.
        If the queue is empty, the function pauses for a specified wait time before checking again.
        In case of any exceptions during this process, a warning is issued and the function continues.

        Args:
            proc_id (int):
                The identifier of the current process.
            logs_queue (Queue):
                The queue containing log data to be saved.
            wait_time (int, optional):
                The number of seconds to wait before checking the queue again if it is empty. Defaults to 1.

        Returns:
            None
    """
    while True:
        try:
            # Continually check if the logs_queue has items to be saved
            if not logs_queue.empty():
                log = logs_queue.get()
                save_data_by_format(**log)
            else:
                # If the queue is empty, pause for a specified wait time
                time.sleep(wait_time)
        except Exception as e:
            warnings.warn(f"Process {proc_id} encountered an exception in data_saving_logs(): {e}. Continuing operation.")


class TestMonitor(Monitor):
    """
        Class responsible for monitoring the testing process and logging real-time information.
        It extends the base Monitor class.

        Attributes:
            distributed (bool):
                If True, indicates distributed training.
            report_per_steps (int):
                Frequency of reporting in steps.
            bad_cases_selection (List[List]):
                Criteria for selecting bad cases, if any.
            data_saving_logs_queue (Queue):
                Queue object for storing logs to be saved.
            prev_test_time (float):
                Time of the previous test, used to calculate elapsed time.
            total_step_num (int):
                Total number of steps in the test.
            step_info (dict):
                Information about the current test step.
            finished_group_num (int):
                Number of groups that have finished testing.
    """

    def monitor_init(self, args: argparse.Namespace, **kwargs):
        """
            Initializes the TestMonitor with the given arguments.

            Args:
                args (argparse.Namespace):
                    Parsed command line arguments.
        """
        self.distributed = args.distributed
        self.report_per_steps = args.report_per_steps
        self.bad_cases_selection = args.bad_cases_selection
        if self.bad_cases_selection is None:
            self.bad_cases_selection = []
        elif not isinstance(self.bad_cases_selection[0], List):
            self.bad_cases_selection = [self.bad_cases_selection]

        # initialize the snapshooter of the monitor
        self.data_saving_logs_queue = Queue()
        # create daemon processes for data saving
        assert args.saving_proc_num >= 1, "'saving_proc_num' should be an integer larger than 1!"
        self.saving_proc_num = args.saving_proc_num
        for proc_id in range(self.saving_proc_num):
            Process(target=data_saving_logs, args=(proc_id, self.data_saving_logs_queue), daemon=True).start()

    def start_epoch(self, total_step_num: int):
        """
            Starts a new testing epoch.

            Args:
                total_step_num (int): Total number of steps in the epoch.
        """
        # para init
        self.prev_test_time = time.time()
        self.total_step_num = total_step_num

        if not hasattr(self, 'step_info'):
            self.step_info = dict(
                group_time=[],
                total_time=0
            )
        if not hasattr(self, 'finished_group_num'):
            self.finished_group_num = 0

    def step(self, step_num: int, test_results: Dict[str, Dict], test_index: List[str]):
        """
            Executes a single test step and logs the results.

            Args:
                step_num (int): Current step number.
                test_results (Dict[str, Dict]): Results of the test step.
                test_index (List[str]): Indexes of the tests.
        """
        # --- Write the testing results of the current step to the testing files --- #
        # loop each result list in the returned Dict
        for name, result in test_results.items():
            # for .txt file, register the result contents into self.step_info. text data doesn't occupy too much memory
            if result['format'].lower() == 'txt':
                if name not in self.step_info.keys():
                    self.step_info[name] = dict()

                for index, content in zip(test_index, result['content']):
                    # for the List-type element, turn it into its string format
                    if isinstance(content, List):
                        content = str(content)
                    self.step_info[name][index] = content

            # for other files, data needs to be saved to the disk in real time to reduce the memory burden
            else:
                # we enqueue a whole batch of data to be saved in order to reduce the number of query operations of each daemon process
                self.data_saving_logs_queue.put(
                    dict(
                        file_format=result['format'].lower(),
                        save_path=os.path.join(self.result_path, name),
                        file_name_list=test_index,
                        file_content_list=result['content'],
                        group_ids=result['group_ids'] if 'group_ids' in result.keys() else None,
                        sample_rate=result['sample_rate'] if 'sample_rate' in result.keys() else None
                    )
                )

        # monitor the approximate size of the queue to be more memory-friendly
        while self.data_saving_logs_queue.qsize() > 3 * self.saving_proc_num:
            self.logger.warning(
                f"There has been more than {3 * self.saving_proc_num} batches in data_saving_logs_queue, "
                f"so the data generation is paused for 30 seconds.")
            time.sleep(30)

        # --- Report the testing midway information to users --- #
        test_step_message = None
        # record the tesing time of the current step
        curr_test_time = time.time()
        self.step_info['group_time'].append(curr_test_time - self.prev_test_time)
        self.prev_test_time = curr_test_time

        # meet the reporting interval
        if step_num % self.report_per_steps == 0:
            curr_group_time = sum(self.step_info['group_time'])

            # the first testing step
            if self.finished_group_num == 0:
                prev_group_time = curr_group_time
            # other testing steps
            else:
                # calculate the average time of all the previous groups
                prev_group_time = self.step_info['total_time'] / self.finished_group_num

            # calculate the number of remaining steps
            self.finished_group_num += 1
            finish_step_num = int(self.finished_group_num * self.report_per_steps)
            remaining_step_num = self.total_step_num - finish_step_num
            # take the weighted average consuming time of each group
            aver_group_time = (prev_group_time + curr_group_time) / 2
            remaining_time = aver_group_time * (remaining_step_num / self.report_per_steps)

            # update the time records
            self.step_info['total_time'] += curr_group_time
            self.step_info['group_time'] = []

            test_step_message = f"Testing Midway Report -- " \
                                f"testing time for the recent {self.report_per_steps} steps: {curr_group_time:.2f}s -- " \
                                f"finished step number: {finish_step_num} -- " \
                                f"remaining step number: {remaining_step_num} -- " \
                                f"expected remaining time: "

            remaining_days, remaining_time = int(remaining_time // (3600 * 24)), remaining_time % (3600 * 24)
            if remaining_days > 0:
                test_step_message += f"{remaining_days:d}d "

            remaining_hours, remaining_time = int(remaining_time // 3600), remaining_time % 3600
            if remaining_hours > 0:
                test_step_message += f"{remaining_hours:d}h "

            remaining_minutes, remaining_time = int(remaining_time // 60), remaining_time % 60
            if remaining_minutes > 0:
                test_step_message += f"{remaining_minutes:d}m "

            remaining_seconds = remaining_time
            test_step_message += f"{remaining_seconds:.2f}s"

        if test_step_message is not None:
            self.logger.info(test_step_message)

    def wait_empty_queues(self, sleep_time: int = 60):
        """
            Waits until the log queue is empty before continuing.

            Args:
                sleep_time (int, optional): Time to wait in seconds when the queue is not empty. Defaults to 60.
        """
        while True:
            if self.data_saving_logs_queue.empty():
                # wait for one more time when the queue becomes empty to left enough time for data saving
                time.sleep(sleep_time)
                break
            else:
                self.logger.info(f"The data saving process is still working. Waiting for {sleep_time} seconds.")
                time.sleep(sleep_time)

        # If using distributed training, synchronize all processes before continuing
        if self.distributed:
            torch.distributed.barrier()

    def finish_epoch(self, meta_info: Dict = None):
        """
            This method completes various tasks at the end of each epoch, such as:
            - Gathering checkpoint information from all GPU processes
            - Generating data path files
            - Producing evaluation reports for overall and group-level performance
            - Presenting top-N bad cases (if 'instance_reports.md' is in self.step_info)
            - Plotting histograms for numerical metrics in step_info

            Arguments:
                meta_info (Dict, optional):
                    Meta information about testing samples. Used for group-level evaluation.

            Note:
                The method is designed for a distributed setting, where results from different processes need to be gathered.
                If the program is not running in a distributed setting, some steps (like gathering checkpoint information) will be skipped.
        """
        # group all the testing samples by their meta info
        group_meta_info = None
        if meta_info is not None:
            group_meta_info = dict()
            # loop each type of meta data
            for meta_type, meta_dict in meta_info.items():
                if meta_type not in group_meta_info.keys():
                    group_meta_info[meta_type] = dict()

                # loop each group of samples
                for index, group in meta_info[meta_type].items():
                    if group not in group_meta_info[meta_type].keys():
                        group_meta_info[meta_type][group] = []
                    group_meta_info[meta_type][group].append(index)

        # --- Gather the checkpoint information of all the processes --- #
        # load the checkpoint of rank0 and delete testing time information
        self.load_state_dict(torch.load(os.path.join(self.result_path, 'rank0_tmp', 'checkpoint.pth'))['monitor'])
        self.step_info.pop('group_time')
        self.step_info.pop('total_time')

        if self.distributed:
            for rank in range(1, torch.distributed.get_world_size()):
                _tmp_dict = torch.load(os.path.join(self.result_path, f'rank{rank}_tmp', 'checkpoint.pth'))['monitor']['step_info']
                for key in self.step_info.keys():
                    self.step_info[key].update(_tmp_dict[key])

        # make sure that all the samples are sorted by their indices
        for key in self.step_info.keys():
            self.step_info[key] = dict(sorted(self.step_info[key].items(), key=lambda x: x[0]))
            # .md files remain their original names
            if key.endswith('.md'):
                np.savetxt(os.path.join(self.result_path, key), list(self.step_info[key].items()), fmt="%s")
            # normal .txt files have the prefix 'idx2' attached at the beginning of their names
            else:
                np.savetxt(os.path.join(self.result_path, f'idx2{key}'), list(self.step_info[key].items()), fmt="%s")

        # --- Gather all the save-during-testing files & Generate their path files --- #
        # generate the data path files
        for file_name in os.listdir(self.result_path):
            # only consider folders
            if not os.path.isdir(os.path.join(self.result_path, file_name)):
                continue
            # only consider the folders not named as 'figures' and 'rank_tmp'
            if file_name.startswith('rank') or file_name == 'figures' or '=' in file_name:
                continue
            idx2path = {}
            for data_file in search_file_in_subfolder(
                    os.path.join(self.result_path, file_name), tgt_match_fn=lambda x: len(x.split('.')) > 1):
                data_index = '.'.join(os.path.basename(data_file).split('.')[:-1])

                # add new file into the Dict
                if data_index not in idx2path.keys():
                    idx2path[data_index] = data_file
                # for multiple files with the same index (probably because of the non-reproducible resuming issue)
                else:
                    # same the latest file
                    if get_file_birthtime(data_file) > get_file_birthtime(idx2path[data_index]):
                        while os.path.exists(idx2path[data_index]):
                            os.remove(idx2path[data_index])
                        idx2path[data_index] = data_file

            idx2path = list(sorted(idx2path.items(), key=lambda x: x[0]))
            np.savetxt(os.path.join(self.result_path, f'idx2{file_name}'), idx2path, fmt="%s")

        # --- Group-level Evaluation Report Production --- #
        result_path = os.path.join(self.result_path, "overall_results.md")
        result_string = ""

        # The overall evaluation performance
        result_string += "# Overall Evaluation (mean ± std):\n"
        content_dict = dict()
        # loop each metric and record the overall model performance
        for metric, result_dict in self.step_info.items():
            result_list = list(result_dict.values())
            # only average the numerical results
            if not isinstance(result_list[0], (int, float)):
                continue

            content_dict[metric] = f"{np.mean(result_list):.4f} ± {np.std(result_list):.4f}"
        result_string += get_list_strings(content_dict=content_dict)

        # record the group-level model performance
        if group_meta_info is not None:
            for meta_name, group_dict in group_meta_info.items():
                result_string += f"# {meta_name}-wise Evaluation:\n" \
                                 f"(***bold&italic*** numbers represent the maximal ones in all groups while " \
                                 f"**bold** numbers represent the minimal ones.)\n\n"
                table_headers, table_contents = [meta_name], dict()
                # loop each group and calculate the group-specific performance
                for group_name, group_list in group_dict.items():
                    # loop each metric
                    for metric, result_dict in self.step_info.items():
                        result_list = [result_dict[index] for index in group_list if index in result_dict.keys()]
                        # skip the non-numerical results
                        if len(result_list) > 0 and not isinstance(result_list[0], (int, float)):
                            continue
                        # average the numerical results and record them
                        if len(result_list) != 0:
                            # create group item lazily
                            if group_name not in table_contents.keys():
                                table_contents[group_name] = []

                            if metric not in table_headers:
                                table_headers.append(metric)
                            table_contents[group_name].append(np.mean(result_list))

                # get the max and min group value for each numerical metric
                for i in range(len(table_headers) - 1):
                    metric_value_list = [value[i] for value in table_contents.values()]
                    max_value, min_value = max(metric_value_list), min(metric_value_list)
                    # loop each group
                    for group in table_contents.keys():
                        # turn the max number into a bold&italic string
                        if table_contents[group][i] == max_value:
                            table_contents[group][i] = f"***{table_contents[group][i]:.4f}***"
                        # turn the min number into a bold string
                        elif table_contents[group][i] == min_value:
                            table_contents[group][i] = f"**{table_contents[group][i]:.4f}**"
                        # turn other numbers into pure strings
                        else:
                            table_contents[group][i] = f"{table_contents[group][i]:.4f}"

                # attach the list of the current group into the result string
                result_string += get_table_strings(contents=list(table_contents.values()),
                                                   first_col=list(table_contents.keys()),
                                                   headers=table_headers)
        np.savetxt(result_path, [result_string], fmt="%s")

        # --- Top-N Bad Cases Presentation --- #
        # only present topn bad cases if instance_reports.md is given
        if 'instance_reports.md' in self.step_info.keys():
            # loop each tri-tuple
            for metric, mode, num in self.bad_cases_selection:
                result_path = os.path.join(self.result_path, f"top{num}_{mode}_{metric}.md")

                if metric in self.step_info.keys():
                    # get the indices of the topn samples
                    selected_samples = sorted(self.step_info[metric].items(), key=lambda x: x[1],
                                              reverse=True if mode.lower() == 'max' else False)[:num]
                    selected_samples = [s[0] for s in selected_samples]

                    # make the .md string for all the top-n bad samples
                    sample_reports = ""
                    for s_index in selected_samples:
                        sample_reports += f"**{s_index}**" + self.step_info['instance_reports.md'][s_index]
                    np.savetxt(result_path, [sample_reports], fmt="%s")

        # --- Histograms Plotting --- #
        # remove the old figures if have
        if os.path.exists(os.path.join(self.result_path, 'figures')):
            shutil.rmtree(os.path.join(self.result_path, 'figures'))

        # loop each metric and plot the histogram figure
        for metric, result_dict in self.step_info.items():
            result_list = list(result_dict.values())
            # only consider the numerical results
            if not isinstance(result_list[0], (int, float)):
                continue

            self.enqueue(
                dict(
                    materials=copy.deepcopy({metric: result_list}), plot_type='hist'
                )
            )

        if not self.empty_queue():
            for i in range(60):
                if not self.empty_queue():
                    self.logger.info("The snapshooter is still snapshotting. Waiting for 10 seconds.")
                    time.sleep(10)

    def state_dict(self):
        """
            Returns the state of the TestMonitor instance. This is useful for checkpointing or logging purposes.

            Returns:
            - A dictionary that includes:
                * step_info: A dictionary storing information about each step.
                * finished_group_num: The number of groups that have finished processing.
        """
        return dict(
            step_info=self.step_info,
            finished_group_num=self.finished_group_num
        )
