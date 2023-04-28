"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import argparse
import os
import sys
import warnings
from contextlib import contextmanager

from packaging.version import parse as V

import torch
import numpy as np
import random
import yaml
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.cuda.amp import autocast
from typing import Dict, Any, List

from speechain.monitor import Monitor, TrainValidMonitor, TestMonitor
from speechain.iterator.abs import Iterator
from speechain.model.abs import Model
from speechain.optim_sche.abs import OptimScheduler

from speechain.utilbox.log_util import logger_stdout_file, model_summary
from speechain.utilbox.import_util import import_class, get_idle_port, parse_path_args, get_idle_gpu
from speechain.utilbox.type_util import str2bool, str2list, str2dict, str2none
from speechain.utilbox.yaml_util import load_yaml


class Runner(object):
    """
    Runner is the entrance of our toolkit. This static class is made up of several static functions. The
    whole pipeline is done by all static functions step by step. The reason why the functions are all static is to
    prevent Runner from becoming the God class after Inheritance.
    If you are interested in this topic, please refer to https://wiki.c2.com/?GodClass for more details.

    In this class, we provide an overridable interface add_parse() that enables users to add more arguments they
    would like their runners to have.

    Basically, we don't recommend users to override the other functions in this class for robustness.
    However, in case that the existing functions cannot meet your research requirements, you can override them in your
    own runners to fit your specific needs. If it happens, we would appreciate it a lot if you could open an issue
    and let us know.

    Wish you have a happy usage journey in this toolkit ^_^!
    """

    @classmethod
    def add_parse(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        The interface where users can add their own arguments.

        Args:
            parser: argparse.ArgumentParser
                The name space where you want to add your arguments.

        Returns:
            parser: argparse.ArgumentParser
                The name space containing your arguments.

        """
        return parser

    @classmethod
    def parse(cls):
        """
        The static function that outputs all the default arguments for the runner.

        Returns:
            a Dict containing the key-value pairs of all arguments

        """
        parser = argparse.ArgumentParser()

        # All-in-one configuration setting
        parser.add_argument(
            "--config",
            type=str,
            # default=None,
            default="recipes/asr/libritts_librispeech/train-960/exp_cfg/960-bpe5k_transformer-wide_ctc_perturb.yaml",
            help="The path of the all-in-one experiment configuration file. You can write all the arguments in this "
                 "all-in-one file instead of giving them to `runner.py` by command lines."
        )

        # Experimental environment
        group = parser.add_argument_group("Group 1: Calculation and System Backend")
        group.add_argument(
            '--seed',
            type=int,
            default=0,
            help="Initial random seed for the experiment. (default: 0)"
        )
        group.add_argument(
            '--cudnn_enabled',
            type=str2bool,
            default=True,
            help="Whether to activate torch.backends.cudnn. (default: True)"
        )
        group.add_argument(
            '--cudnn_benchmark',
            type=str2bool,
            default=False,
            help="Whether to activate torch.backends.cudnn.benchmark. "
                 "When True, the process of model training will be speed up and the model performance may improve "
                 "somewhat. But your results will become less reproducible. (default: False)"
        )
        group.add_argument(
            '--cudnn_deterministic',
            type=str2bool,
            default=True,
            help="Whether to activate torch.backends.cudnn.deterministic. "
                 "This will improve the reproducibility of your experiments. (default: True)"
        )
        group.add_argument(
            '--train_num_workers',
            type=int,
            default=1,
            help="The number of worker processes in the `torch.utils.data.DataLoader` of each epoch. "
                 "If you have complicated logic of data loading and data augmentation in the memory before passing the "
                 "data to the model (e.g., speech speed perturbation, environmental noise addition, ...), raising this "
                 "argument may improve the speed of data loading and pre-augmentation. But the choice of the argument "
                 "value should be within your machine capability (i.e., the number of CPU cores). "
                 "If you want to debug your programs, we recommend you to set this argument to 0. (default: 1)"
        )
        group.add_argument(
            '--valid_num_workers',
            type=int,
            default=1,
            help="The number of worker processes in the `torch.utils.data.DataLoader` of each epoch. "
                 "If you have complicated logic of data loading and data augmentation in the memory before passing the "
                 "data to the model (e.g., speech speed perturbation, environmental noise addition, ...), raising this "
                 "argument may improve the speed of data loading and pre-augmentation. But the choice of the argument "
                 "value should be within your machine capability (i.e., the number of CPU cores). "
                 "If you want to debug your programs, we recommend you to set this argument to 0. (default: 1)"
        )
        group.add_argument(
            '--test_num_workers',
            type=int,
            default=1,
            help="The number of worker processes in the `torch.utils.data.DataLoader` of each epoch. "
                 "If you have complicated logic of data loading and data augmentation in the memory before passing the "
                 "data to the model (e.g., speech speed perturbation, environmental noise addition, ...), raising this "
                 "argument may improve the speed of data loading and pre-augmentation. But the choice of the argument "
                 "value should be within your machine capability (i.e., the number of CPU cores). "
                 "If you want to debug your programs, we recommend you to set this argument to 0. (default: 1)"
        )
        group.add_argument(
            '--pin_memory',
            type=str2bool,
            default=False,
            help="Whether to activate `pin_memory` for the Dataloader of each epoch. "
                 "If True, the pinned memory in the dataloaders will be activated and the data loading will be further "
                 "speed up. "
                 "pin_memory=True is often used together with non_blocking=True. Note that this combination requires a "
                 "large amount of memory and CPU cores. (default: False)"
        )
        group.add_argument(
            '--non_blocking',
            type=str2bool,
            default=False,
            help="Whether to activate `non_blocking` when transferring data from the memory to GPUs. "
                 "If True, the process of model training will be speed up. "
                 "non_blocking=True is often used together with pin_memory=True. Note that this combination requires a "
                 "large amount of memory and CPU cores. (default: False)"
        )

        # gradient descent related
        group = parser.add_argument_group("Group 2: Gradient Calculation and Back-Propagation")
        group.add_argument(
            '--use_amp',
            type=str2bool,
            default=True,
            help="Whether activate AMP (Automatic Mixed Precision) during the back-propagation. "
                 "If True, the GPU consumption of your model will be smaller so that you can include more data "
                 "instances in a single batch. (default: True)"
        )
        group.add_argument(
            '--grad_clip',
            type=float,
            default=5.0,
            help="Gradient clipping threshold during the back-propagation. (default: 5.0)"
        )
        group.add_argument(
            '--grad_norm_type',
            type=float,
            default=2.0,
            help="Normalization type used when clipping the gradients. (default: 2.0)"
        )
        group.add_argument(
            '--accum_grad',
            type=int,
            default=1,
            help="The number of gradient accumulation steps. "
                 "To mimic the gradients calculated by large batches with only a small amount of GPUs, please raise "
                 "this argument. "
                 "The virtual batch size will become (accum_grad * the actual batch size). "
                 "Note that the model trained by accum_grad is not identical to the one actually trained by large "
                 "batches because of the different randomness in each training step and the existence of BatchNorm. "
                 "(default: 1)"
        )
        group.add_argument(
            '--ft_factor',
            type=float,
            default=1.0,
            help="The finetuing factor used to scale down learning rates during the parameter optimization. "
                 "If `ft_factor` is smaller than 1.0, the learning rates will be proportionally decreased without "
                 "changing its scheduling strategy. Usually, ft_factor could be set from 0.1 to 0.5 depending on your "
                 "finetuning scenarios. (default: 1.0)"
        )

        # multi-GPU distributed training
        group = parser.add_argument_group("Group 3: Multi-GPU Distribution")
        group.add_argument(
            "--dist_backend",
            default="nccl",
            type=str,
            help="Communication backend for multi-GPU distribution. "
                 "If you are using NVIDIA GPUs, we recommend you set this argument to 'nccl'. (default: nccl)",
        )
        group.add_argument(
            "--dist_url",
            type=str,
            default="tcp://127.0.0.1",
            help="Communication URL for multi-GPU distribution. "
                 "The default value is 'tcp://127.0.0.1' for single-node distributed training and an idle port will be "
                 "automatically selected. "
                 "The port number cannot be set manually, which means that the argument 'tcp://127.0.0.1:xxxxx' will "
                 "have the same effect with 'tcp://127.0.0.1'. "
                 "If you want to train your model on multiple nodes, please set dist_url='env://' "
                 "(Note: multi-node model distribution is still in beta). "
                 "In this case, env values of 'MASTER_PORT', 'MASTER_ADDR', 'WORLD_SIZE', and 'RANK' are referred in "
                 "the command line.",
        )
        group.add_argument(
            "--world_size",
            default=1,
            type=int,
            help="The number of nodes for model distribution. "
                 "This argument is fixed to 1. Currently, we don't recommend you to modify its value."
                 "If you want to conduct multi-node model distribution, please give `world_size` by `WORLD_SIZE=XXX` "
                 "in your terminal (Note: multi-node model distribution is still in beta)."
        )
        group.add_argument(
            '--rank',
            default=0,
            type=int,
            help="The global rank of the current node for model distribution. "
                 "This argument is fixed to 0. Currently, we don't recommend you to modify its value."
                 "If you want to conduct multi-node model distribution, please give `rank` by `RANK=XXX` in your "
                 "terminal (Note: multi-node model distribution is still in beta)."
        )
        group.add_argument(
            '--ngpu',
            type=int,
            default=1,
            help="The number of GPUs used to run your experiment. "
                 "If ngpu is larger than 1, multi-GPU model distribution will be activated. (default: 1)"
        )
        group.add_argument(
            '--gpus',
            type=str2none,
            default=None,
            help="This argument specifies the GPUs used to run your experiment. "
                 "If you want to specify multiple GPUs, please give this argument in the form of 'x,x,x' "
                 "where different GPUs are separated by a comma (please don't end this argument with ','). "
                 "Of course, you could also specify your target GPUs by `CUDA_VISIBLE_DEVICES` in the terminal."
                 "If this argument is not given, the framework will automatically select `ngpu` idle GPUs. "
        )
        group.add_argument(
            '--same_proc_seed',
            type=str2bool,
            default=False,
            help="Whether to set the same initial random seed for all the GPU processes in DDP mode. "
                 "The different random seeds can prevent model distribution from the process homogeneity, "
                 "e.g., different GPU processes may have the same on-the-fly data augmentation strategy "
                 "(noise addition, SpecAugment, ...) if they have the same initial random seed. "
                 "Note: please set this argument to True if you want to use random data selection for your dataloaders "
                 "in the DDP mode. (default: False)"
        )

        # Training monitoring
        group = parser.add_argument_group("Group 4: Model Training")
        group.add_argument(
            '--train_result_path',
            type=str,
            default=None,
            help="Where to place all the experiment folder that contains all the result files. "
                 "If not given, `train_result_path` wil be automatically initialized by your input `config`. "
                 "For example, if your input `config` is "
                 "{SPEECHAIN_ROOT}/recipes/asr/librispeech/train-960/exp_cfg/XXXXX.yaml, your `train_result_path` "
                 "will be automatically initialized to `{SPEECHAIN_ROOT}/recipes/asr/librispeech/train-960/exp/`."
                 "(default: None)")
        group.add_argument(
            '--attach_config_folder_to_path',
            type=str2bool,
            default=True,
            help="Whether to attach an additional folder named by your input `--config` at the end of your input "
                 "`train_result_path`. (default: True)"
        )
        group.add_argument(
            '--train',
            type=str2bool,
            default=False,
            help="Whether to go through the model training branch. (default: False)"
        )
        group.add_argument(
            '--dry_run',
            type=str2bool,
            default=False,
            help="Whether to turn on the dry-running mode. "
                 "In this mode, only the data loading will be done to see its speed and robustness. "
                 "Model calculation and parameter optimization will be skipped. (default: False)"
        )
        group.add_argument(
            '--no_optim',
            type=str2bool,
            default=False,
            help="Whether to turn on the no-optimization mode. "
                 "In this mode, only the data loading and model calculation will be done to see their speed, "
                 "robustness, and memory consumption. (default: False) "
                 "(Note: 'dry_run' has the higher priority than 'no_optim'. It means that the model calculation will "
                 "be skipped if you give both '--dry_run True' and '--no_optim True'.) "
        )
        group.add_argument(
            '--resume',
            type=str2bool,
            default=False,
            help="Whether to resume your model training or testing experiment from the checkpoints. "
                 "If True, there must be .pth checkpoint files of your existing experiment in `train_result_path` or "
                 "`test_result_path`. This argument is shared by the training and testing branches. (default: False)"
        )
        group.add_argument(
            '--start_epoch',
            type=int,
            default=1,
            help="The starting epoch of your experiments. This argument will be automatically initialized by your "
                 "checkpoint files if `--resume` is given. (default: 1)"
        )
        group.add_argument(
            '--num_epochs',
            type=int,
            default=1000,
            help="The maximum number of training epochs of your experiments. (default: 1000)"
        )
        group.add_argument(
            '--valid_per_epochs',
            type=int,
            default=1,
            help="The interval of going through the validation phase during training. "
                 "If not given, validation will be done right after parameter optimization in each epoch. (default: 1)"
        )
        group.add_argument(
            '--report_per_steps',
            type=int,
            default=0,
            help="The interval of reporting step information logs during model training or testing. "
                 "Positive integers mean the absolute reporting intervals that a step report will be made after each "
                 "'report_per_steps' steps; "
                 "Negative integers mean the relative reporting intervals that there will be -'report_per_steps' "
                 "reports in each epoch. "
                 "If not given, there will be default 10 reports in each epoch. "
        )
        group.add_argument(
            '--best_model_selection',
            type=str2list,
            default=None,
            help="The ways of selecting the best models. This argument should be given as a list of quad-tuples, i.e., "
                 "('metric_group', 'metric_name', 'metric_mode', 'model_number'). "
                 "'metric_group' can be either 'train' or 'valid' which indicates the group the metric belongs to; "
                 "'metric_name' is the name of the metric you select; "
                 "'metric_mode' can be either 'min' or 'max' which indicates how to select the models by this metric; "
                 "'model_number' indicates how many best models will be saved by this metric. "
                 "Note: the metric of the first tuple in the list will be used to do early-stopping for model training."
                 "(default: None)"
        )
        group.add_argument(
            '--early_stopping_patience',
            type=int,
            default=None,
            help="The maximum number of epochs when the model doesn't improve its performance before stopping the "
                 "model training. If not given, early-stopping will not be adapted. (default: None)"
        )
        group.add_argument(
            '--early_stopping_threshold',
            type=float,
            default=0.001,
            help="The threshold to refresh the early-stopping status in the monitor during model training. "
                 "Positive float numbers in (0.0, 1.0) mean the relative threshold over the current best performance. "
                 "Negative float numbers main the absolute threshold over the current best performance. "
                 "early_stopping_threshold=0 means no early-stopping threshold is applied to the current best "
                 "performance when deciding whether to refresh the status. (default: 0.005)"
        )
        group.add_argument(
            '--last_model_number',
            type=int,
            default=1,
            help="The number of models saved for the last several epochs. "
                 "This argument cannot be lower than 1 otherwise the training will not be able to resume. "
                 "(default: 1)"
        )

        # Training Snapshotting
        group = parser.add_argument_group("Group 5: Real-time Model Visualization Snapshotting")
        group.add_argument(
            '--monitor_snapshot_conf',
            type=str2dict,
            default=dict(),
            help="The configuration given to `matploblib.plot()` in `{SPEECHAIN_ROOT/speechain/snapshooter.py}` to "
                 "plot curve figures for real-time model visualization during model training. "
                 "This argument should be given in the form of a Dict. (default: an empty Dict)"
        )
        group.add_argument(
            '--visual_snapshot_number',
            type=int,
            default=0,
            help="The number of the validation data instances used to make snapshots made during model visualization. "
                 "This argument should be smaller than the number of your validation data instances. "
                 "(default: 0)"
        )
        group.add_argument(
            '--visual_snapshot_interval',
            type=int,
            default=5,
            help="The snapshotting interval of model visualization during model training. "
                 "This argument should be a positive integer which means that model visualization will be done once "
                 "in every `visual_snapshot_interval` epochs. (default: 5)"
        )

        # Testing
        group = parser.add_argument_group("Group 6: Model Testing")
        group.add_argument(
            '--test_result_path',
            type=str,
            default=None,
            help="Where to place all the result files generated during model testing. "
                 "If not given, `test_result_path` wil be automatically initialized by your input `train_result_path` "
                 "and `test_model`. For example, if your `train_result_path` is "
                 "`{SPEECHAIN_ROOT}/recipes/asr/librispeech/train-960/exp`, and `test_model` is `MMMMM`, "
                 "then your `test_result_path` will be automatically initialized to "
                 "`{SPEECHAIN_ROOT}/recipes/asr/librispeech/train-960/exp/XXXXX/MMMMM/` where 'XXXXX' is the name of "
                 "your configuration file given by `--config`."
        )
        group.add_argument(
            '--test',
            type=str2bool,
            default=False,
            help="Whether to go through the model testing branch. (default: False)"
        )
        group.add_argument(
            '--test_model',
            type=str2list,
            default=None,
            help="The names of the model you want to evaluate during model testing. "
                 "If given, `{train_result_path}/XXXXX/model/{test_model}.pth` will be used to initialize the parameters "
                 "of the Model object. If you only want to evaluate multiple models in one job, please give the "
                 "strings of their names in a List. (default: None)"
        )
        group.add_argument(
            '--attach_model_folder_when_test',
            type=str2bool,
            default=True,
            help="Whether to attach an additional sub-folder named by your input `--test_model` in the testing "
                 "result folder. (default: True)"
        )
        group.add_argument(
            '--bad_cases_selection',
            type=str2list,
            default=None,
            help="The selection methods of the top-N bad cases during model testing. "
                 "This argument should be given as a list of tri-tuples "
                 "('selection_metric', 'selection_mode', 'case_number'). "
                 "For example, ('wer', 'max', 50) means 50 testing waveforms with the largest WER will be selected. "
                 "Multiple tuples can be given to present different sets of top-n bad cases. (default: None)"
        )

        # Experiment configuration
        group = parser.add_argument_group("Group 7: Experiment .yaml Configuration File")
        group.add_argument(
            '--data_cfg',
            type=str2dict,
            default=None,
            help="The path of the configuration file for data loading and batching. "
                 "This argument is required for both model training and testing."
        )
        group.add_argument(
            '--train_cfg',
            type=str2dict,
            default=None,
            help="The path of the configuration file for model construction and parameter optimization. "
                 "This argument is required for both model training (both 'model' and 'optim_sche' need to be given) "
                 "and testing (only 'model' needs to be given)."
        )
        group.add_argument(
            '--infer_cfg',
            type=str2dict,
            default=None,
            help="The configuration file for model inference during model testing. "
                 "This argument is required for model testing."
                 "For more details about how to give infer_cfg, please refer to the handbook.md. (default: None)"
        )

        # Add customized arguments if needed
        parser = cls.add_parse(parser)
        return parser.parse_args()

    @classmethod
    def build_iterators(cls, data_cfg: Dict[str, Dict], args: argparse.Namespace) \
            -> Dict[str, Dict[str, Iterator] or Iterator]:
        """
        This static function builds all iterators used in the experiment. The configuration of iterators is given in
        your specified 'data_cfg'.

        The iterators are returned as a dictionary where the first-level keys indicate different iterator groups:
        'train', 'valid', and 'test'. The second-level keys in each group indicates the iterators belonging to the
        group. In the value of each second-level key, there are two third-level keys: 'type' and 'conf'. 'type'
        indicates the iterator type and 'conf' indicates the iterator configuration. For more details, please refer
        to ./speechain/iterator/README.md

        Args:
            data_cfg: Dict
                The dictionary containing all the information to initialize the iterators
            args: argparse.Namespace
                The arguments of the runner in this experiment.

        Returns:
            The dictionary of the iterators of all groups (train, valid, test).

        """

        def recur_iterator_init(_data_cfg: Dict, _dset: str):
            leaf_flag = len(_data_cfg) == 2 and ('type' in _data_cfg.keys() and 'conf' in _data_cfg.keys())
            if leaf_flag:
                iterator_class = import_class('speechain.iterator.' + _data_cfg["type"])
                return iterator_class(seed=args.seed,
                                      ngpu=args.ngpu,
                                      num_workers=getattr(args, f'{_dset}_num_workers'),
                                      pin_memory=args.pin_memory,
                                      distributed=args.distributed,
                                      **_data_cfg["conf"])
            else:
                return {key: recur_iterator_init(value, _dset) for key, value in _data_cfg.items()}

        def recur_batch_num_init(_iterators: Dict or Iterator):
            leaf_flag = isinstance(_iterators, Iterator)
            if leaf_flag:
                return len(_iterators)
            else:
                sub_leaf_flag = sum([isinstance(value, Iterator) for value in _iterators.values()]) == len(_iterators)
                if sub_leaf_flag:
                    return [len(value) for value in _iterators.values()]
                else:
                    return {key: recur_batch_num_init(value) for key, value in _iterators.items()}

        def flatten_dict_to_list(_input: Dict or int or List[int]):
            leaf_flag = isinstance(_input, (int, List))
            if leaf_flag:
                return [_input] if isinstance(_input, int) else _input
            else:
                _output = []
                for value in _input.values():
                    _output += flatten_dict_to_list(value)
                return _output

        # get the target groups of the current experiment
        if args.train:
            assert 'train' in data_cfg.keys() and 'valid' in data_cfg.keys(), \
                "If args.train is set to True, please give 'train' and 'valid' as first-level keys of data_cfg."
            dset_keys = ['train', 'valid']
        elif args.test:
            assert 'test' in data_cfg.keys(), \
                "If args.test is set to True, please give 'test' as first-level keys of data_cfg."
            dset_keys = ['test']
        else:
            raise RuntimeError("Please set either args.train or args.test to True!")

        # recursively initialize all the iterators in the Dict
        mode = 'train' if args.train else 'test'
        iterators = {dset: recur_iterator_init(data_cfg[dset], dset) for dset in dset_keys}
        batch_nums = recur_batch_num_init(iterators[mode])

        # set the relative reporting interval during training or testing
        if args.report_per_steps <= 0:
            _reports_per_epoch = 10 if args.report_per_steps == 0 else int(-args.report_per_steps)
            args.report_per_steps = min(flatten_dict_to_list(batch_nums)) // _reports_per_epoch
        # check the absolute reporting interval during training and testing
        else:
            assert int(args.report_per_steps) <= min(flatten_dict_to_list(batch_nums)), \
                f"If args.report_per_steps is given as a positive integer, " \
                f"it should be smaller than the minimal {mode} batch number ({min(batch_nums)}). " \
                f"But got report_per_steps={int(args.report_per_steps)}!"

            # in case that report_per_steps is given as a float number
            args.report_per_steps = int(args.report_per_steps)

        return iterators

    @classmethod
    def build_model(cls, model_cfg: Dict[str, Any], args: argparse.Namespace, device: torch.device) -> Model:
        """
        This static function builds the model used in the experiment. The configuration of the model is given in
        the value of the 'model' key in your specified 'model_cfg'.

        Args:
            model_cfg: Dict
                Model Configuration
            args:
            device:

        Returns:
            The target Model object initialized by your given configuration

        """
        assert "model_type" in model_cfg.keys(), "Please specify the model_type!"
        assert "module_conf" in model_cfg.keys(), "Please specify the module_conf!"
        if 'criterion_conf' not in model_cfg.keys():
            model_cfg['criterion_conf'] = None

        model_class = import_class('speechain.model.' + model_cfg['model_type'])
        return model_class(model_conf=model_cfg['model_conf'] if "model_conf" in model_cfg.keys() else dict(),
                           module_conf=model_cfg['module_conf'],
                           criterion_conf=model_cfg['criterion_conf'],
                           device=device,
                           result_path=args.train_result_path,
                           non_blocking=args.non_blocking,
                           distributed=args.distributed).cuda(device=device)

    @classmethod
    def build_optim_sches(cls,
                          model: Model,
                          optim_sche_cfg: Dict[str, Any],
                          args: argparse.Namespace) -> Dict[str, OptimScheduler] or OptimScheduler:
        """
        This static function builds the OptimSchedulers used in the pipeline. The configuration of the
        OptimSchedulers is given in the value of 'optim_sches' key in your specified 'train_cfg'.

        This function must be done after DDP wrapping because we need to make sure that the model parameters received
        by the optimizer in each process are identical. With the identical model parameters, it's safe to consider that
        the optimizer parameters are also identical.

        Args:
            model: Model
                The initialized model.
            optim_sche_cfg: Dict
                OptimScheduler Configuration
            args: argparse.Namespace
                The input arguments. Used to pass accum_grad, grad_clip, and grad_norm_type to your optimedulers.

        Returns:
            The Dict of the initialized OptimSchedulers.

        """
        # single-optimizer scenario
        if len(optim_sche_cfg) == 2 and ('type' in optim_sche_cfg.keys() and 'conf' in optim_sche_cfg.keys()):
            optim_sche_cfg = dict(main=optim_sche_cfg)

        optim_sches = dict()
        for name, optim_sche in optim_sche_cfg.items():
            optim_sche_class = import_class('speechain.optim_sche.' + optim_sche['type'])
            optim_sches[name] = optim_sche_class(model=model,
                                                 distributed=args.distributed,
                                                 use_amp=args.use_amp,
                                                 accum_grad=args.accum_grad,
                                                 ft_factor=args.ft_factor,
                                                 grad_clip=args.grad_clip,
                                                 grad_norm_type=args.grad_norm_type,
                                                 **optim_sche['conf'])

        # multi-optimizer scenario
        if len(optim_sches) > 1:
            # adjust whether there are parameter overlapping among updated_modules of all the OptimSchedulers
            is_all_para = [o.updated_modules is None for o in optim_sches.values()]
            # updated_modules of all the OptimSchedulers cannot be None at the same time
            if sum(is_all_para) == len(is_all_para):
                raise RuntimeError
            else:
                # collect the updated_modules of all the OptimScheduler
                para_list = [o.updated_modules for o in optim_sches.values()]
                # adjust whether there are redundant keys
                para_set = set(para_list)
                # there is parameter overlapping if there are redundant keys
                if len(para_set) != len(para_list):
                    raise RuntimeError

        # resuming from an existing checkpoint
        if args.resume:
            try:
                checkpoint = torch.load(
                    os.path.join(args.train_result_path, "checkpoint.pth"), map_location=model.device)
                for name in optim_sches.keys():
                    optim_sches[name].load_state_dict(checkpoint['optim_sches'][name])
            except FileNotFoundError:
                print(f"No checkpoint is found in {args.train_result_path}. "
                      f"The training process will start from scratch.")

        return optim_sches

    @classmethod
    def resume(cls,
               args: argparse.Namespace,
               model: Model,
               monitor: TrainValidMonitor) -> int:
        """
        load the model parameters to the current process. This operation is necessary in our toolkit because we need to
        make sure that the models in all the processes have the same buffer and parameter tensors.

        Args:
            args: argparse.Namespace
                The input arguments.
            model: Model
                The model to be trained.
            monitor: TrainValidMonitor
                The train-valid monitor used to monitor the training phase

        Returns:
            The number of the starting epoch. If the training resumes from an existing checkpoint, then the starting
            epoch will be loaded from the checkpoint; otherwise, 1 will be returned.

        """
        # start the training from the existing checkpoint
        if args.resume:
            # load the existing checkpoint
            checkpoint = torch.load(
                os.path.join(args.train_result_path, "checkpoint.pth"), map_location=model.device)
            # load the latest training epoch
            start_epoch = checkpoint['start_epoch']
            # for compatibility with old versions
            if 'latest_model' in checkpoint.keys():
                model.load_state_dict(checkpoint['latest_model'])
            else:
                model.load_state_dict(
                    torch.load(
                        os.path.join(args.train_result_path, "models", "latest.pth"), map_location=model.device)
                )

            # loading the monitor
            if monitor is not None:
                # for compatibility with old versions
                if 'monitor' not in checkpoint.keys():
                    monitor.load_state_dict(dict(
                        train_monitor=checkpoint['train_monitor'],
                        valid_monitor=checkpoint['valid_monitor']
                    ))
                else:
                    monitor.load_state_dict(checkpoint['monitor'])
                # info logging
                monitor.logger.info(f"The training process resumes from the epoch no.{start_epoch}.")

        # start the training from scratch
        else:
            start_epoch = 1

        return start_epoch


    @classmethod
    def dict_transform(cls, src_dict, transform_func):
        """

        Args:
            src_dict:
            transform_func:

        Returns:

        """
        # Multi-dataloader
        if isinstance(src_dict, Dict):
            return {key: transform_func(value) for key, value in src_dict.items()}
        # Single-dataloader
        else:
            return transform_func(src_dict)

    @classmethod
    def measure_time(cls, monitor: Monitor):
        @contextmanager
        def empty_context(names=None):
            yield

        if monitor is None:
            return empty_context
        else:
            return monitor.measure_time

    @classmethod
    def is_empty_batch(cls, input_batch: Dict):
        # Single-dataloader case
        if len(input_batch) == 0:
            return True
        # Multi-dataloader case
        else:
            for value in input_batch.values():
                if isinstance(value, Dict):
                    for sub_value in value.values():
                        if len(sub_value) == 0:
                            return True
                    return False

    @classmethod
    def train(cls,
              args: argparse.Namespace,
              data_cfg: Dict,
              iterators: Dict[str, Dict[str, Iterator]] or Dict[str, Iterator],
              model: Model,
              optim_sches: Dict[str, OptimScheduler] or OptimScheduler,
              logger,
              monitor: TrainValidMonitor):
        """

        Args:
            args: argparse.Namespace
                The input arguments.
            data_cfg: Dict
                The data loading configuration. Used to initialize the iterator for model visualization.
            iterators: Dict
                The dictionary that contains all the iterators for training and validation.
            model: Model
                The model to be trained.
            optim_sches: Dict
                The dictionary that contains all the OptimSchedulers used to update the model parameters.
            logger:

            monitor: TrainValidMonitor
                The wrapper class for a training monitor and a validation monitor.
                The training monitor controls the training process of the model and generates the real-time logging
                information.
                The validation monitor controls the validation process of the model and generates the real-time
                logging information.

        """
        assert args.start_epoch <= args.num_epochs, "Your given start_epoch is larger than your given num_epochs!"

        # --- checking the data lengths of all training iterators --- #
        # multiple dataloaders scenario
        if isinstance(iterators['train'], Dict):
            train_batch_nums = set([len(iterator) for iterator in iterators['train'].values()])
            min_train_batch_num = min(train_batch_nums)
            if len(train_batch_nums) != 1:
                logger.info(f"Your training iterators have different batch numbers: {train_batch_nums}. "
                            f"The actual batch number during training is set to {min_train_batch_num}!")
        # single dataloader scenario
        elif isinstance(iterators['train'], Iterator):
            min_train_batch_num = len(iterators['train'])
        else:
            raise RuntimeError("Please don't nest data_cfg['train'] more than twice!")

        # --- checking the data lengths of all validation iterators --- #
        # multiple dataloaders scenario
        if isinstance(iterators['valid'], Dict):
            valid_batch_nums = set([len(iterator) for iterator in iterators['valid'].values()])
            min_valid_batch_num = min(valid_batch_nums)
            if len(valid_batch_nums) != 1:
                logger.info(f"Your validation iterators have different batch numbers: {valid_batch_nums}. "
                            f"The actual batch number during validation is set to {min_valid_batch_num}!")
        # single dataloader scenario
        elif isinstance(iterators['valid'], Iterator):
            min_valid_batch_num = len(iterators['valid'])
        else:
            raise RuntimeError("Please don't nest data_cfg['valid'] more than twice!")

        # synchronize the batch numbers across all the distributed processes
        if args.distributed:
            _world_size = torch.distributed.get_world_size()
            # make sure that all processes have the same number of training steps
            _all_batch_num = torch.LongTensor([0 for _ in range(_world_size)]).cuda(model.device)
            torch.distributed.all_gather_into_tensor(
                _all_batch_num, torch.LongTensor([min_train_batch_num]).cuda(model.device))
            min_train_batch_num = _all_batch_num.min().item()

            # make sure that all processes have the same number of validation steps
            _all_batch_num = torch.LongTensor([0 for _ in range(_world_size)]).cuda(model.device)
            torch.distributed.all_gather_into_tensor(
                _all_batch_num, torch.LongTensor([min_valid_batch_num]).cuda(model.device))
            min_valid_batch_num = _all_batch_num.min().item()

        # --- Initialize the iterator for model visualization --- #
        if args.visual_snapshot_number > 0:
            if not args.distributed or args.rank == 0:
                _valid_keys = list(data_cfg['valid'].keys())
                if len(_valid_keys) == 2 and ('type' in _valid_keys and 'conf' in _valid_keys):
                    visual_iterator = Iterator(dataset_type=data_cfg['valid']['conf']['dataset_type'],
                                               dataset_conf=data_cfg['valid']['conf']['dataset_conf'],
                                               batches_per_epoch=args.visual_snapshot_number,
                                               shuffle=False, ngpu=1, distributed=False, is_descending=None)
                else:
                    visual_domain = _valid_keys[0]
                    logger.info("There are multiple sub-Dict in your given data_cfg['valid']. "
                                f"The one named {visual_domain} is used to initialize the visualization iterator.")
                    visual_iterator = \
                        {visual_domain: Iterator(dataset_type=data_cfg['valid'][visual_domain]['conf']['dataset_type'],
                                                 dataset_conf=data_cfg['valid'][visual_domain]['conf']['dataset_conf'],
                                                 batches_per_epoch=args.visual_snapshot_number,
                                                 shuffle=False, ngpu=1, distributed=False, is_descending=None)}
            else:
                visual_iterator = None
        else:
            visual_iterator = None

        # loop each epoch until the end
        for epoch in range(args.start_epoch, args.num_epochs + 1):
            # update the random seeds for the current epoch to keep in line with the dataloaders
            cls.set_random_seeds(args.seed + epoch)
            # start the current training epoch
            if monitor is not None:
                monitor.start_train_epoch(epoch)
            # initialize all the training dataloaders
            data_loaders = cls.dict_transform(iterators['train'], lambda x: iter(x.build_loader(epoch)))

            # --- Training Stage --- #
            model.train()
            # loop all the training batches
            for step in range(1, min_train_batch_num + 1):
                step_num = int(step + (epoch - 1) * min_train_batch_num)

                # --- data loading part --- #
                with cls.measure_time(None if monitor is None else monitor.train_monitor)("data_load_time"):
                    train_batch = cls.dict_transform(src_dict=data_loaders, transform_func=next)
                    # single-GPU case, directly skip the current step when meeting an empty batch
                    if not args.distributed:
                        # skip the empty validation batch
                        if cls.is_empty_batch(train_batch):
                            continue

                    # multi-GPU case, scatter the skip flag to all nodes
                    else:
                        skip_flag_list = torch.LongTensor(
                            [False for _ in range(torch.distributed.get_world_size())]).cuda(model.device)
                        if cls.is_empty_batch(train_batch):
                            skip_flag = torch.LongTensor([True]).cuda(model.device)
                        else:
                            skip_flag = torch.LongTensor([False]).cuda(model.device)
                        # as long as one node meets an empty batch, all nodes will simultaneously skip the current step
                        torch.distributed.all_gather_into_tensor(skip_flag_list, skip_flag)
                        if skip_flag_list.sum() >= 1:
                            continue

                # forward the batch to get the training criteria and optimize the model
                train_metrics, optim_lr = None, None
                # whether to skip the model forward part and model optimization part
                if not args.dry_run:
                    # --- model forward part --- #
                    with autocast(enabled=args.use_amp):
                        with cls.measure_time(None if monitor is None else monitor.train_monitor)("model_forward_time"):
                            losses, train_metrics = model(batch_data=train_batch, epoch=epoch)

                    # whether to skip the model optimization part
                    if not args.no_optim:
                        # --- loss backward and optimization part --- #
                        optim_lr = dict()
                        for name, optim_sche in optim_sches.items():
                            optim_sche.step(losses=losses,
                                            time_func=cls.measure_time(
                                                None if monitor is None else monitor.train_monitor
                                            ), optim_name=name, step_num=step_num, epoch_num=epoch, logger=logger)
                            optim_lr[name] = optim_sche.get_lr()

                # log the information of the current training step
                if monitor is not None:
                    monitor.train_step(step_num=step, optim_lr=optim_lr, train_metrics=train_metrics)

            # finish the current training epoch
            if monitor is not None:
                monitor.finish_train_epoch()

            # --- Validation Stage --- #
            # start the validation part of the current epoch
            if monitor is not None:
                monitor.start_valid_epoch(epoch)

            valid_flag = (epoch - 1) % args.valid_per_epochs == 0
            if valid_flag:
                # initialize all the validation dataloaders
                data_loaders = cls.dict_transform(iterators['valid'], lambda x: iter(x.build_loader(epoch)))

                # make sure that no gradient appears during validation
                model.eval()
                with torch.inference_mode():
                    # loop all validation batches
                    for step in range(min_valid_batch_num):
                        # --- data loading part --- #
                        with cls.measure_time(None if monitor is None else monitor.valid_monitor)("data_load_time"):
                            valid_batch = cls.dict_transform(src_dict=data_loaders, transform_func=next)
                            # single-GPU case, directly skip the current step when meeting an empty batch
                            if not args.distributed:
                                # skip the empty validation batch
                                if cls.is_empty_batch(valid_batch):
                                    continue
                            # multi-GPU case, scatter the skip flag to all nodes
                            else:
                                skip_flag_list = torch.LongTensor(
                                    [False for _ in range(torch.distributed.get_world_size())]).cuda(model.device)
                                if cls.is_empty_batch(valid_batch):
                                    skip_flag = torch.LongTensor([True]).cuda(model.device)
                                else:
                                    skip_flag = torch.LongTensor([False]).cuda(model.device)
                                # as long as one node meets an empty batch,
                                # all nodes will skip the current step at the same time
                                torch.distributed.all_gather_into_tensor(skip_flag_list, skip_flag)
                                if skip_flag_list.sum() >= 1:
                                    continue

                        # forward the batch to get the validation criteria
                        valid_metrics = None
                        # whether to skip the model forward part
                        if not args.dry_run:
                            # --- model forward part --- #
                            # with autocast(enabled=args.use_amp) is not used here for accurate validation
                            with cls.measure_time(
                                    None if monitor is None else monitor.valid_monitor
                            )("model_forward_time"):
                                try:
                                    valid_metrics = model(batch_data=valid_batch)
                                except Exception as e:
                                    warnings.warn(f'Rank no.{args.rank} meets error {e}! '
                                                  f'no.{step} validation step will be skipped!')
                                    if logger is not None:
                                        logger.warning(f'Rank no.{args.rank} meets error {e}! '
                                                       f'no.{step} validation step will be skipped!')
                                    continue

                        # no step log for the validation step
                        if monitor is not None:
                            monitor.valid_step(valid_metrics=valid_metrics)

            # --- Visualization Stage --- #
            if args.visual_snapshot_number > 0 and (epoch - 1) % args.visual_snapshot_interval == 0:
                # make sure that all processes go through the validation phase smoothly
                if visual_iterator is not None:
                    if not isinstance(visual_iterator, Dict):
                        visual_domain = None
                        visual_dataloader = visual_iterator.build_loader()
                        visual_indices = visual_iterator.get_batch_indices()
                    else:
                        visual_domain = list(visual_iterator.keys())[0]
                        visual_dataloader = visual_iterator[visual_domain].build_loader()
                        visual_indices = visual_iterator[visual_domain].get_batch_indices()

                    # make sure that no gradient appears during validation
                    model.eval()
                    visual_dataloader = iter(visual_dataloader)
                    with torch.inference_mode():
                        for step in range(args.visual_snapshot_number):
                            visual_sample = next(visual_dataloader)
                            if cls.is_empty_batch(visual_sample):
                                logger.info(f"The visual sample {visual_indices[step][0]} is empty, "
                                            f"so its visualization is skipped!")
                                continue
                            # feed the current sample to the model
                            monitor.valid_model_snapshot(epoch=epoch, domain=visual_domain,
                                                         sample_index=visual_indices[step][0],
                                                         used_sample=visual_sample)
                # synchronize all the GPU processes at the end of the visualization stage
                if args.distributed:
                    torch.distributed.barrier()

            # finish_valid_epoch() should be called before checkpoint saving
            finish_valid_flag = None
            if not args.distributed or args.rank == 0:
                finish_valid_flag = monitor.finish_valid_epoch(valid_flag=valid_flag, valid_per_epochs=args.valid_per_epochs)

            # store the checkpoint of the current epoch for later resuming
            if not args.distributed or args.rank == 0:
                if not args.dry_run and not args.no_optim:
                    torch.save(
                        {
                            "start_epoch": epoch + 1,
                            "latest_model": model.state_dict() if not args.distributed else model.module.state_dict(),
                            "monitor": monitor.state_dict(),
                            "optim_sches": {name: o.state_dict() for name, o in optim_sches.items()}
                        },
                        os.path.join(args.train_result_path, "checkpoint.pth")
                    )

            # early-stopping checking for single-GPU
            if not args.distributed and finish_valid_flag:
                break

            # early-stopping checking for multi-GPU
            if args.distributed:
                stop_flag = torch.BoolTensor([False]).cuda(model.device)
                flag_list = None

                if args.rank == 0:
                    if finish_valid_flag:
                        stop_flag = torch.BoolTensor([True]).cuda(model.device)
                    flag_list = [stop_flag for _ in range(torch.distributed.get_world_size())]

                torch.distributed.scatter(stop_flag, flag_list)
                if stop_flag.item():
                    break

        # check whether all the monitor queues become empty in every minute
        if not args.distributed or args.rank == 0:
            monitor.wait_empty_queues()

        # synchronize all the GPU processes at the end
        if args.distributed:
            torch.distributed.barrier()

    @classmethod
    def test(cls,
             args: argparse.Namespace,
             test_model: str,
             iterators: Dict[str, Dict[str, Iterator]],
             model: Model):
        """

        Args:
            args: argparse.Namespace
                The input arguments.
            iterators: Dict
                The dictionary that contains all the iterators for training and validation.
            test_model: Model
                The model to be trained.

        """

        # parse infer_cfg depending on different situations
        if isinstance(args.infer_cfg, str):
            infer_cfg_dict = {'.'.join(args.infer_cfg.split("/")[-1].split(".")[:-1]): load_yaml(open(args.infer_cfg))}

        elif isinstance(args.infer_cfg, List):
            infer_cfg_dict = dict()
            for cfg in args.infer_cfg:
                if isinstance(cfg, str):
                    infer_cfg_dict['.'.join(cfg.split("/")[-1].split(".")[:-1])] = load_yaml(open(cfg))
                elif isinstance(cfg, Dict):
                    cfg = dict(sorted(cfg.items(), key=lambda x: x[0]))
                    infer_cfg_dict['_'.join([f"{key}={value}" for key, value in cfg.items()])] = cfg
                else:
                    raise TypeError("If infer_cfg is given in the form of a List, "
                                    "it must be either a List[str] or a List[Dict]!")

        elif isinstance(args.infer_cfg, Dict):
            if 'shared_args' in args.infer_cfg.keys() and 'exclu_args' in args.infer_cfg.keys():
                assert isinstance(args.infer_cfg['shared_args'], Dict) and \
                       isinstance(args.infer_cfg['exclu_args'], List), \
                    "If infer_cfg is given by 'shared_args' and 'exclu_args', " \
                    "infer_cfg['shared_args'] must be a Dict and infer_cfg['exclu_args'] must be a List."
                infer_cfg_dict = dict()
                for cfg in args.infer_cfg['exclu_args']:
                    assert isinstance(cfg, Dict), ""
                    cfg.update(args.infer_cfg['shared_args'])
                    cfg = dict(sorted(cfg.items(), key=lambda x: x[0]))
                    infer_cfg_dict['_'.join([f"{key}={value}" for key, value in cfg.items()])] = cfg

            elif 'shared_args' not in args.infer_cfg.keys() and 'exclu_args' not in args.infer_cfg.keys():
                if len(args.infer_cfg) == 0:
                    infer_cfg_dict = dict(default_inference=dict())
                else:
                    args.infer_cfg = dict(sorted(args.infer_cfg.items(), key=lambda x: x[0]))
                    infer_cfg_dict = \
                        {'_'.join([f"{key}={value}" for key, value in args.infer_cfg.items()]): args.infer_cfg}

            else:
                raise RuntimeError("If infer_cfg is given in the form of a Dict, "
                                   "'shared_args' and 'exclu_args' must be or not be in the key list at the same time!")

        elif args.infer_cfg is None:
            infer_cfg_dict = dict(default_inference=dict())

        else:
            raise TypeError("infer_cfg must be given in the form of a string, a List, or a Dict!")

        # loop each test configuration
        for infer_cfg_name, infer_cfg in infer_cfg_dict.items():
            # configuration-specific result path
            test_result_path = os.path.join(
                args.train_result_path if args.test_result_path is None else args.test_result_path, infer_cfg_name)
            os.makedirs(test_result_path, exist_ok=True)

            # load the existing testing configuration for resuming
            infer_cfg_path = os.path.join(test_result_path, "infer_cfg.yaml")
            if args.resume and os.path.exists(infer_cfg_path):
                infer_cfg = load_yaml(open(infer_cfg_path))

            # save the testing configuration file to infer_cfg_path
            if not args.distributed or args.rank == 0:
                if len(infer_cfg) > 0:
                    with open(infer_cfg_path, 'w', encoding="utf-8") as f:
                        yaml.dump(infer_cfg, f, sort_keys=False)

            # unlike training and validation, the testing iterators are looped one by one
            for name, iterator in iterators['test'].items():
                # replace the slash with a percent symbol
                name = name.replace('/', '%')
                # add the identity symbol to the path for multi-GPU testing
                if args.attach_model_folder_when_test:
                    test_dset_path = os.path.join(test_result_path, test_model, name)
                else:
                    test_dset_path = os.path.join(test_result_path, name)
                test_rank_path = os.path.join(test_dset_path, f'rank{args.rank}_tmp')
                logger = logger_stdout_file(test_rank_path, file_name='test')

                # initialize top-n bad case presentation
                if args.bad_cases_selection is None:
                    if model.bad_cases_selection is not None:
                        args.bad_cases_selection = model.bad_cases_selection
                    else:
                        logger.info("There is no configuration of topN bad case selection in either your input "
                                    "arguments or default values of your selected model. "
                                    "So there will not be any reports about topN bad cases.")
                # the main testing process
                if args.bad_cases_selection is not None:
                    logger.info(
                        f"The configuration of topN bad case selection in the current testing process is {args.bad_cases_selection}.")

                # initialize the testing monitor
                monitor = TestMonitor(logger=logger, args=args, result_path=test_dset_path)

                # check the resuming status
                if args.resume:
                    # loading the existed checkpoint
                    try:
                        test_checkpoint = torch.load(os.path.join(test_rank_path, 'checkpoint.pth'))
                        monitor.load_state_dict(test_checkpoint['monitor'])
                        start_step = test_checkpoint['start_step']
                        logger.info(f"The testing process resumes from the step no.{start_step}. ")
                    # checkpoint does not exist
                    except FileNotFoundError:
                        start_step = 0
                        logger.info(f"No checkpoint is found in {test_rank_path}. "
                                    f"The testing process will start from scratch. ")
                else:
                    start_step = 0
                    logger.info(f"The testing process will start from scratch. ")

                # initialize the dataloaders from the given starting point
                data_loaders = cls.dict_transform(iterator, lambda x: iter(x.build_loader(start_step=start_step)))
                test_indices = cls.dict_transform(iterator, lambda x: x.get_batch_indices())
                # if there are multiple dataloaders for the current testing set,
                # the sample indices of the first element will be used to make the reports
                if isinstance(test_indices, Dict):
                    test_indices = test_indices[list(test_indices.keys())[0]]
                # report the total number of testing steps needed to be done
                total_step_num = len(test_indices)
                logger.info(f"Totally {total_step_num} testing steps.")

                # make sure that no gradient appears during testing
                model.eval()
                with torch.inference_mode():
                    monitor.start_epoch(total_step_num=total_step_num)
                    # iterate the testing batches
                    for i in range(total_step_num):
                        if i < start_step:
                            continue

                        # only fetch the testing data right before decoding and evaluation
                        test_batch = cls.dict_transform(src_dict=data_loaders, transform_func=next)
                        # skip the empty testing batch
                        if cls.is_empty_batch(test_batch):
                            continue
                        # evaluate the current testing batch and get the evaluation results
                        try:
                            test_results = model.evaluate(test_batch=test_batch, infer_conf=infer_cfg)
                        # skip the current step if encounter an error (any kind)
                        except Exception as e:
                            logger.warn(
                                f"Rank no.{torch.distributed.get_rank() if args.distributed else '0'} meets the error "
                                f"{e} at step no.{i}. "
                                f"Indices of the involved testing samples in this step is {test_indices[i]}.")
                            continue
                        # record evaluation results
                        monitor.step(step_num=i + 1, test_results=test_results, test_index=test_indices[i])

                        # reduce the number of IO operations to speed up the testing
                        if (i + 1) % monitor.report_per_steps == 0 or i == total_step_num - 1:
                            # save the checkpoint of the current step for both resuming and multi-GPU evaluation
                            # the iteration conditions of the test dataloader will also be saved for resuming
                            torch.save(dict(start_step=i + 1, monitor=monitor.state_dict()),
                                       os.path.join(test_rank_path, 'checkpoint.pth'))

                # waiting for the data saving daemon process to finish before calling finish_epoch()
                monitor.wait_empty_queues()

                if not args.distributed or args.rank == 0:
                    # obtain the group information of the current iterator
                    group_info = None
                    if isinstance(iterator, Iterator):
                        # Dict[str, Dict[str, str]]
                        group_info = iterator.get_group_info()
                    elif isinstance(iterator, Dict):
                        # List[Dict[str, Dict[str, str]]]
                        group_info_list = [value.get_group_info() for value in iterator.values()]
                        for group_dict in group_info_list:
                            if group_dict is not None:
                                group_info = group_dict
                                break
                    else:
                        raise RuntimeError

                    # finish the evaluation and store the results to the disk
                    monitor.finish_epoch(meta_info=group_info)

    @classmethod
    def set_random_seeds(cls, seed: int):
        """
        Set random seeds for python environment, numpy environment and torch environment

        Note:
            1. torch.random.manual_seed(seed) is the same with torch.manual_seed(seed),
                so it is not necessary to be included here.
            2. torch.cuda.manual_seed_all(seed) is also not included here because we initialize the processes on
                different GPUs with different random seeds depending on the GPU number to avoid the process homogeneity.

        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    @classmethod
    def gather_all_iter_ascii(cls, iterator: Iterator, device: torch.device):
        """

        Args:
            iterator:
            device:

        Returns:

        """
        # turn the message into ASCII codes and gather the codes length
        _iter_asc = torch.LongTensor([ord(char) for char in str(iterator)])
        _iter_asc_len = torch.LongTensor([_iter_asc.size(0)]).cuda(device)

        _iter_asc_lens = torch.LongTensor([0 for _ in range(torch.distributed.get_world_size())]).cuda(device)
        torch.distributed.all_gather_into_tensor(_iter_asc_lens, _iter_asc_len)

        # padding the ASCII codes to the same length and gather them
        if _iter_asc_len < _iter_asc_lens.max():
            _iter_asc = torch.cat((_iter_asc,
                                   torch.zeros(_iter_asc_lens.max().item() - _iter_asc_len.item(),
                                               dtype=torch.int64)))

        _iter_ascs = torch.zeros((torch.distributed.get_world_size(), _iter_asc_lens.max().item()),
                                 dtype=torch.int64, device=device)
        torch.distributed.all_gather_into_tensor(_iter_ascs, _iter_asc.cuda(device))
        return [_iter_asc for _iter_asc in _iter_ascs], _iter_asc_lens

    @classmethod
    def main_worker(cls, gpu: int, args: argparse.Namespace):
        """
        The main body of a process on one GPU.

        Args:
            gpu:
            args:

        """
        # --- 0. Random Seed Preparation --- #
        # set different random seeds for the different GPU processes in DDP mode to avoid the process homogeneity
        if args.distributed and not args.same_proc_seed:
            args.seed += gpu
        cls.set_random_seeds(args.seed)

        # --- 1. Experimental Reproducibility Preparation --- #
        torch.backends.cudnn.enabled = args.cudnn_enabled
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        torch.backends.cudnn.deterministic = args.cudnn_deterministic
        # torch.use_deterministic_algorithms(torch.backends.cudnn.deterministic)
        # For more details about 'CUBLAS_WORKSPACE_CONFIG',
        # please refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        if V(torch.version.cuda) >= V("10.2"):
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # --- 2. DDP Model Distribution Initialization --- #
        if args.distributed:
            # load the global node rank from the os environment in the multi-node setting
            if args.dist_url == "env://":
                args.rank = int(os.environ["RANK"])

            if args.ngpu > 1:
                # Here, the rank is turned from the local node-level rank to the global process-level rank
                # the input argument 'gpu' is the local rank of the current process in the specific node
                args.rank = args.rank * args.ngpu + gpu
            # initialize the distributed environment, connections among all the processes are established here
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)

        # --- 3. Experimental Environment Logging --- #
        if args.config is not None:
            _config_split = args.config.split('/')
        else:
            _config_split = None
        # automatically decide the result path if not given
        if args.train_result_path is None:
            assert _config_split is not None, \
                "If you want to automatically generate train_result_path, please give the configuration by '--config'."
            args.train_result_path = '/'.join(_config_split[:-2] + ['exp', '.'.join(_config_split[-1].split('.')[:-1])])
        # attach a folder named by args.config to the end of your given result path
        elif args.attach_config_folder_to_path:
            # if `--config` is given, attach the name of exp_cfg to the end of train_result_path
            if _config_split is not None:
                args.train_result_path = os.path.join(
                    args.train_result_path, '.'.join(_config_split[-1].split('.')[:-1]))

        # initialize the logger and save current script command
        logger = logger_stdout_file(args.train_result_path,
                                    'train' if args.train else None, args.distributed, args.rank)

        # logging the beginning info of the experiment
        logger.info(f"Current script command: {' '.join([xi for xi in sys.argv])}")
        if args.distributed:
            logger.info(f"Multi-GPU distribution information: "
                        f"backend={args.dist_backend}, init_method={args.dist_url}, "
                        f"nnode={int(args.world_size / args.ngpu)}, ngpu_per_node={args.ngpu}, "
                        f"used_gpus={args.gpus}.")

        # initialize the computational equipments
        assert torch.cuda.is_available(), "CUDA is not available! It fails to conduct GPU training."
        # args.gpu is the GPU used in the current process while args.gpus are all the available GPUss
        args.gpu = args.gpus[gpu] if args.ngpu > 1 else args.gpus
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.device(device)
        logger.info(f"Used GPU in the master process: {device}")

        # --- 4. Configuration Loading --- #
        # resume from an existing checkpoint, loading the old data and train configurations
        if args.resume:
            # loading the existing data and train configurations
            # But the input data configuration has higher priority than the existing one
            if args.data_cfg is not None:
                data_cfg = load_yaml(open(args.data_cfg)) if isinstance(args.data_cfg, str) else args.data_cfg
            else:
                data_cfg = load_yaml(open(os.path.join(args.train_result_path,
                                                       f"{'train' if args.train else 'test'}_data_cfg.yaml")))
            # training configuration will be loaded from the existing file
            train_cfg = load_yaml(open(os.path.join(args.train_result_path, "train_cfg.yaml")))
        # start from scratch, loading the new data and train configurations
        else:
            assert args.data_cfg is not None and args.train_cfg is not None, \
                "Please specify a data configuration file and a train configuration file!"
            data_cfg = load_yaml(open(args.data_cfg)) if isinstance(args.data_cfg, str) else args.data_cfg
            train_cfg = load_yaml(open(args.train_cfg)) if isinstance(args.train_cfg, str) else args.train_cfg

        # --- 5. Data Iterator Initialization --- #
        iterators = cls.build_iterators(data_cfg=data_cfg, args=args)

        # logging the information of the iterators
        _iter_message = "The information of the iterators:"
        for dset, iters in iterators.items():
            # single iterator for the current dataset
            if isinstance(iters, Iterator):
                # gather the iterator message from all the process in the multi-GPU distributed training mode
                if args.distributed:
                    _iter_ascs, _iter_asc_lens = cls.gather_all_iter_ascii(iters, device)

                    # recover the codes from all the processes back to the text
                    for i, asc in enumerate(_iter_ascs):
                        _iter_text = ''.join([chr(a) for a in asc[:_iter_asc_lens[i]].tolist()])
                        _iter_message += f"\nThe iterator in the {dset} set of the rank no.{i}: {_iter_text}"
                # directly report the message in the single-GPU mode
                else:
                    _iter_message += f"\nThe iterator in the {dset} set: {iters}"

            # multiple iterators for the current dataset
            elif isinstance(iters, Dict):
                for name, iterator in iters.items():
                    # gather the iterator message from all the process in the multi-GPU distributed training mode
                    if args.distributed:
                        _iter_ascs, _iter_asc_lens = cls.gather_all_iter_ascii(iterator, device)

                        # recover the codes from all the processes back to the text
                        for i, asc in enumerate(_iter_ascs):
                            _iter_text = ''.join([chr(a) for a in asc[:_iter_asc_lens[i]].tolist()])
                            _iter_message += f"\nThe {name} iterator in the {dset} set of the rank no.{i}: {_iter_text}"
                    # directly report the message in the single-GPU mode
                    else:
                        _iter_message += f"\nThe {name} iterator in the {dset} set: {iterator}"

        logger.info(_iter_message)

        # --- 6. Model Initialization --- #
        assert "model" in train_cfg.keys(), "Please fill in the 'model' tag of your given train_cfg!"
        model = cls.build_model(train_cfg['model'], args=args, device=device)
        logger.info(model_summary(model))

        # for the process of single-GPU training or the rank 0 process of multi-GPUs training
        if not args.distributed or args.rank == 0:
            # dumping all the configuration files into train_result_path for resuming
            with open(os.path.join(args.train_result_path, "exp_cfg.yaml"), 'w', encoding="utf-8") as f:
                yaml.dump(vars(args), f, sort_keys=False)
            with open(os.path.join(args.train_result_path, f"{'train' if args.train else 'test'}_data_cfg.yaml"),
                      'w', encoding="utf-8") as f:
                yaml.dump(data_cfg, f, sort_keys=False)
            with open(os.path.join(args.train_result_path, "train_cfg.yaml"), 'w', encoding="utf-8") as f:
                yaml.dump(train_cfg, f, sort_keys=False)

        # --- 7.1. Model Training Branch --- #
        if args.train:
            # initialize the Monitor for training and validation
            monitor = None if args.distributed and args.rank != 0 else \
                TrainValidMonitor(logger=logger, args=args, model=model)

            # loading the model from the existing checkpoint for resuming the training process
            args.start_epoch = cls.resume(args=args, model=model, monitor=monitor)

            # DDP Wrapping of the model must be done after model checkpoint loading
            if args.distributed:
                # turn the batchnorm layers into the sync counterparts
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                # Here the model buffers and parameters of the master process are broadcast to the other processes
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)

            # initialize the OptimSchedulers after DDP wrapping (including optimization resuming)
            assert "optim_sches" in train_cfg.keys(), "Please fill in the 'optim_sches' tag!"
            optim_sches = cls.build_optim_sches(model=model, optim_sche_cfg=train_cfg['optim_sches'], args=args)

            # logging the information of the optimschedulers
            for name, optim_sche in optim_sches.items():
                logger.info(f"The {name} OptimScheduler: {optim_sche}")

            # start the training process
            cls.train(args=args, data_cfg=data_cfg, iterators=iterators, model=model,
                      optim_sches=optim_sches, logger=logger, monitor=monitor)

        # --- 7.2. Model Testing Branch --- #
        elif args.test:
            if isinstance(args.test_model, str):
                args.test_model = [args.test_model]
            elif not isinstance(args.test_model, List):
                raise ValueError

            # loop each model to be tested
            for model_name in args.test_model:
                # get the path of the target model parameters
                _models_path = os.path.join(args.train_result_path, 'models')
                # for compatibility with the older version
                if os.path.exists(os.path.join(_models_path, f'{model_name}.mdl')):
                    model_path = os.path.join(_models_path, f'{model_name}.mdl')
                elif os.path.exists(os.path.join(_models_path, f'{model_name}.pth')):
                    model_path = os.path.join(_models_path, f'{model_name}.pth')
                else:
                    raise RuntimeError(f"{os.path.join(_models_path, '%s.pth' % model_name)} is not found!")

                # load the target model parameters
                model.load_state_dict(torch.load(model_path, map_location=model.device))

                # start the testing process
                cls.test(args=args, test_model=model_name, iterators=iterators, model=model)

        else:
            raise RuntimeError

        # --- 8. Release Computational Resource --- #
        if args.distributed and args.rank == 0:
            torch.distributed.destroy_process_group()
        sys.exit(0)

    @classmethod
    def main(cls, args: argparse.Namespace):
        """
        The beginning of a experiment branch (training or testing).
        This function decides the single-GPU or multi-GPU training sub-branch.

        Args:
            args: argparse.Namespace
                The input arguments for the experiment.

        """
        # This block is for safely calling torch.cuda API in the main process
        try:
            torch.multiprocessing.set_start_method('spawn')
        except RuntimeError:
            pass

        # --- 1. Initialization of the used GPUs in the current experiment --- #
        # 'CUDA_VISIBLE_DEVICES' has the higher priority than the argument 'gpus'
        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
            args.gpus = os.environ['CUDA_VISIBLE_DEVICES']
            if not isinstance(args.gpus, str):
                args.gpus = str(args.gpus)

        # if 'CUDA_VISIBLE_DEVICES' is not given, initialize it by args.gpus
        elif args.gpus is not None:
            if isinstance(args.gpus, List):
                args.gpus = ','.join([str(g) if not isinstance(g, str) else g for g in args.gpus if g != ''])
            elif isinstance(args.gpus, int):
                args.gpus = str(args.gpus)
            else:
                assert isinstance(args.gpus, str)
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

        # if both 'CUDA_VISIBLE_DEVICES' and args.gpus are not given, automatically select available GPUs
        else:
            args.gpus = get_idle_gpu(args.ngpu)
            # make sure that GPU no.0 is the first GPU if it is selected
            args.gpus = ','.join(sorted([str(gpu.id) for gpu in args.gpus]))
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

        # convert the GPU absolute number to the GPU relative index to fit 'CUDA_VISIBLE_DEVICES'
        args.gpus = [idx for idx, _ in enumerate(args.gpus.split(','))]

        # check the GPU configuration
        assert len(args.gpus) >= args.ngpu, \
            f"The visible GPUs {args.gpus} are fewer than the GPUs you would like to use {args.ngpu}! " \
            f"Please use the argument '--gpus' to directly specify your target GPUs."
        if len(args.gpus) == 1:
            args.gpus = args.gpus[0]

        # --- 2. Initialization of DDP distribution pipeline --- #
        # get the world_size from the command line, world_size here means the number of nodes
        if args.dist_url == "env://":
            args.world_size = int(os.environ["WORLD_SIZE"])
            raise NotImplementedError("Multi-node DDP distributed training is not supported now.....")

        # distributed is set to true if multiple GPUs are specified or multiple nodes are specified
        # args.world_size > 1 means multi-node distribution while args.ngpu > 1 means multi-GPU distribution
        args.distributed = args.world_size > 1 or args.ngpu > 1

        # multi-GPU distributed training and testing
        if args.ngpu > 1:
            # check whether the input number of GPUs is valid
            ngpus_per_node = torch.cuda.device_count()
            if args.ngpu > ngpus_per_node:
                warnings.warn(
                    f"Your input args.ngpu ({args.ngpu}) is larger than the GPUs you have on your machine "
                    f"({ngpus_per_node}). Currently, the real args.ngpu becomes {ngpus_per_node}."
                )
                args.ngpu = ngpus_per_node
            # here world_size becomes the total number of processes on all nodes
            args.world_size = args.ngpu * args.world_size

            # automatic port selection if no specified port (only one ':' in args.dist_url)
            if len(args.dist_url.split(':')) < 3:
                args.dist_url += f':{get_idle_port()}'
            else:
                raise RuntimeError

            # run one process on each GPU
            mp.spawn(cls.main_worker, nprocs=args.ngpu, args=(args,))

        # single-GPU training and testing
        elif args.ngpu == 1:
            if isinstance(args.gpus, List) and len(args.gpus) > 1:
                warnings.warn(f"Your input args.ngpu {args.gpus} is more than one. "
                              f"Currently, the GPU no.{args.gpus[0]} will be used.")
                args.gpus = args.gpus[0]
            cls.main_worker(args.gpus, args)

        # CPU testing with the multiprocessing strategy
        elif args.test:
            raise NotImplementedError("Multiprocessing CPU testing part has not been implemented yet......")

        # CPU training is not supported
        else:
            raise RuntimeError("Our toolkit doesn't support CPU training. Please specify a number of GPUs......")

    @classmethod
    def run(cls):
        """
        The preparation area of Runner where the configuration is parsed and converted into code-friendly format.

        """
        # --- 0. Get the Command Line Arguments --- #
        args = cls.parse()

        # --- 1. Read the Non-Config Arguments from the Command Line --- #
        # Currently, 'world_size' and 'rank' are not provided to users to set
        given_args = ['world_size', 'rank']
        # The arguments that users give in the command line should not be refreshed by the argument '--config'
        for i in sys.argv:
            if i.startswith('--'):
                given_args.append(i.replace('-', ''))

        # check the train and test flags
        if 'train' in given_args and 'test' in given_args:
            assert (args.train ^ args.test) is True, \
                "A running job can only conduct either training process or testing process, " \
                "so args.train and args.test cannot be True at the same time. " \
                "If you want to conduct training and testing sequentially, " \
                "please make two running jobs where the first job has args.train=True and args.test=False and " \
                "the second job has args.train=False and args.test=True."
        elif 'train' in given_args:
            given_args.append('test')
            args.test = not args.train
        elif 'test' in given_args:
            given_args.append('train')
            args.train = not args.test

        # the command 'CUDA_VISIBLE_DEVICES' has the higher priority than the argument 'gpus'
        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
            given_args.append('gpus')
            args.gpus = None

        # --- 2. Overwrite the Arguments by '--config' --- #
        # overwrite args from the args.config
        # Note: the ones given in the command line has the higher priority than args.config
        if args.config is not None:
            args.config = parse_path_args(args.config)
            config = load_yaml(open(args.config, mode='r', encoding='utf-8'))
            for c in config.keys():
                if c not in given_args:
                    # remove the port number in 'dist_url' if given
                    if c == 'dist_url':
                        assert len(config[c].split(':')) <= 3
                        if len(config[c].split(':')) == 3:
                            config[c] = ':'.join(config[c].split(':')[:-1])
                    # skip the existing 'report_per_steps' (either use default value or give it in the command line)
                    if c == 'report_per_steps':
                        continue
                    # set the argument from config to args
                    setattr(args, c, config[c])

        # make sure that all the paths are absolute paths
        if args.train_result_path is not None:
            args.train_result_path = parse_path_args(args.train_result_path)
        if args.test_result_path is not None:
            args.test_result_path = parse_path_args(args.test_result_path)
        if args.data_cfg is not None and not isinstance(args.data_cfg, Dict):
            args.data_cfg = parse_path_args(args.data_cfg)
        if args.train_cfg is not None and not isinstance(args.train_cfg, Dict):
            args.train_cfg = parse_path_args(args.train_cfg)
        if args.infer_cfg is not None:
            if isinstance(args.infer_cfg, str):
                args.infer_cfg = parse_path_args(args.infer_cfg)
            elif isinstance(args.infer_cfg, List):
                args.infer_cfg = [parse_path_args(cfg) if isinstance(cfg, str) else cfg for cfg in args.infer_cfg]
            elif not isinstance(args.infer_cfg, Dict):
                raise TypeError("infer_cfg should be either a string, a List, or a Dict, "
                                f"but got type(args.infer_cfg)={type(args.infer_cfg)}.")

        # --- 3. Start the Experimental Pipeline --- #
        assert (args.train ^ args.test) is True, \
            "A running job can only conduct either training process or testing process, " \
            "so args.train and args.test cannot be True at the same time. " \
            "If you want to conduct training and testing sequentially, " \
            "please make two running jobs where the first job has args.train=True and args.test=False and " \
            "the second job has args.train=False and args.test=True."
        cls.main(args)


if __name__ == '__main__':
    Runner.run()
