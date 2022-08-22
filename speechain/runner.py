"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import argparse
import copy
import os
import sys
import time
import warnings
from functools import partial

import GPUtil
import torch
import numpy as np
import random
import yaml
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Any, List

from speechain.monitor import *
from speechain.iterator.abs import Iterator
from speechain.model.abs import Model
from speechain.optim_sche.abs import OptimScheduler

from speechain.utilbox.log_util import logger_stdout_file, model_summary, distributed_zero_first
from speechain.utilbox.import_util import import_class, get_port
from speechain.utilbox.type_util import str2bool


class Runner(object):
    """
    Runner is the entrance of our toolkit. This static class is made up of several static functions. The
    whole pipeline is done by all static functions step by step. The reason why the functions are all static is to
    prevent Runner from becoming the God class after Inheritance.
    If you are interested in this topic, please refer to https://wiki.c2.com/?GodClass for more details.

    In this class, we provide an overridable interface add_parse() that enables users to add more arguments they
    would like their runners to have.

    Basically, we don't recommend users to override the other functions in this class for toolkit robustnesss.
    However, in case that the existing functions cannot meet your research requirements, you can override them in your
    own runners to fit your specific needs. If it happens, we would appreciate it a lot if you could open an issue
    and let us know.

    Wish you have a happy usage journey in this toolkit ^_^!
    """

    @classmethod
    def add_parse(cls, parser: argparse.ArgumentParser):
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
            default='/ahc/work4/heli-qi/euterpe-heli-qi/recipes/asr/librispeech/train_960/transformer/exp_cfg/bpe10k_nomiddlespace_smooth0.2.yaml',
            help="All-in-one argument setting file. "
                 "You can write all the arguments in this file instead of giving them by command lines."
        )

        # Experimental environment
        group = parser.add_argument_group("Experimental environment.")
        group.add_argument(
            '--seed',
            type=int,
            default=0,
            help="Random seed for initializing your experiment. (default: 0)"
        )
        group.add_argument(
            '--cudnn_benchmark',
            type=str2bool,
            default=False,
            help="Whether to activate torch.backends.cudnn.benchmark. "
                 "This option can speed up the GPU calculation, "
                 "but we don't recommend you to turn it on if you are doing seq2seq tasks such as ASR. "
                 "(default: False)"
        )
        group.add_argument(
            '--cudnn_deterministic',
            type=str2bool,
            default=False,
            help="Whether to activate torch.backends.cudnn.deterministic. "
                 "If you turn on cudnn_benchmark, "
                 "you must also set this arugment to True for the safe calculation. (default: False)"
        )
        group.add_argument(
            '--num_workers',
            type=int,
            default=1,
            help="Number of workers used by each Dataloader. "
                 "If you find that the speed of loading data from the disk is very slow, "
                 "you can try to raise the value of this argument within your machine capability. "
                 "If you want to debug, please set this argument to 0. (default: 1)"
        )
        group.add_argument(
            '--pin_memory',
            type=str2bool,
            default=False,
            help="Whether enable the pin_memory option of each Dataloader. "
                 "This option can activate the pinned memory in your dataloasers and speed up the data loading. "
                 "Often used together with non_blocking=True. (default: False)"
        )
        group.add_argument(
            '--non_blocking',
            type=str2bool,
            default=False,
            help="Whether enable the non_blocking option when putting data on GPUs. "
                 "This option can speed up the model processing. "
                 "Often used together with pin_memory=True. (default: False)"
        )

        # gradient descent related
        group = parser.add_argument_group("Gradient calculation and back-propagation.")
        group.add_argument(
            '--use_amp',
            type=str2bool,
            default=True,
            help="Whether use Automatic Mixed Precision for back-propagation. (default: True)"
        )
        group.add_argument(
            '--grad_clip',
            type=float,
            default=1.0,
            help="Gradient clipping to prevent NaN gradients during training. (default: 1.0)"
        )
        group.add_argument(
            '--grad_norm_type',
            type=float,
            default=2.0,
            help="Gradient normalization type used when clipping the gradients. (default: 2.0)"
        )
        group.add_argument(
            '--accum_grad',
            type=int,
            default=1,
            help="Gradient accumulation steps for back-propagation. "
                 "This argument is used for mimicking large batch training on few GPUs. (default: 1)"
        )
        group.add_argument(
            '--ft_factor',
            type=float,
            default=1.0,
            help="The finetuing factor used to scale down the learning rates of your optimizers. "
                 "A usual setting for finetuning is ft_factor=0.1 (default: 1.0)"
        )

        # multi-GPU distributed training
        group = parser.add_argument_group("GPU-related configuration")
        group.add_argument(
            "--dist_backend",
            default="nccl",
            type=str,
            help="distributed backend. "
                 "If you are using NVIDIA GPUs, we recommend you set this argument to 'nccl'.",
        )
        group.add_argument(
            "--dist_url",
            type=str,
            default="tcp://127.0.0.1",
            help="If you want to train your model on multiple nodes, please set dist_url='env://'. "
                 "In this case, env values of 'MASTER_PORT', 'MASTER_ADDR', 'WORLD_SIZE', and 'RANK' are referred. "
                 "The default value is 'tcp://127.0.0.1' for single-node distributed training and "
                 "a free port will be automatically selected. "
                 "You can also specify the port manually in this argument by 'tcp://127.0.0.1:xxxxx'.",
        )
        group.add_argument(
            "--world_size",
            default=1,
            type=int,
            help="The number of nodes for distributed training. "
                 "If you want to conduct multi-node distributed training by the command line, "
                 "please set this argument to -1."
        )
        group.add_argument(
            '--rank',
            default=0,
            type=int,
            help="The global rank of the node for distributed training. "
                 "If you want to conduct multi-node distributed training by the command line, "
                 "please set this argument to -1."
        )
        group.add_argument(
            '--ngpu',
            type=int,
            default=1,
            help="Number of GPUs used to run your experiment. "
                 "This argument replaces the traditional 'multiprocessing-distributed' for distributed training. "
                 "(default: 1)"
        )
        group.add_argument(
            '--gpus',
            type=str,
            default=None,
            help="Specified GPUs used to run your experiment. "
                 "If you want to specify multiple GPUs, please give this argument in the form of 'x,x,x' "
                 "where different GPU numbers are separated by a comma (please don't end this argument with ','). "
                 "If there are no specified GPUs, they will be automatically selected from the available free GPUs. "
                 "Of course, you could also give your specified GPUs by CUDA_VISIBLE_DEVICES!"
        )

        # Training monitoring
        group = parser.add_argument_group("Training process monitoring.")
        group.add_argument(
            '--result_path',
            type=str,
            default=None,
            help="The folder path to store all the result files of your experiment. "
                 "If None, the result path will be automatically decided by your input data_cfg and train_cfg. "
                 "(default: None)")
        group.add_argument(
            '--train',
            action='store_true',
            help="Whether go through the training phase or skip it. "
                 "For training, please attach the argument '--train' to your command. "
                 "(default: False)"
        )
        group.add_argument(
            '--dry_run',
            action='store_true',
            help="Whether to turn on the dry running mode. "
                 "In this mode, only the data loading will be done to see its speed and robustness. "
                 "For dry running, please attach the argument '--dry_run' to your command. (default: False)"
        )
        group.add_argument(
            '--no_optim',
            action='store_true',
            help="Whether to skip the optimization part. "
                 "In this mode, only the data loading and model forward will be done to see its speed and robustness. "
                 "For dry running, please attach the argument '--no_optim' to your command. "
                 "Note: 'dry_run' has the higher priority than 'no_optim'. "
                 "It means that the model forward part will be skipped if you give both '--dry_run' and '--no_optim'. "
                 "(default: False)"
        )
        group.add_argument(
            '--resume',
            action='store_true',
            help="Whether continue your unfinished training and testing jobs. "
                 "If True, there must be the checkpoint file of your last experiment. "
                 "This argument is shared by the training and testing branches. (default: False)"
        )
        group.add_argument(
            '--start_epoch',
            type=int,
            default=1,
            help="The starting epoch of your experiments. (default: 1)"
        )
        group.add_argument(
            '--num_epochs',
            type=int,
            default=1000,
            help="Maximum number of training epochs of your experiments. (default: 1000)"
        )
        group.add_argument(
            '--valid_per_epochs',
            type=int,
            default=1,
            help="The interval of goint through validation phase during training. "
                 "If not specified, validation will follow training in every epoch. (default: 1)"
        )
        group.add_argument(
            '--report_per_steps',
            type=int,
            default=0,
            help="The interval of reporting the step information during training and testing. "
                 "Positive integers (absolute interval) mean a report will be made after each 'report_per_steps' steps, "
                 "negative integers (relative interval) mean there will be '-report_per_steps' reports in each epoch. "
                 "(default: 0 -> there will be 10 reports in each epoch.)"
        )
        group.add_argument(
            '--best_model_num',
            type=int,
            default=5,
            help="The number of the best models recorded during training. (default: 5)"
        )
        group.add_argument(
            '--best_model_mode',
            type=str,
            default=None,
            help="The way to select the best models. Must be either 'max' or 'min'."
        )
        group.add_argument(
            '--best_model_metric',
            type=str,
            default=None,
            help="The name of the metric used to pick up the best models."
        )
        group.add_argument(
            '--early_stopping_patience',
            type=int,
            default=10,
            help="The maximum number of epochs where the model doesn't improve its performance. "
                 "(default: 10)"
        )
        group.add_argument(
            '--early_stopping_threshold',
            type=float,
            default=0.01,
            help="The threshold to refresh early-stopping in the monitor. "
                 "Positive values in (0.0, 1.0) represent the relative threshold over the current best results, "
                 "negative values represent the absolute threshold over the current best results. "
                 "0 means no threshold is applied in the monitor."
                 "(default: 0.01)"
        )

        # Training Snapshotting
        group = parser.add_argument_group("Training Snapshotting-related")
        group.add_argument(
            '--monitor_snapshot_conf',
            type=Dict[str, Any],
            default=dict(),
            help="The configuration of the SnapShooters of the monitors during training and validation. "
                 "This argument should be given in the form of a Dict. "
                 "(default: an empty Dict)"
        )
        group.add_argument(
            '--model_snapshot_number',
            type=int,
            default=3,
            help="The number of the SnapShots made by your model in each validation epoch. "
                 "The snapshots will be taken by the first sample of each validation batch, "
                 "so this argument should be smaller than the number of your validation batches. "
                 "(default: 3)"
        )
        group.add_argument(
            '--model_snapshot_interval',
            type=int,
            default=5,
            help="The snapshotting interval of your model during validation. "
                 "This argument determines how frequently the snapshots are updated (unit: epoch). "
                 "(default: 5)"
        )

        # Testing
        group = parser.add_argument_group("Testing-related")
        group.add_argument(
            '--test',
            action='store_true',
            help="Whether go through the testing phase or skip it. "
                 "For testing, please attach the argument '--test' to your command. (default: False)"
        )
        group.add_argument(
            '--test_model',
            type=str,
            default=None,
            help="The names of the model you want to evaluate in the testing phase. "
                 "Multiple model names can be given as a list. "
                 "If you only want to evaluate one model, directly giving the string of its name is OK. (default: None)"
        )
        group.add_argument(
            '--bad_cases_selection',
            type=list,
            default=None,
            help="The selection method of the top-n bad cases. "
                 "This argument should be given as a tri-tuple ('metric', 'max' or 'min', N). "
                 "For example, ('wer', 'max', 10) means the testing samples with top-10 largest wer will be selected. "
                 "Multiple tuples can be given to present different sets of top-n bad cases. (default: None)"
        )

        # Experiment configuration
        group = parser.add_argument_group("Experiment configuration files.")
        group.add_argument(
            '--data_cfg',
            type=str,
            default=None,
            help="The configuration file of data loading and batching."
        )
        group.add_argument(
            '--train_cfg',
            type=str,
            default=None,
            help="The configuration file of the structure of the model and optimschedulers."
        )
        group.add_argument(
            '--test_cfg',
            type=str,
            default=None,
            help="The configuration file of the testing hyperparameters. "
                 "Multiple testing configuration files can be given as a list. "
                 "If you only want to use one file, directly giving the string of its name is OK. (default: None)"
        )

        # Add customized arguments if needed
        parser = cls.add_parse(parser)

        return parser.parse_args()


    @classmethod
    def build_iterators(cls, data_cfg: Dict[str, Dict], args: argparse.Namespace) \
            -> Dict[str, Dict[str, Iterator]]:
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
        # turn the keys in data_cfg into their lowercase forms
        for key in data_cfg.keys():
            data_cfg[key.lower()] = data_cfg.pop(key)

        # all the available combinations of keys
        dset_keys = dict(
            train_test=['train', 'valid', 'test'],
            train=['train', 'valid'],
            test=['test']
        )

        # check the first-level keys of data_cfg
        assert list(data_cfg.keys()) in dset_keys.values(), \
            f"The first-level tags of data_cfg must be one of {list(dset_keys.values())}."

        # initialize all iterators
        iterators = dict()
        batch_nums = dict()

        # get the mode of the current experiment
        if args.train:
            _mode = 'train'
        elif args.test:
            _mode = 'test'
        else:
            raise RuntimeError

        # looping each dataset in the configuration
        for dset, iter_list in data_cfg.items():
            # filter the unused dset for the current mode
            if dset not in dset_keys[_mode]:
                continue

            # initialize the valid dset
            iterators[dset] = dict()
            batch_nums[dset] = list()

            # looping each iterator in the current dataset
            for name, iterator in iter_list.items():
                iterator_class = import_class('speechain.iterator.' + iterator["type"])
                iterators[dset][name] = iterator_class(seed=args.seed,
                                                       num_workers=args.num_workers,
                                                       pin_memory=args.pin_memory,
                                                       distributed=args.distributed,
                                                       **iterator["conf"])
                batch_nums[dset].append(len(iterators[dset][name]))


        # set the relative reporting interval during training or testing
        if args.report_per_steps <= 0:
            _reports_per_epoch = 10 if args.report_per_steps == 0 else int(-args.report_per_steps)
            args.report_per_steps = min(batch_nums[_mode]) // _reports_per_epoch
        # check the absolute reporting interval during training and testing
        else:
            assert int(args.report_per_steps) <= min(batch_nums[_mode]), \
                f"If args.report_per_steps is given as a positive integer, " \
                f"it should be smaller than the minimal {_mode} batch number ({min(batch_nums[_mode])}). " \
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
        assert "model_conf" in model_cfg.keys(), "Please specify the model_conf!"
        assert "module_conf" in model_cfg.keys(), "Please specify the module_conf!"
        assert "criterion_conf" in model_cfg.keys(), "Please specify the criterion_conf!"

        model_class = import_class('speechain.model.' + model_cfg['model_type'])
        return model_class(model_conf=model_cfg['model_conf'],
                           module_conf=model_cfg['module_conf'],
                           criterion_conf=model_cfg['criterion_conf'],
                           device=device,
                           args=args).cuda(device=device)

    @classmethod
    def build_optim_sches(cls,
                          model: Model,
                          optim_sche_cfg: Dict[str, Any],
                          args: argparse.Namespace) -> Dict[str, OptimScheduler]:
        """
        This static function builds the OptimSchedulers used in the pipeline. The configuration of the
        OptimSchedulers is given in the value of 'optim_sches' key in your specified 'train_cfg'.

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
        optim_sches = dict()

        for name, optim_sche in optim_sche_cfg.items():
            optim_sche_class = import_class('speechain.optim_sche.' + optim_sche['type'])
            optim_sches[name] = optim_sche_class(model=model,
                                                 accum_grad=args.accum_grad,
                                                 ft_factor=args.ft_factor,
                                                 grad_clip=args.grad_clip,
                                                 grad_norm_type=args.grad_norm_type,
                                                 **optim_sche['conf'])
        return optim_sches

    @classmethod
    def resume(cls,
               args: argparse.Namespace,
               model: Model,
               optim_sches: Dict[str, OptimScheduler],
               train_monitor: Monitor,
               valid_monitor: Monitor) -> int:
        """

        Args:
            args: argparse.Namespace
                The input arguments.
            model: Model
                The model to be trained.
            optim_sches: Dict
                The dictionary of the OptimSchedulers used to update the model parameters.
            train_monitor: Monitor
                The training monitor used to monitor the training part
            valid_monitor: Monitor
                The validation monitor used to monitor the validation part

        Returns:
            The number of the starting epoch. If the training resumes from an existing checkpoint, then the starting
            epoch will be loaded from the checkpoint; otherwise, 1 will be returned.

        """
        # start the training from the existing checkpoint
        if args.resume:
            # load the existing checkpoint
            try:
                checkpoint = torch.load(os.path.join(args.result_path, "checkpoint.pth"), map_location=model.device)
                # load the checkpoint information into the current experiment
                start_epoch = checkpoint['start_epoch']
                model.load_state_dict(checkpoint['latest_model'])
                if train_monitor is not None and valid_monitor is not None:
                    train_monitor.load_state_dict(checkpoint['train_monitor'])
                    valid_monitor.load_state_dict(checkpoint['valid_monitor'])
                    # info logging
                    train_monitor.logger.info(f"The training process resumes from the epoch no.{start_epoch}.")
                for name, optim_sche in optim_sches.items():
                    optim_sche.load_state_dict(checkpoint['optim_sches'][name])

            # checkpoint does not exist
            except FileNotFoundError:
                start_epoch = 1
                train_monitor.logger.info(f"No checkpoint is found in {args.result_path}. "
                                          f"The training process will start from scratch.")

        # start the training from scratch
        else:
            start_epoch = 1

        return start_epoch


    @classmethod
    def pickup_first_sample(cls, valid_batch: Any):
        """

        Args:
            valid_batch:

        Returns:

        """
        if isinstance(valid_batch, Dict):
            return {key: cls.pickup_first_sample(value) for key, value in valid_batch.items()}
        elif isinstance(valid_batch, torch.Tensor):
            return valid_batch[0].unsqueeze(0)
        elif isinstance(valid_batch, List):
            return [valid_batch[0]]
        else:
            raise TypeError


    @classmethod
    def dict_transform(cls, src_dict: Dict, transform_func):
        """

        Args:
            src_dict:
            transform_func:

        Returns:

        """
        tgt_dict = dict()
        # loop the sub-dict of each dataloader
        for key, value in src_dict.items():
            tgt_dict[key] = transform_func(value)
        # if there is only one sub-dict (single-dataloder training)
        if len(tgt_dict) == 1:
            tgt_dict = tgt_dict[key]
        return tgt_dict


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
    def train(cls,
              args: argparse.Namespace,
              iterators: Dict[str, Dict[str, Iterator]],
              model: Model,
              optim_sches: Dict[str, OptimScheduler],
              logger,
              train_monitor: TrainMonitor,
              valid_monitor: ValidMonitor):
        """

        Args:
            args: argparse.Namespace
                The input arguments.
            iterators: Dict
                The dictionary that contains all the iterators for training and validation.
            model: Model
                The model to be trained.
            optim_sches: Dict
                The dictionary that contains all the OptimSchedulers used to update the model parameters.
            train_monitor: Monitor
                The training monitor that controls the training process of the model and generates the real-time logging
                information.
            valid_monitor: Monitor
                The validation monitor that controls the validation process of the model and generates the real-time
                logging information.

        Returns:

        """
        assert args.start_epoch <= args.num_epochs, "The starting epoch is larger than args.num_epochs!"

        # checking the data lengths of all training iterators
        train_batch_nums = set([len(iterator) for iterator in iterators['train'].values()])
        min_train_batch_num = min(train_batch_nums)
        if len(train_batch_nums) != 1:
            logger.info(f"Your training iterators have different batch numbers: {train_batch_nums}. "
                        f"The real batch number during training is set to {min_train_batch_num}!")

        # checking the data lengths of all validation iterators
        valid_batch_nums = set([len(iterator) for iterator in iterators['valid'].values()])
        min_valid_batch_num = min(valid_batch_nums)
        if len(valid_batch_nums) != 1:
            logger.info(f"Your validation iterators have different batch numbers: {valid_batch_nums}. "
                        f"The real batch number during validation is set to {min_valid_batch_num}!")

        # synchronize the batch numbers across all the distributed processes
        if args.distributed:
            # make sure that all processes have the same number of training steps
            _batch_num_list = [torch.LongTensor([0]).cuda(model.device)
                               for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(_batch_num_list, torch.LongTensor([min_train_batch_num]).cuda(model.device))
            min_train_batch_num = min([batch_num.item() for batch_num in _batch_num_list])

            # make sure that all processes have the same number of validation steps
            torch.distributed.all_gather(_batch_num_list, torch.LongTensor([min_valid_batch_num]).cuda(model.device))
            min_valid_batch_num = min([batch_num.item() for batch_num in _batch_num_list])

        # Initialize the gradient scaler for AMP training
        scaler = GradScaler() if args.use_amp else None

        # loop each epoch until the end
        for epoch in range(args.start_epoch, args.num_epochs + 1):
            # set the random seeds for the current epoch
            cls.set_random_seeds(args.seed + epoch)
            # start the current training epoch
            if train_monitor is not None:
                train_monitor.start_epoch(epoch)
            # initialize all the training dataloaders
            data_loaders = {name: iter(iterator.build_loader(epoch)) for name, iterator in iterators['train'].items()}

            # loop all the training batches
            model.train()
            for step in range(1, min_train_batch_num + 1):
                # --- data loading part --- #
                with cls.measure_time(train_monitor)("data_load_time"):
                    train_batch = cls.dict_transform(src_dict=data_loaders, transform_func=next)

                # forward the batch to get the training criteria and optimize the model
                train_metrics = None
                optim_lr = None
                # whether to skip the model forward part and model optimization part
                if not args.dry_run:
                    # --- model forward part --- #
                    with autocast(enabled=scaler is not None):
                        with cls.measure_time(train_monitor)("model_forward_time"):
                            losses, train_metrics = model(batch_data=train_batch)

                    # whether to skip the model optimization part
                    if not args.no_optim:
                        # --- loss backward and optimization part --- #
                        optim_lr = dict()
                        for name, optim_sche in optim_sches.items():
                            optim_sche.step(losses=losses, scaler=scaler,
                                            time_func=cls.measure_time(train_monitor),
                                            optim_name=name, step_num=int(step + (epoch - 1) * min_train_batch_num))
                            optim_lr[name] = optim_sche.get_lr()

                # log the information of the current training step
                if train_monitor is not None:
                    train_monitor.step(step_num=step, optim_lr=optim_lr, train_metrics=train_metrics)

            # finish the current training epoch
            if train_monitor is not None:
                train_monitor.finish_epoch()

            # check the validation interval
            if epoch % args.valid_per_epochs == 0:
                # start the validation part of the current epoch
                if valid_monitor is not None:
                    valid_monitor.start_epoch(epoch)
                # initialize all the validation dataloaders
                data_loaders = {name: iter(iterator.build_loader(epoch)) for name, iterator in iterators['valid'].items()}
                valid_indices = {name: iterator.batches for name, iterator in iterators['valid'].items()}

                # make sure that no gradient appears during validation
                model.eval()
                with torch.no_grad():
                    # loop all validation batches
                    for step in range(min_valid_batch_num):
                        # --- data loading part --- #
                        with cls.measure_time(valid_monitor)("data_load_time"):
                            valid_batch = cls.dict_transform(src_dict=data_loaders, transform_func=next)
                            first_sample = cls.pickup_first_sample(valid_batch)

                        # forward the batch to get the validation criteria
                        valid_metrics = None
                        # whether to skip the model forward part
                        if not args.dry_run:
                            # --- model forward part --- #
                            with cls.measure_time(valid_monitor)("model_forward_time"):
                                valid_metrics = model(batch_data=valid_batch)

                                # evaluate the model at the current validation step and visualize the results
                                if epoch % args.model_snapshot_interval == 0 and step < args.model_snapshot_number:
                                    # make sure that all processes go through the validation phase smoothly
                                    # with distributed_zero_first(args.distributed, args.rank):
                                    if valid_monitor is not None:
                                        # obtain the index of the choosen sample for visualization
                                        sample_index = cls.dict_transform(src_dict=valid_indices,
                                                                          transform_func=lambda x: x[step][0])
                                        # feed the choosen sample to the model
                                        valid_monitor.model_snapshot(epoch=epoch, sample_index=sample_index,
                                                                     used_sample=first_sample)

                        # no step log for the validation step
                        if valid_monitor is not None:
                            valid_monitor.step(valid_metrics=valid_metrics)

                # early-stopping checking for single-GPU
                if not args.distributed and valid_monitor.finish_epoch():
                    break

                # early-stopping checking for multi-GPU
                if args.distributed:
                    stop_flag = torch.BoolTensor([False]).cuda(model.device)
                    flag_list = None

                    if args.rank == 0:
                        if valid_monitor.finish_epoch():
                            stop_flag = torch.BoolTensor([True]).cuda(model.device)
                        flag_list = [stop_flag for _ in range(torch.distributed.get_world_size())]

                    torch.distributed.scatter(stop_flag, flag_list)
                    if stop_flag.item():
                        break


            # store the checkpoint of the current epoch for resuming later
            if not args.distributed or args.rank == 0:
                if not args.dry_run and not args.no_optim:
                    torch.save(
                        {
                            "start_epoch": epoch + 1,
                            "latest_model": model.state_dict() if not args.distributed else model.module.state_dict(),
                            "train_monitor": train_monitor.state_dict(),
                            "valid_monitor": valid_monitor.state_dict(),
                            "optim_sches": {name: o.state_dict() for name, o in optim_sches.items()}
                        },
                        os.path.join(args.result_path, "checkpoint.pth")
                    )

        # check whether all the monitor queues become empty in every minute
        if not args.distributed or args.rank == 0:
            while not train_monitor.empty_queue() or not valid_monitor.empty_queue():
                message = ""
                if not train_monitor.empty_queue():
                    message += "The training snapshooter is still snapshotting. "
                if not valid_monitor.empty_queue():
                    message += "The validation snapshooter is still snapshotting. "
                logger.info(message + "Waiting for 1 minute......")
                time.sleep(60)

        # synchronize all the processes at the end
        if args.distributed:
            torch.distributed.barrier()


    @classmethod
    def test(cls,
             args: argparse.Namespace,
             iterators: Dict[str, Dict[str, Iterator]],
             model: Model):
        """

        Args:
            args: argparse.Namespace
                The input arguments.
            logger:

            iterators: Dict
                The dictionary that contains all the iterators for training and validation.
            model: Model
                The model to be trained.
            monitor: Monitor

        """
        # load the test configuration into a Dict
        assert 'test_cfg' in args and args.test_cfg is not None, "Please specify at least one test configuration file!"
        if isinstance(args.test_cfg, str):
            args.test_cfg = [args.test_cfg]
        test_cfg_dict = {cfg.split("/")[-1].split(".")[0]: cfg for cfg in args.test_cfg}

        # loop each test configuration
        for test_cfg_name, test_cfg in test_cfg_dict.items():
            # configuration-specific result path
            test_result_path = os.path.join(args.result_path, test_cfg_name)
            os.makedirs(test_result_path, exist_ok=True)

            # load the existing testing configuration for resuming
            test_cfg_path = os.path.join(test_result_path, "test_cfg.yaml")
            test_cfg = yaml.load(open(test_cfg_path)) if args.resume else yaml.load(open(test_cfg))

            # save the testing configuration file to result_path
            if not args.distributed or args.rank == 0:
                with open(test_cfg_path, 'w', encoding="utf-8") as f:
                    yaml.dump(test_cfg, f, sort_keys=False)

            # unlike training and validation, the testing iterators are looped one by one
            for name, iterator in iterators['test'].items():
                test_loader = iterator.build_loader()
                test_indices = iterator.get_sample_indices()

                # add the identity symbol to the path for multi-GPU testing
                test_dset_path = os.path.join(test_result_path, args.test_model, name + f'.{args.rank}')
                logger = logger_stdout_file(test_dset_path, file_name='test')
                monitor = TestMonitor(logger=logger, args=args, result_path=test_dset_path)

                if args.resume:
                    # loading the existed checkpoint
                    try:
                        test_checkpoint = torch.load(os.path.join(test_dset_path, 'checkpoint.pth'))
                        monitor.load_state_dict(test_checkpoint['monitor'])
                        start_step = test_checkpoint['start_step']
                        logger.info(f"The testing process resumes from the step no.{start_step}.")
                    # checkpoint does not exist
                    except FileNotFoundError:
                        start_step = 0
                        logger.info(f"No checkpoint is found in {test_dset_path}. "
                                    f"The testing process will start from scratch.")
                else:
                    start_step = 0

                # make sure that no gradient appears during testing
                model.eval()
                with torch.no_grad():
                    monitor.start_epoch(total_step_num=len(iterator))
                    # iterate the testing batches
                    for i, test_batch in enumerate(test_loader):
                        if i < start_step:
                            continue
                        test_results = model.evaluate(test_batch=test_batch, **test_cfg)
                        monitor.step(step_num=i + 1, test_results=test_results, test_index=test_indices[i])

                # make sure that all the processes finish all the testing steps at the same time
                if args.distributed:
                    torch.distributed.barrier()

                if not args.distributed or args.rank == 0:
                    # finish the evaluation and store the results to the disk
                    monitor.finish_epoch(meta_info=iterator.get_meta_info())


    @classmethod
    def set_random_seeds(cls, seed: int):
        """

        Args:
            seed:

        Returns:

        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


    @classmethod
    def main_worker(cls, gpu: int, args: argparse.Namespace):
        """
        The main body of a process (a GPU).

        Args:
            gpu:
            args:

        """
        # initialize random seeds
        cls.set_random_seeds(args.seed)

        # initialize cudnn backend mode
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        torch.backends.cudnn.deterministic = args.cudnn_deterministic

        # initialize the distributed training environment
        if args.distributed:
            # load the global node rank from the os environment in the multi-node setting
            if args.dist_url == "env://" and args.rank == -1:
                args.rank = int(os.environ["RANK"])

            if args.ngpu > 1:
                # Here, the global rank is turned from the node rank to the process rank
                # the input argument 'gpu' is the local rank of the current process
                args.rank = args.rank * args.ngpu + gpu
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)

        # automatically decide the result path if not given
        if args.result_path is None:
            assert args.config is not None, \
                "If you want to automatically generate the result_path, please give the configuration by '--config'."
            _config_split = args.config.split('/')
            args.result_path = '/'.join(_config_split[:-2] + ['exp'] + ['.'.join(_config_split[-1].split('.')[:-1])])

        # initialize the logger and save current script command
        _log_file_name = 'train' if args.train else 'test'
        logger = logger_stdout_file(args.result_path, _log_file_name, args.distributed, args.rank)

        # logging the beginning info of the experiment
        logger.info(f"Current script command: {' '.join([xi for xi in sys.argv])}")
        if args.distributed:
            logger.info(f"Multi-GPU distribution information: "
                        f"backend={args.dist_backend}, init_method={args.dist_url}, "
                        f"nnode={int(args.world_size / args.ngpu)}, ngpu_per_node={args.ngpu}, "
                        f"used_gpus={args.gpus}.")

        # initialize the computational equipments
        assert torch.cuda.is_available(), "CUDA is not available! It fails to conduct GPU training."
        args.gpu = args.gpus[gpu] if args.ngpu > 1 else args.gpus
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.device(device)
        logger.info(f"Used GPU in the master process: {device}")

        # resume from an existing checkpoint, loading the old data and train configurations
        if args.resume:
            # loading the existing data and train configurations
            # But the input data configuration has higher priority than the existing one
            if "data_cfg" in args:
                data_cfg = yaml.load(open(args.data_cfg))
            else:
                data_cfg = yaml.load(open(os.path.join(args.result_path, "data_cfg.yaml")))
            train_cfg = yaml.load(open(os.path.join(args.result_path, "train_cfg.yaml")))
        # start from scratch, loading the new data and train configurations
        else:
            assert args.data_cfg is not None and args.train_cfg is not None, \
                "Please specify a data configuration file and a train configuration file!"
            data_cfg = yaml.load(open(args.data_cfg))
            train_cfg = yaml.load(open(args.train_cfg))

        # initialize the iterators
        iterators = cls.build_iterators(data_cfg=data_cfg, args=args)

        # logging the information of the iterators
        _iter_message = "The information of the iterators:"
        for dset, iters in iterators.items():
            for name, iterator in iters.items():
                # gather the iterator message from all the process in the multi-GPU distributed training mode
                if args.distributed:
                    # turn the message into ASCII codes and gather the codes length
                    _iter_asc = torch.LongTensor([ord(char) for char in str(iterator)])
                    _iter_asc_len = torch.LongTensor([_iter_asc.size(0)]).cuda(device)
                    _iter_asc_lens = [torch.LongTensor([0]).cuda(device) for _ in range(torch.distributed.get_world_size())]
                    torch.distributed.all_gather(_iter_asc_lens, _iter_asc_len)

                    # padding the ASCII codes to the same length and gather them
                    _iter_asc_lens = torch.LongTensor(_iter_asc_lens)
                    if _iter_asc_len < _iter_asc_lens.max():
                        _iter_asc = torch.cat((_iter_asc,
                                               torch.zeros(_iter_asc_lens.max().item() - _iter_asc_len.item(),
                                                           dtype=torch.int64)))
                    _iter_ascs = [torch.zeros(_iter_asc_lens.max().item(), dtype=torch.int64).cuda(device)
                                  for _ in range(torch.distributed.get_world_size())]
                    torch.distributed.all_gather(_iter_ascs, _iter_asc.cuda(device))

                    # recover the codes from all the processes back to the text
                    for i, asc in enumerate(_iter_ascs):
                        _iter_text = ''.join([chr(a) for a in asc[:_iter_asc_lens[i]].tolist()])
                        _iter_message += f"\nThe {name} iterator in the {dset} set of the rank no.{i}: {_iter_text}"
                # directly report the message in the single-GPU mode
                else:
                    _iter_message += f"\nThe {name} iterator in the {dset} set: {iterator}"
        logger.info(_iter_message)

        # initialize the model
        assert "model" in train_cfg.keys(), "Please fill in the 'model' tag of your given train_cfg!"
        model = cls.build_model(train_cfg['model'], args=args, device=device)
        logger.info(model_summary(model))

        # for the process of single-GPU training or the rank 0 process of multi-GPUs training
        if not args.distributed or args.rank == 0:
            # dumping all the configuration files into result_path for resuming
            with open(os.path.join(args.result_path, "exp_cfg.yaml"), 'w', encoding="utf-8") as f:
                yaml.dump(vars(args), f, sort_keys=False)
            with open(os.path.join(args.result_path, "data_cfg.yaml"), 'w', encoding="utf-8") as f:
                yaml.dump(data_cfg, f, sort_keys=False)
            with open(os.path.join(args.result_path, "train_cfg.yaml"), 'w', encoding="utf-8") as f:
                yaml.dump(train_cfg, f, sort_keys=False)

        # --- The environment initialization above is shared by both the training branch and testing branch --- #

        # Model training branch
        if args.train:
            # initialize the Optimizers
            assert "optim_sches" in train_cfg.keys(), "Please fill in the 'optim_sches' tag!"
            optim_sches = cls.build_optim_sches(model=model, optim_sche_cfg=train_cfg['optim_sches'], args=args)

            # logging the information of the optimschedulers
            for name, optim_sche in optim_sches.items():
                logger.info(f"The {name} OptimScheduler: {optim_sche}")

            # initialize the Monitor for training and validation
            if not args.distributed or args.rank == 0:
                train_monitor = TrainMonitor(logger=logger, args=args)
                valid_monitor = ValidMonitor(logger=logger, args=args, model=model)
            else:
                train_monitor = None
                valid_monitor = None

            # loading the model from the existing checkpoint for resuming the training process
            args.start_epoch = cls.resume(args=args, model=model, optim_sches=optim_sches,
                                          train_monitor=train_monitor, valid_monitor=valid_monitor)

            # DDP Wrapping of the model must be done after model loading
            if args.distributed:
                # turn the batchnorm layers into the sync counterparts
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)

            # start the training process
            cls.train(args=args, iterators=iterators, model=model, optim_sches=optim_sches,
                      logger=logger, train_monitor=train_monitor, valid_monitor=valid_monitor)

        # Model testing branch
        elif args.test:
            # load the target model parameters
            model.load_state_dict(
                torch.load(os.path.join(args.result_path, 'models', f'{args.test_model}.mdl'),
                           map_location=model.device)
            )

            # start the testing process
            cls.test(args=args, iterators=iterators, model=model)

        # release the computational resource in the multi-GPU training setting
        if args.distributed:
            torch.distributed.destroy_process_group()


    @classmethod
    def main(cls, args: argparse.Namespace):
        """
        The beginning of a branch (training or testing).
        This function decides the single-GPU or multi-GPU training branches.

        Args:
            args: argparse.Namespace
                The input arguments for the experiment.

        """

        # This block is for calling torch.cuda API in the main process for the single-GPU training setting
        try:
            torch.multiprocessing.set_start_method('spawn')
        except RuntimeError:
            pass

        # turn the specified GPUs into a list, they can be given by either arguments or command line
        # the argument 'gpu' has the higher priority than the command 'CUDA_VISIBLE_DEVICES'
        if args.gpus is not None:
            args.gpus = [int(gpu) for gpu in args.gpus.split(',')] if isinstance(args.gpus, str) else [args.gpus]
        elif 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
            args.gpus = os.environ['CUDA_VISIBLE_DEVICES']
            args.gpus = [idx for idx, _ in enumerate(args.gpus.split(','))] if isinstance(args.gpus, str) else [args.gpus]
        # automatically generate available GPUs
        else:
            args.gpus = sorted(GPUtil.getGPUs(), key=lambda g: g.memoryUtil)[:args.ngpu]
            # make sure that GPU no.0 is the first GPU if it is selected
            args.gpus = sorted([gpu.id for gpu in args.gpus])

        # check the GPU configuration
        assert len(args.gpus) >= args.ngpu, \
            "The visible GPUs (args.gpus) are fewer than the GPUs you would like to use (args.ngpu)!"
        if len(args.gpus) == 1:
            args.gpus = args.gpus[0]

        # get the world_size from the command line, world_size here means the number of nodes
        if args.dist_url == "env://" and args.world_size == -1:
            args.world_size = int(os.environ["WORLD_SIZE"])

        # distributed is set to true if multiple GPUs are specified or multiple nodes are specified
        args.distributed = args.world_size > 1 or args.ngpu > 1

        # multi-GPU distributed training and testing
        if args.ngpu > 1:
            # check whether the input number of GPUs is valid
            ngpus_per_node = torch.cuda.device_count()
            if args.ngpu > ngpus_per_node:
                warnings.warn(f"Your input args.ngpu {args.ngpu} is larger than the GPUs you have on your machine {ngpus_per_node}. "
                              f"Currently, the real args.ngpu becomes {ngpus_per_node}.")
                args.ngpu = ngpus_per_node
            # here world_size becomes the number of processes on all nodes
            args.world_size = args.ngpu * args.world_size

            # automatic port selection if no specified port (only one ':' in args.dist_url)
            if len(args.dist_url.split(':')) < 3:
                args.dist_url += f':{get_port()}'

            # run one process on each GPU
            mp.spawn(cls.main_worker, nprocs=args.ngpu, args=(args,))

        # single-GPU training and testing
        elif args.ngpu == 1:
            cls.main_worker(args.gpus, args)

        # CPU testing with the multiprocessing strategy
        elif args.test:
            raise NotImplementedError("Multiprocessing CPU testing function has not been implemented yet......")

        # CPU training is not supported
        else:
            raise RuntimeError("Our toolkit doesn't support CPU training. Please specify a number of GPUs......")


    @classmethod
    def run(cls):
        """
        The entrypoint of Runner.
        This function sorts up the configuration and decides the training or testing branches.

        """
        # obtain the input arguments
        args = cls.parse()

        # overwrite the configuration from the args.config
        # Note: args.config has the higher priority than the command line arguments
        if args.config is not None:
            config = yaml.load(open(args.config, mode='r', encoding='utf-8'))
            for c in config:
                setattr(args, c, config[c])

        # ToDo(heli-qi): The configuration should be refreshed by the new arguments entered in the terminal

        # start the experiment pipeline
        assert (args.train and args.test) is False, \
            "A runner job can only deal with either training or testing. " \
            "If you want to conduct training and testing sequentially, please use two runner jobs."
        cls.main(args)


if __name__ == '__main__':
    Runner.run()
