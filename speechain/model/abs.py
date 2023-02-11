"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import argparse
import os
import torch
import copy
import numpy as np

from typing import Dict, List
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import nullcontext

from speechain.module.abs import Module
from speechain.module.transformer.pos_enc import PositionalEncoding
from speechain.module.prenet.spk_embed import SpeakerEmbedPrenet

from speechain.utilbox.yaml_util import load_yaml
from speechain.utilbox.md_util import get_list_strings
from speechain.utilbox.data_loading_util import parse_path_args
from speechain.utilbox.train_util import text2tensor_and_len, spk2tensor


class Model(torch.nn.Module, ABC):
    """
    Model is the base class for all models in this toolkit. The main job of a model includes:
        1. (optional) preprocess the input batch data to the trainable format
        2. calculate the model prediction results by the Module members
        3. evaluate the prediction results by the Criterion members

    Each model has several built-in Module members that make up the neural network structure of the model. These Module
    members will be initialized by the `module_conf` given in your configuration.

    There are a built-in dictionary named `init_class_dict` and a built-in list named `default_init_modules` in the
    base class. init_class_dict` contains all the available initialization functions of the model parameters while
    `default_init_modules` includes the network layers that have their own initialization functions.

    """

    # available parameter initialization functions
    init_class_dict = dict(
        xavier=torch.nn.init.xavier_normal_,
        xavier_normal=torch.nn.init.xavier_normal_,
        xavier_uniform=torch.nn.init.xavier_uniform_,
        kaiming=torch.nn.init.kaiming_normal_,
        kaiming_normal=torch.nn.init.kaiming_normal_,
        kaiming_uniform=torch.nn.init.kaiming_uniform_,
        uniform=torch.nn.init.uniform_,
        normal=torch.nn.init.normal_,
        zeros=torch.nn.init.zeros_
    )

    # some modules have their own parameter initialization methods
    default_init_modules = [
        torch.nn.Embedding,
        torch.nn.LayerNorm,
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        PositionalEncoding,
        SpeakerEmbedPrenet
    ]

    def __init__(self,
                 args: argparse.Namespace,
                 device: torch.device,
                 model_conf: Dict,
                 module_conf: Dict,
                 criterion_conf: Dict = None):
        """
        In this initialization function, there are two parts of initialization: model-specific customized initialization
        and model-independent general initialization.

        Model-specific customized initialization is done by two interface functions: module_init() and criterion_init().
        module_init() initializes the neural network structure of the model while criterion_init() initializes the
        criteria used to optimize (loss functions) and evaluate (validation metrics) the model.

        After the customized initialization, there are 3 steps for general initialization:
            1. Pretrained parameters will be loaded into your model if the key `pretrained_model` is given. Multiple
            pretrained models can be specified and each of them can be loaded into different parts of your model. The
            mismatch between the names of pretrained parameters and the parameters of your model is handled by the key
            'mapping'. The value of the key `mapping` is a dictionary where each key-value item corresponds to a mapping
            of parameter names. The key is the parameter name in the pretrained parameters while the value is the
            parameter name of your model.

            2. If `pretrained_model` is not given, the parameters of your model will be initialized by the function that
            matches your input query 'init'. Please refer to the built-in dictionary `init_class_dict` for the available
            initialization functions. If `init` is not given, the default initialization function
            `torch.nn.init.xavier_normal_` will be used to initialize your model.

            3. Finally, the specified parts of your model will be frozen if 'frozen_modules' is given. If there is only
            one frozen module, you can directly give the string of its name to 'frozen_modules' like
            'frozen_modules: {module_name}'; if there are multiple modules you want to freeze, you can give their names
            in a list as
            ```
            frozen_modules:
              - {module_name1}
              - {module_name2}
              - ...
            ```
            Moreover, the frozen granularity depends on your input `frozen_modules`.
            For example,
                1. If you give 'frozen_modules: encoder_prenet', all parameters of the prenet of your encoder will be
                frozen
                2. If you give 'frozen_modules: encoder_prenet.conv', only the convolution layers of the prenet of your
                encoder will be frozen
                3. If you give 'frozen_modules: encoder_prenet.conv.0', only the first convolution layer of the prenet
                of your encoder will be frozen
                4. If you give 'frozen_modules: encoder_prenet.conv.0.bias', only the bias vector of the first
                convolution layer of the prenet of your encoder will be frozen

        Args:
            args: argparse.Namespace
                Experiment pipeline arguments received from the `Runner` object in `runner.py`.
            device: torch.device
                The computational device used for model calculation in the current GPU process.
            model_conf: Dict
                The model configuration used for general model initialization.
            module_conf: Dict
                The module configuration used for network structure initialization.
            criterion_conf: Dict
                The criterion configuration used for criterion (loss functions and evaluation metrics) initialization.
        """
        super(Model, self).__init__()

        # input argument checking
        assert model_conf is not None, "model_conf cannot be None!"
        assert module_conf is not None, "module_conf cannot be None!"
        # criterion_conf is default to be an empty dictionary
        criterion_conf = dict() if criterion_conf is None else criterion_conf
        # customize_conf is default to be an empty dictionary
        if 'customize_conf' not in model_conf.keys():
            model_conf['customize_conf'] = dict()

        # general argument registration
        self.non_blocking = args.non_blocking
        self.distributed = args.distributed
        self.device = device

        # snapshotting-related argument registration
        self.result_path = args.train_result_path
        if "visual_infer_conf" in model_conf.keys():
            # configuration is given as a .yaml file
            if isinstance(model_conf["visual_infer_conf"], str):
                self.visual_infer_conf = load_yaml(open(parse_path_args(model_conf["visual_infer_conf"])))
            # configuration is explicitly given
            elif isinstance(model_conf["visual_infer_conf"], Dict):
                self.visual_infer_conf = model_conf["visual_infer_conf"]
            else:
                raise RuntimeError("model_conf['visual_infer_conf'] must be given as either a string or a Dict.")
        else:
            self.visual_infer_conf = dict()

        # --- 1. Model Construction --- #
        self.module_init(**module_conf, **model_conf['customize_conf'])
        self.criterion_init(**criterion_conf)
        # initialize the bad case selection methods by the hook function
        self.bad_cases_selection = self.bad_cases_selection_init_fn()

        # --- 2.1. Pretrained Model Loading --- #
        pretrained_model = model_conf['pretrained_model'] if 'pretrained_model' in model_conf.keys() else None
        if pretrained_model is not None:
            pretrained_model = pretrained_model if isinstance(pretrained_model, list) else [pretrained_model]

            for ptm in pretrained_model:
                # argument checking
                if isinstance(ptm, str):
                    ptm = dict(path=ptm)
                elif isinstance(ptm, Dict):
                    assert 'path' in ptm.keys(), \
                        "If model['model_conf']['pretrained_model'] is given as a Dict, " \
                        "please give a key named 'path' to specify where your pretrained model is placed."
                    if os.path.exists(ptm['path']):
                        raise RuntimeError(f"The specified path of your pretrained model {ptm['path']} doesn't exist! "
                                           f"Please check the input path.")
                else:
                    raise TypeError(f"The elements in model['model_conf']['pretrained_model'] must be either a string "
                                    f"or a Dict, but got {ptm}")

                _pt_model = torch.load(parse_path_args(ptm['path']), map_location=self.device)
                mapping = ptm['mapping'] if 'mapping' in ptm.keys() else None
                if mapping is None:
                    self.load_state_dict(_pt_model, strict=False)
                else:
                    assert isinstance(mapping, dict) and len(mapping) >= 1, \
                        f"mapping must be given as a dict and cannot be empty! " \
                        f"Got type(mapping)={type(mapping)} and len(mapping)={len(mapping)}"

                    _src_modules = OrderedDict()
                    # loop each name-parameter pair in the model
                    for name, para in _pt_model.items():
                        # loop each source-target mapping pair
                        for src, tgt in mapping.items():
                            # attach '.' to the end is for making the name unique
                            src, tgt = src + '.', tgt + '.'
                            # change the parameter name in the middle
                            if src in name:
                                name = name.replace(src, tgt)
                        # record the parameter no matter whether its name is modified or not
                        _src_modules[name] = para
                    self.load_state_dict(_src_modules, strict=False)

        # --- 2.2. Model Parameter Initialization --- #
        else:
            # the default initialization method is xavier (i.e. xavier_normal)
            init = model_conf["init"] if "init" in model_conf.keys() else 'xavier'
            assert init in self.init_class_dict.keys(), \
                f"Only the initialization methods {self.init_class_dict.keys()} are supported, but got init={init}."

            for name, para in self.named_parameters():
                # initialize all the bias vectors to zero
                if ".bias" in name and para.dim() == 1:
                    torch.nn.init.zeros_(para)
                # initialize all the weight vectors except for those of normalization layers (BatchNorm & LayerNorm)
                elif para.dim() > 1:
                    self.init_class_dict[init](para)

            # initialize the modules that have their own default init methods
            for module in self.modules():
                if isinstance(module, tuple(self.default_init_modules)):
                    module.reset_parameters()

        # --- 3. Model Parameter Freezing --- #
        frozen_modules = model_conf['frozen_modules'] if 'frozen_modules' in model_conf.keys() else None
        if frozen_modules is not None:
            frozen_modules = frozen_modules if isinstance(frozen_modules, list) else [frozen_modules]

            for name, para in self.named_parameters():
                frozen_flag = False
                for module in frozen_modules:
                    frozen_flag = name.startswith(module + '.')

                if frozen_flag:
                    para.requires_grad = False
                else:
                    raise RuntimeError(f"frozen_modules: Parameters of {name} are not found in the model!")

    @abstractmethod
    def module_init(self, **kwargs):
        """
        The interface function that initializes the Module members of the model. These Module members make up the
        neural network structure of the model. Some models have their customized part that also needs to be
        initialization in this function, e.g. the tokenizer of ASR and TTS models.

        Note: This interface function must be overridden for each Model subclass.

        Args:
            **kwargs:
                The combination of the arguments in your given `module_conf` and `model_conf['customize_conf']`.

        """
        raise NotImplementedError

    @abstractmethod
    def criterion_init(self, **criterion_conf):
        """
        The interface function that initializes the Criterion members of the model. These Criterion members can be
        divided into two parts: the loss functions used for training and the evaluation metrics used for validation.

        Args:
            **criterion_conf:
                The arguments in your given `criterion_conf`.
        Returns:

        """
        raise NotImplementedError

    @staticmethod
    def bad_cases_selection_init_fn() -> List[List[str or int]] or None:
        """
        This hook function returns the default bad case selection method of each Model object. This default value will
        be referred by the _Runner_ to present the top-N bad cases.

        The original hook implementation in the base Model class returns None which means no default value.

        Returns: List[List[str or int]]
            The returned default value should be a list of tri-list where each tri-list is in the form of
            [`selection_metric`, `selection_mode`, `case_number`]. For example, ['wer', 'max', 50] means 50 testing
            waveforms with the largest WER will be selected.

        """
        return None

    def batch_to_cuda(self, data: Dict[str, torch.Tensor] or torch.Tensor) -> Dict[str, torch.Tensor] or torch.Tensor:
        """
        The recursive function that transfers the batch data to the specified device in the current process.

        Args:
            data: Dict or torch.Tensor
                The input batch data. It should be either a Tensor or a Dict of Tensors. For the Dict input, the
                function itself will be called once by each Tensor element.

        Returns: Dict or torch.Tensor
            If the input is a Dict, the returned output will also be a Dict of Tensors transferred to the target device;
            If the input is a Tensor, the returned output will be its copy on the target device.

        """
        # if the data is in the form of Dict, recursively process each key-value pair
        if isinstance(data, Dict):
            return {key: self.batch_to_cuda(value) for key, value in data.items()}
        # if the data is in the form of tensor, put it on GPUs by .cuda()
        elif isinstance(data, torch.Tensor):
            return data.cuda(device=self.device, non_blocking=self.non_blocking)
        # do nothing for other types of data
        else:
            return data

    def forward(self, batch_data: Dict, epoch: int = None, **kwargs):
        """
        The general model forward function shared by all the _Model_ subclasses. This forward function has 3 steps:
            1. preprocess and transfer the batch data to GPUs
            2. obtain the model prediction results
            3. calculate the loss function and evaluate the prediction results

        For each step above, we provide interface functions for you to override and make your own implementation.

        Args:
            batch_data: Dict
                The input batch data received from the `train` or `valid` dataloader object in the experimental
                pipeline.
            epoch: int = None
                The number of the current epoch. Used for real-time model visualization and model prediction.
            **kwargs:
                The additional arguments for real-time model visualization. If given, the code will go through the model
                visualization branch.

        Returns:
            In the training branch, the loss functions and evaluation metrics will be returned each of which is in the
            form of a Dict.
            In the validation branch, only the evaluation metrics will be returned.
            In the visualization branch, the model snapshots on the given validation instance will be returned.

        """
        # --- 1. Batch Data Preprocessing and GPU transferring --- #
        # --- data preparation below is shared by all the three branches: training, validation, and visualization --- #
        # preprocess the batch data if needed
        batch_data = self.batch_preprocess_fn(batch_data)

        # put the batch data onto GPUs
        batch_data = self.batch_to_cuda(batch_data)

        # --- 2.1. Model Visualization Branch --- #
        # if there are additional arguments other than batch_data and epoch, the visualization branch is activated
        if len(kwargs) != 0:
            return self.visualize(epoch=epoch, **batch_data, **kwargs)

        # --- 2.2. Model Forward Calculation --- #
        # --- model forward is shared by both the training and validation branches --- #
        # context function used when doing the loss backward for efficient gradient accumulation in the DDP mode
        forward_context = nullcontext if self.training else torch.inference_mode
        with forward_context():
            # Feed the input batch into the model and get the outputs, copy.deepcopy() here is for the data safety
            model_outputs = self.module_forward(epoch=epoch, **copy.deepcopy(batch_data))

        # copy.deepcopy() cannot receive the non-leaf nodes in the computation graph (model_outputs). Since
        # model_outputs cannot be detached from the graph (gradients necessary), copy.deepcopy() is not used below.
        def combine_input_output(_batch_data: Dict, _model_outputs: Dict):
            combination, batch_keys = dict(), list(_batch_data.keys())
            # if the input batch data is in the form of Dict, it means there are multiple dataloaders
            if isinstance(_batch_data[batch_keys[0]], Dict):
                for key in batch_keys:
                    combination[key] = dict(**_batch_data[key], **_model_outputs[key])
            # if the input batch data is in the form of Tensor, it means there is only one dataloader.
            else:
                combination.update(_batch_data)
                combination.update(_model_outputs)
            return combination

        # --- 3.1. Model Training Branch --- #
        if self.training:
            # In the training stage, both the trainable losses and non-trainable metrics will be returned
            losses, metrics = self.criterion_forward(**combine_input_output(batch_data, model_outputs))
            metrics.update(self.get_recordable_para())

            # post-checking for training losses, they must be trainable tensors
            assert sum([isinstance(loss, torch.Tensor) and loss.requires_grad for loss in losses.values()]) \
                   == len(losses), "Training losses must be trainable tensors!"
            # post-checking for validation metrics, they must be either non-trainable tensors or other datatypes
            assert sum([not isinstance(metric, torch.Tensor) or not metric.requires_grad
                        for metric in metrics.values()]) == len(metrics), \
                "Validation metrics must be either non-trainable tensors or other datatypes!"

            # the non-trainable metrics will be averaged across all the processes in the distributed mode
            if self.distributed:
                metrics = self.aver_metrics_across_procs(metrics, batch_data)
            return losses, metrics

        # --- 3.2. Model Validation Branch --- #
        else:
            # In the validation stage, only the non-trainable metrics will be returned
            with torch.inference_mode():
                metrics = self.criterion_forward(**combine_input_output(batch_data, model_outputs))
            metrics.update(self.get_recordable_para())

            # post-checking for validation metrics, they must be either non-trainable tensors or other datatypes
            assert sum([not isinstance(metric, torch.Tensor) or not metric.requires_grad
                        for metric in metrics.values()]) == len(metrics), \
                "Validation metrics must be either non-trainable tensors or other datatypes!"

            # the non-trainable metrics will be averaged across all the processes in the distributed mode
            if self.distributed:
                metrics = self.aver_metrics_across_procs(metrics, batch_data)
            return metrics

    def batch_preprocess_fn(self, batch_data: Dict) -> Dict:
        """
        This hook function does the preprocessing for the input batch data before using them in self.model_forward().
        This function is not mandatory to be overridden and the original implementation in the base Model class does
        the tensor transformation for the string-like data in batch_data (i.e., text and spk_ids).

        Note: the key names in the returned Dict should match the argument names in self.model_forward().

        Args:
            batch_data: Dict
                The raw data of the input batch to be preprocessed in this hook function.

        Returns: Dict
            The processed data of the input batch that is ready to be used in `self.model_forward()`.

        """

        def process_strings(data_dict: Dict):
            """
            turn the text and speaker strings into tensors and get their lengths

            """
            # --- Process the Text String and its Length --- #
            if 'text' in data_dict.keys():
                assert isinstance(data_dict['text'], List)
                data_dict['text'], data_dict['text_len'] = text2tensor_and_len(
                    text_list=data_dict['text'], text2tensor_func=self.tokenizer.text2tensor,
                    ignore_idx=self.tokenizer.ignore_idx
                )

            # --- Process the Speaker ID String --- #
            if 'spk_ids' in data_dict.keys():
                assert isinstance(data_dict['spk_ids'], List) and hasattr(self, 'spk2idx')
                data_dict['spk_ids'] = spk2tensor(spk_list=data_dict['spk_ids'], spk2idx_dict=self.spk2idx)

            return data_dict

        # check whether the batch_data is made by multiple dataloaders
        leaf_flags = [not isinstance(value, Dict) for value in batch_data.values()]
        if sum(leaf_flags) == 0:
            return {key: process_strings(value) for key, value in batch_data.items()}
        elif sum(leaf_flags) == len(batch_data):
            return process_strings(batch_data)
        else:
            raise RuntimeError("Wrong composition of batch_data!")

    def aver_metrics_across_procs(self, metrics: Dict[str, torch.Tensor], batch_data: Dict) -> Dict[str, torch.Tensor]:
        """
        This function averages the evaluation metrics across all GPU processes in the DDP mode for model distribution.

        Args:
            metrics: Dict[str, torch.Tensor]
                The evaluation metrics to be averaged across all GPU processes.
            batch_data: Dict
                The input batch data used to calculate the batch size for averaging evaluation metrics.

        Returns: Dict[str, torch.Tensor]
            The evaluation metrics _Dict_ after averaging. The key names remain the same.

        """

        def get_batch_size(input_dict: Dict):
            _batch_size = None
            for value in input_dict.values():
                # len() considers all types of array: torch.Tensor, np.ndarray, List, ...
                if _batch_size is None:
                    _batch_size = len(value)
                else:
                    assert _batch_size == len(value)
            return _batch_size

        # check the batch size
        multi_flag = sum([isinstance(value, Dict) for value in batch_data.values()]) == len(batch_data)
        # we take the summation of all data-labels pairs in a single batch made by multiple dataloaders
        if multi_flag:
            batch_size = sum([get_batch_size(value) for value in batch_data.values()])
        else:
            batch_size = get_batch_size(batch_data)
        batch_size = torch.tensor([batch_size], dtype=torch.long, device=self.device)

        # sum up all the weighed metrics at rank no.0
        for key in metrics.keys():
            # each metric should be one-dimensional scalar
            if metrics[key].dim() == 0:
                metrics[key] = metrics[key][None]
            elif metrics[key].dim() != 1:
                raise RuntimeError(f"Each metric value must be one-dimensional scalar, "
                                   f"but got metrics[{key}]={metrics[key]}!")

            # batch_size acts as the weight for each metric value in the current process
            metrics[key] *= batch_size.type(metrics[key].dtype)
            # sum up the weighted metric values at rank no.0
            torch.distributed.reduce(metrics[key], dst=0, op=torch.distributed.ReduceOp.SUM)

        # sum up the batch size across at rank no.0 to get the overall batch size
        torch.distributed.reduce(batch_size, dst=0, op=torch.distributed.ReduceOp.SUM)
        if torch.distributed.get_rank() == 0:
            for key in metrics.keys():
                # turn the object value to the overall batch-level
                metrics[key] /= batch_size.type(metrics[key].dtype)

        return metrics

    @abstractmethod
    def module_forward(self, **batch_data) -> Dict:
        """
        This function forwards the input batch data by all _Module_ members.
        Note:
            1. This interface function must be overridden for each Model subclass.
            2. The argument names should match the key names in the returned Dict of `self.batch_preprocess_fn()`.
            3. The key names in the returned Dict should match the argument names of `self.loss_calculation()` and
            `self.metrics_calculation()`.

        Args:
            **batch_data:
                Processed data of the input batch received from `self.batch_preprocess_fn()`.

        Returns: Dict
            Prediction results (logits) of the model on the input batch data.
            Some intermediate results (e.g., attention matrices) can also be returned for later use.

        """
        raise NotImplementedError

    @abstractmethod
    def criterion_forward(self, **kwargs) -> \
            (Dict[str, torch.Tensor], Dict[str, torch.Tensor]) or Dict[str, torch.Tensor]:
        """
        This interface function is activated after `self.model_forward()`. It receives the model prediction results
        from `self.model_forward()` and input batch data from `self.batch_preprocess_fn()`.

        Args:
            **kwargs:
                The combination of the returned arguments from `self.batch_preprocess_fn()` and `self.model_forward()`.

        Returns: (Dict[str, torch.Tensor], Dict[str, torch.Tensor]) or Dict[str, torch.Tensor]
            The returned values should be different for the training and validation branches.
            1. For training, two Dict[str, torch.Tensor] should be returned where the first one contains all the
            trainable training losses for optimization and the second one contains all the non-trainable evaluation
            metrics used to record the training status.
            2. For validation, only one Dict[str, torch.Tensor] should be returned which contains all the non-trainable
            evaluation metrics used to record the validation status.

        """
        raise NotImplementedError

    def get_recordable_para(self) -> Dict[str, torch.Tensor]:
        """

        Returns:

        """
        def recur_get_module_recordable_para(curr_node, prefix_list: List[str] = None):
            if prefix_list is None:
                prefix_list = []
            if isinstance(curr_node, Dict):
                _output = dict()
                for _key, _value in curr_node.items():
                    _output.update(recur_get_module_recordable_para(_value, prefix_list + [_key]))
                return _output
            else:
                if curr_node is None:
                    return {}
                elif isinstance(curr_node, torch.Tensor):
                    return {'_'.join(prefix_list): curr_node.clone().detach()}
                else:
                    raise RuntimeError

        output = dict()
        for key, value in self._modules.items():
            if isinstance(value, Module):
                output.update(recur_get_module_recordable_para(value.get_recordable_para(), [key]))
        return output

    def matrix_snapshot(self, vis_logs: List, hypo_attention: Dict, subfolder_names: List[str] or str, epoch: int):
        """

        Used by the abstract function visualize() to make the snapshot materials for attention matrices.

        """
        if isinstance(subfolder_names, str):
            subfolder_names = [subfolder_names]
        keys = list(hypo_attention.keys())

        # process the input data by different data types
        if isinstance(hypo_attention[keys[0]], Dict):
            for key, value in hypo_attention.items():
                self.matrix_snapshot(vis_logs=vis_logs, hypo_attention=value,
                                     subfolder_names=subfolder_names + [key], epoch=epoch)

        # snapshot the information in the materials
        elif isinstance(hypo_attention[keys[0]], np.ndarray):
            vis_logs.append(
                dict(
                    plot_type='matrix', materials=hypo_attention, epoch=epoch,
                    sep_save=False, data_save=True, subfolder_names=subfolder_names
                )
            )

    def attention_reshape(self, hypo_attention: Dict, prefix_list: List = None) -> Dict:
        """

        Used by the abstract function visualize() to reshape the attention matrices before matrix_snapshot().

        """
        if prefix_list is None:
            prefix_list = []

        # process the input data by different data types
        if isinstance(hypo_attention, Dict):
            return {key: self.attention_reshape(value, prefix_list + [key]) for key, value in hypo_attention.items()}
        elif isinstance(hypo_attention, List):
            return {str(index - len(hypo_attention)): self.attention_reshape(
                hypo_attention[index], prefix_list + [str(index - len(hypo_attention))])
                for index in range(len(hypo_attention) - 1, -1, -1)}
        elif isinstance(hypo_attention, torch.Tensor):
            hypo_attention = hypo_attention.squeeze()
            if hypo_attention.is_cuda:
                hypo_attention = hypo_attention.detach().cpu()

            if hypo_attention.dim() == 2:
                return {'.'.join(prefix_list + [str(0)]): hypo_attention.numpy()}
            elif hypo_attention.dim() == 3:
                return {'.'.join(prefix_list + [str(index)]): element.numpy()
                        for index, element in enumerate(hypo_attention)}
            else:
                raise RuntimeError

    @abstractmethod
    def visualize(self, epoch: int, sample_index: str, **valid_sample):
        """

        Args:
            epoch:
            sample_index:
            **valid_sample:

        Returns:

        """
        raise NotImplementedError

    def evaluate(self, test_batch: Dict, infer_conf: Dict):
        """
        The shared evaluation function by all _Model_ subclasses. This evaluation function has 2 steps:
            1. preprocess and transfer the batch data to GPUs
            2. calculate the inference results

        For each step above, we provide interface functions for you to override and make your own implementation.

        Args:
            test_batch: Dict
                The input batch data received from the `test` dataloader object in the experimental pipeline.
            infer_conf: Dict
                The configuration used for model inference.

        Returns:
            A Dict of the inference results where each key-value item corresponds to one evaluation metric you want to
            save to the disk.

        """
        # preprocess the batch data if needed
        test_batch = self.batch_preprocess_fn(test_batch)

        # put the batch data onto GPUs
        test_batch = self.batch_to_cuda(test_batch)

        # get the inference results
        evaluate_results = self.inference(infer_conf=infer_conf, **test_batch)
        if hasattr(self, 'instance_report_cache') and self.instance_report_cache is not None:
            evaluate_results['instance_reports.md'] = self.instance_report_cache
            self.instance_report_cache = None

        # post-check the format of evaluate_results
        if isinstance(evaluate_results, Dict):
            for key, value in evaluate_results.items():
                if 'format' not in value.keys() or 'content' not in value.keys():
                    raise RuntimeError("Each element of the returned value of self.inference() must contain the keys "
                                       "named both 'format' and 'content'!")
        else:
            raise RuntimeError(f"The returned value of self.inference() must be a Dict, "
                               f"but got {type(evaluate_results)}!")
        return evaluate_results

    @abstractmethod
    def inference(self, infer_conf: Dict, **kwargs) -> Dict[str, Dict[str, str or List]]:
        """
        This function receives the test data and test configuration. The inference results will be packaged into a
        Dict[str, Dict] which is passed to TestMonitor for disk storage. The returned Dict should be in the form of
        ```
        dict(
            {file_name}=dict(
                format={file_format},
                content={file_content}
            )
        )
        ```
        The first-level key is used to decide the name of the meta file as `idx2{file_name}`. Its value is also a Dict
        and there must be two keys in this sub-Dict: 'format' and 'content'. The configuration of the sub-Dict is
        different for different file formats:

            1. For pure text metadata files, the value of 'format' must be 'txt' and the value of 'content' must be a
            list of Python built-in data type (i.e.,. int, float, str, bool, ...).
            Each line of the file `idx2{file_name}` will be made up of the index of a test data instance and its
            metadata value in the `content` List which are separated by a blank.
            For example,
            `dict(cer=dict(format='txt', content=[0.1, 0.2, 0.3]))` will create a pure text file named 'idx2cer' which
            looks like
            ```
            {test_index1} 0.1
            {test_index2} 0.2
            {test_index3} 0.3
            ```
            Note: if the first-level key ends with '.md', there will not be 'idx2' attached at the beginning of the
            file name.

            2. For audio files, the value of 'format' must be either 'wav' or 'flac' and the value of 'content' must be
            a list of array-like data type (e.g. numpy.ndarry, torch.Tensor, ...).
            Moreover, there must be an additional key named 'sample_rate' to indicate the sampling rate of the waveforms
            to be saved in audio files.
            There will be a folder named `{file_name}` that contains all the audio files and a pure text file named
            `idx2{file_name}` that contains the absolute paths of all the saved audio files.
            For example,
            `dict(wav=dict(format='flac', content=[np_arr1, np_arr2, np_arr3]))` will create a folder named 'wav' and
            a pure text file named 'idx2wav' in the same directory. The file 'idx2wav' looks like:
            ```
            {test_index1} /x/xx/wav/{test_index1}.flac
            {test_index2} /x/xx/wav/{test_index2}.flac
            {test_index3} /x/xx/wav/{test_index3}.flac
            ```
            where `/x/xx/` is your result path given in your `exp_cfg`.

            3. For binary files, the value of 'format' in the sub-Dict must be 'npy' and the value of 'content' must be
            a list of numpy.ndarry (torch.Tensor is not supported).
            There will be a folder named `{file_name}` that contains all the .npy files and a pure text file
            named `idx2{file_name}` that contains the absolute paths of all the saved binary files.
            For example,
            `dict(feat=dict(format='npy', content=[np_arr1, np_arr2, np_arr3]))`
            will create a folder named 'feat' and a pure text file named 'idx2feat'. The 'idx2feat' file is like:
            ```
            {test_index1} /x/xx/feat/{test_index1}.npy
            {test_index2} /x/xx/feat/{test_index2}.npy
            {test_index3} /x/xx/feat/{test_index3}.npy
            ```
            where `/x/xx/` is your result path given in your `exp_cfg`.

        """
        raise NotImplementedError

    def register_instance_reports(self, md_list_dict: Dict[str, List], extra_string_list: List[str] = None):
        """

        Args:
            md_list_dict:
            extra_string_list:

        Returns:

        """
        # --- 1. Arguments Checking --- #
        if extra_string_list is not None:
            assert isinstance(extra_string_list, List)

        ele_len = []
        for value in md_list_dict.values():
            assert isinstance(value, List)
            if extra_string_list is not None:
                assert len(value) == len(extra_string_list)
            ele_len.append(len(value))

        if len(set(ele_len)) == 1:
            ele_len = ele_len[0]
        else:
            raise RuntimeError

        # --- 2. Generate .md Instance Report for the current step --- #
        instance_reports = []
        for i in range(ele_len):
            ele_dict = {key: value[i] if isinstance(value[i], str) else str(value[i])
                        for key, value in md_list_dict.items()}
            _curr_report = '\n\n' + get_list_strings(ele_dict) + '\n'

            if extra_string_list is not None:
                _curr_report += extra_string_list[i] + '\n'
            instance_reports.append(_curr_report)

        self.instance_report_cache = dict(format='txt', content=instance_reports)
