"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import argparse
import os

import torch
import copy

from typing import Dict
from abc import ABC, abstractmethod
from collections import OrderedDict

from speechain.utilbox.yaml_util import load_yaml


class Model(torch.nn.Module, ABC):
    """
    Model is the base class for all models in this toolkit. The main job of the model is processing the
    input batch data, outputing the model prediction results, and evaluating the results by itself. Each model has several
    built-in Module members that make up of the main body of the model. These modules will be initialized
    automatically by the module_conf given in your configuration.

    There are a built-in dictionary and a built-in list in this class: init_class_dict and default_init_modules.
    init_class_dict contains all the available initialization method of this class while default_init_modules
    includes the network layers that have their own initialization method.

    """

    # available parameter initialization method
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
        torch.nn.BatchNorm2d
    ]

    def __init__(self,
                 model_conf: Dict,
                 module_conf: Dict,
                 args: argparse.Namespace,
                 device: torch.device,
                 criterion_conf: Dict = None):
        """
        Initialization function of Model. The main body of the model is initialized by self.model_init() and
        the criteria (loss functions and metrics) are initialized by self.criterion_init().

        The codes of this function is fixed and can be done automatically with your given configuration, so we don't
        recommend you to override this function. However, if these codes really cannot satisfy your,
        we would appreciate it a lot if you could make an issue and let us know.

        Args:
            model_conf: Dict
                General configuration of the model
            module_conf: Dict
                Module configuration
            criterion_conf: Dict
                Criterion configuration
            args: argparse.Namespace

            device: torch.device
        """
        assert model_conf is not None, "model_conf cannot be None!"
        assert module_conf is not None, "module_conf cannot be None!"

        super(Model, self).__init__()

        # general arguments
        self.non_blocking = args.non_blocking
        self.distributed = args.distributed
        self.device = device

        # model snapshotting-related arguments
        self.result_path = args.result_path
        if "visual_infer_conf" in model_conf.keys():
            # configuration is given as a .yaml file
            if isinstance(model_conf["visual_infer_conf"], str):
                self.visual_infer_conf = load_yaml(open(model_conf["visual_infer_conf"]))
            # configuration is explicitly given
            elif isinstance(model_conf["visual_infer_conf"], Dict):
                self.visual_infer_conf = model_conf["visual_infer_conf"]
            else:
                raise RuntimeError
        else:
            self.visual_infer_conf = dict()

        # initialize the model
        criterion_conf = dict() if criterion_conf is None else criterion_conf
        self.model_init(model_conf=model_conf, criterion_conf=criterion_conf, module_conf=module_conf)


    def model_init(self, model_conf: Dict, criterion_conf: Dict, module_conf: Dict):
        """
        The initialization function for the main body of the model. The initialization is done automatically by the
        input module configuration.

        The tag name of each module is used as the name of the corresponding built-in member. The value of the
        sub-tag 'type' is used as the query to pick up the target Module class. Then, the values
        of the sub-tag 'conf' will be fed into the chosen class to initialize the target module.

        Pretrained parameters will be loaded into your model if given. multiple sources of pretrained parameters can
        be given and each one can be loaded into specific parts of your model. The mismatch between the names of
        pretrained parameters and the target parts of your model can be solved by the 'mapping' tag in each source of
        pretrained parameters. The name of the pretrained parameters is the key while the name of the target part is
        the value as: 'mapping: src_name: tgt_name ...'

        If no pretrained parameter is given, the parameters of your model will be initialized by the method that
        matches your input query 'init'. Then, the initialization method in self.init_class_dict will be selected if
        its key fit your query. If no initialization query is given, torch.nn.init.xavier_normal_ will be used to
        initialize your model.

        Finally, the specific parts of your model will be frozen if 'frozen_modules' is given. If there is only one
        frozen module, you can only give one string to 'frozen_modules' as 'frozen_modules: xxx'; if there are
        multiple required modules, you need to give a list as
        'frozen_modules:
          - xxx
          - xxx
          ...'
        Moreover, the frozen grain varies from entire module to specific layers according to your input. For example,
        if you give 'frozen_modules: encoder_prenet', then all parameters of the encoder_prenet will be frozen; if you
        give 'frozen_modules: encoder_prenet.conv', then only the convolution layers of the encoder_prenet will be frozen.
        You can even freeze only the first convolution layer by giving 'frozen_modules: encoder_prenet.conv.0'.

        Args:
            model_conf: Dict
                General configuration of the model. Used for parameter initialization and freezing.
            module_conf: Dict
                Module configuration.

        """
        # --- Model Construction --- #
        self.bad_cases_selection = None
        if 'customize_conf' not in model_conf.keys():
            model_conf['customize_conf'] = dict()
        self.model_construction(**module_conf, **criterion_conf, **model_conf['customize_conf'])


        # --- Pretrained Model Loading --- #
        pretrained_model = model_conf['pretrained_model'] if 'pretrained_model' in model_conf.keys() else None
        if pretrained_model is not None:
            pretrained_model = pretrained_model if isinstance(pretrained_model, list) else [pretrained_model]

            for ptm in pretrained_model:
                assert 'path' in ptm.keys() and os.path.exists(ptm['path']), \
                    f"The pretrained model path {ptm['path']} doesn't exist! Please check the input path."
                _pt_model = torch.load(ptm['path'], map_location=self.device)

                mapping = ptm['mapping'] if 'mapping' in ptm.keys() else None
                if mapping is None:
                    self.load_state_dict(_pt_model)
                else:
                    assert isinstance(mapping, dict) and len(mapping) >= 1, \
                        f"mapping must be given as a dict and cannot be empty! " \
                        f"Got type(mapping)={type(mapping)} and len(mapping)={len(mapping)}"

                    _src_modules = OrderedDict()
                    for src, tgt in mapping.items():
                        # . at the tails is for making the name unique
                        src, tgt = src + '.', tgt + '.'
                        for name, para in _pt_model.items():
                            if name.startswith(src):
                                name = name.replace(src, tgt)
                                _src_modules[name] = para
                    self.load_state_dict(_src_modules)

        # --- Model Parameter Initialization --- #
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


        # --- Model Parameter Freezing --- #
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
    def model_construction(self, **kwargs):
        """

        Args:
            **kwargs:

        Returns:

        """
        raise NotImplementedError


    def batch_to_cuda(self, data: Dict or torch.Tensor):
        """
        The function that puts the processed batch data onto GPUs

        Args:
            data: Dict or torch.Tensor
                The input batch data. Each element is a Dict where each key-value pair represents an input.

        Returns:
            A Dict that contains the GPU data.

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

        Args:
            batch_data: Dict
                The input batch data.
            epoch: int

        Returns:
            The loss functions will be returned in the training phase.
            The evaluation metrics will be returned in the validation phase.

        """
        # --- Batch Data Preparation --- #
        # preprocess the batch data if needed
        batch_data = self.batch_preprocess(batch_data)

        # put the batch data onto GPUs
        batch_data = self.batch_to_cuda(batch_data)
        # --- data preparation above is shared by all the three branches: training, validation, and visualization --- #


        # --- Model Visualization Branch --- #
        # if there are additional arguments other than batch_data and epoch, the visualization branch is activated
        if len(kwargs) != 0:
            return self.visualize(epoch=epoch, **batch_data, **kwargs)


        # --- Model Forward Calculation --- #
        # Feed the input batch into the model and get the outputs, copy.deepcopy() here is for the data safety
        model_outputs = self.model_forward(epoch=epoch, **copy.deepcopy(batch_data))

        # copy.deepcopy() cannot receive the non-leaf nodes in the computation graph (model_outputs).
        # Since model_outputs cannot be detached from the graph (gradients necessary), copy.deepcopy() is not used below.
        def combine_input_output(_batch_data: Dict, _model_outputs: Dict):
            combination = dict()
            # if the input batch data is in the form of Dict, it means there are multiple dataloaders
            if isinstance(_batch_data[list(_batch_data.keys())[0]], Dict):
                combination.update(
                    batch_data=_batch_data,
                    model_outputs=_model_outputs
                )
            # if the input batch data is in the form of Tensor, it means there is only one dataloader.
            else:
                combination.update(_batch_data)
                combination.update(_model_outputs)
            return combination
        # --- model forward is shared by both the training and validation branches --- #


        # --- Model Training Branch --- #
        if self.training:
            # In the training stage, both the trainable losses and non-trainable metrics will be returned
            losses, metrics = self.loss_calculation(**combine_input_output(batch_data, model_outputs))
            # the non-trainable metrics will be averaged across all the processes in the distributed mode
            if self.distributed:
                metrics = self.aver_metrics_across_procs(metrics, batch_data)
            return losses, metrics

        # --- Model Validation Branch --- #
        else:
            # In the validation stage, only the non-trainable metrics will be returned
            metrics = self.metrics_calculation(**combine_input_output(batch_data, model_outputs))
            # the non-trainable metrics will be averaged across all the processes in the distributed mode
            if self.distributed:
                metrics = self.aver_metrics_across_procs(metrics, batch_data)
            return metrics


    def batch_preprocess(self, batch_data: Dict):
        """
        As for the returned Dict of this function, please note that the key names should match the ones you use in
        self.model_forward().

        Args:
            batch_data:

        Returns:

        """
        return batch_data


    def aver_metrics_across_procs(self, metrics: Dict[str, torch.Tensor], batch_data: Dict) -> Dict[str, torch.Tensor]:
        """
        This function averages the input metrics across all processes in the multi-GPU distributed training setting.
        This function doesn't need to be overridden if you are doing single-dataloader supervised training.
        For multi-dataloader training & validation scheme, you need to override it to fit your model.

        Args:
            metrics:
            batch_data:

        Returns:

        """
        # check the batch size
        batch_size = None
        for key, value in batch_data.items():
            assert isinstance(value, torch.Tensor), \
                "The default aver_metrics_across_procs() function is only designed for single-dataloader training. " \
                "You need to override this function if you want to conduct the multi-dataloader training."
            if batch_size is None:
                batch_size = value.size(0)
            else:
                assert value.size(0) == batch_size
        batch_size = torch.tensor([batch_size], dtype=torch.long, device=self.device)

        # sum up all the weighed metrics at rank no.0
        for key in metrics.keys():
            # each metric should be one-dimensional scalar
            if metrics[key].dim() == 0:
                metrics[key] = metrics[key][None]
            elif metrics[key].dim() != 1:
                raise RuntimeError

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
    def model_forward(self, **batch_data):
        """
        As for the returned Dict of this function, please note that the key names should match the ones you use in
        self.loss_calculation() and self.metrics_calculation()

        Args:
            **batch_data:

        Returns:

        """
        raise NotImplementedError


    @abstractmethod
    def loss_calculation(self, **batch_data) -> (Dict[str, torch.Tensor], Dict[str, torch.Tensor]):
        """
        As for the returned Dict of this function, please note that the key names should match the ones you use in
        the argument 'optim_losses' of yout OptimScheduler.

        Args:
            **batch_data:

        Returns:

        """
        raise NotImplementedError


    @abstractmethod
    def metrics_calculation(self, **batch_data) -> Dict[str, torch.Tensor]:
        """
        As for the returned Dict of this function, please note that the key names should match the ones you use in
        the argument 'best_model_metric' of the runner for training process monitoring.

        Args:
            **batch_data:

        Returns:

        """
        raise NotImplementedError


    @abstractmethod
    def visualize(self, epoch: int, sample_index: str, **valid_sample):
        raise NotImplementedError


    def evaluate(self, test_batch: Dict, infer_conf: Dict):
        """
        The function that is called in each testing step.

        Args:
            test_batch: Dict
                The input testing batch data
            infer_conf:

        Returns:
            A Dict of the evaluation results that you want to save to the disk.

        """
        # preprocess the batch data if needed
        test_batch = self.batch_preprocess(test_batch)

        # put the batch data onto GPUs
        test_batch = self.batch_to_cuda(test_batch)

        # return the inference results
        return self.inference(infer_conf, **test_batch)


    @abstractmethod
    def inference(self, infer_conf, **kwargs) -> Dict[str, Dict]:
        """
        This function receives the test data and test configuration. After obtaining the inference results,
        all the results will be packaged into a Dict[str, Dict] which is passed to TestMonitor.

        Each key-value item in the returned Dict corresponds to a file on the disk. The file name is decided by the
        key and the file content is decided by the value. Currently, three file types are supported in our toolkit:
        1. .txt files (include .md files)
        2. .wav files
        3. .flac files
        4. .npz files

        The value of each item must also be a Dict and there must be two items in this sub-Dict: 'format' and 'content'.
        'format' indicates the file type and 'content' is a List that contains the data to be saved to the file.

        Each file type has a specific rule of key-value format:
            1. For .txt files, the value of 'format' in the sub-Dict must be 'txt' and the value of 'content' must be
            made up of fundamental data type (e.g. int, float, str, bool, ...).
            And the file will be named by ('idx2' + first-level key) and each line of the file will be made up of
            (a test sample index + a blank + an element in the 'content' List). For example,
            >>> dict(cer=dict(format='txt', content=[0.1, 0.2, 0.3]))
            will create a file named 'idx2cer' and the file content will be
            >>> xxx-xxxx1 0.1
            >>> xxx-xxxx2 0.2
            >>> xxx-xxxx3 0.3
            Note: if the first-level key ends with '.md', there will not be 'idx2' attached at the front.

            2. For .wav and .flac files, the value of 'format' in the sub-Dict must be 'wav' or 'flac' and the value of
            'content' must be made up of array-like data type (e.g. numpy.ndarry, torch.Tensor, ...). Moreover,
            there must be a new key-value item named 'sample_rate' that indicates the sampling rate of the waveform to
            be saved.
            There will be a folder named by the first-level key that contains all the waveform files and a .txt file
            named by ('idx2' + first-level key) that contains the absolute paths of the saved waveform files.
            For example,
            >>> dict(wav=dict(format='flac', content=[np_arr1, np_arr2, np_arr3]))
            will create a folder named 'wav' and a .txt file named 'idx2wav'.
            The folder 'wav' is like:
            >>> xxx-xxxx1.flac
            >>> xxx-xxxx2.flac
            >>> xxx-xxxx3.flac
            And 'idx2wav' is like:
            >>> xxx-xxxx1 /x/xx/xxx/xxx-xxxx1.flac
            >>> xxx-xxxx2 /x/xx/xxx/xxx-xxxx2.flac
            >>> xxx-xxxx3 /x/xx/xxx/xxx-xxxx3.flac

            3. For .npz files, the value of 'format' in the sub-Dict must be 'npz' and the value of
            'content' must be made up of numpy.ndarry (torch.Tensor is not supported).
            In each .npz file, there will be two key-value items: 'feat' and 'index'. 'feat' contains your given
            numpy.ndarry and 'index' contains the sample index of the array (Currently, only 'feat' is useful while
            'index' is only for reference).
            There will be a folder named by the first-level key that contains all the .npz files and a .txt file
            named by ('idx2' + first-level key) that contains the absolute paths of the saved feature files.
            For example,
            >>> dict(feat=dict(format='npz', content=[np_arr1, np_arr2, np_arr3]))
            will create a folder named 'feat' and a .txt file named 'idx2feat'.
            The folder 'feat' is like:
            >>> xxx-xxxx1.npz
            >>> xxx-xxxx2.npz
            >>> xxx-xxxx3.npz
            And 'idx2feat' is like:
            >>> xxx-xxxx1 /x/xx/xxx/xxx-xxxx1.npz
            >>> xxx-xxxx2 /x/xx/xxx/xxx-xxxx2.npz
            >>> xxx-xxxx3 /x/xx/xxx/xxx-xxxx3.npz

        """
        raise NotImplementedError
