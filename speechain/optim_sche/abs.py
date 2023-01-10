"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import torch
import itertools
import warnings

from abc import ABC, abstractmethod
from typing import Dict, List, Any
from torch.cuda.amp import GradScaler
from contextlib import nullcontext

from speechain.model.abs import Model
from speechain.utilbox.import_util import import_class


class OptimScheduler(ABC):
    """
    OptimScheduler is the base class of all OptimScheduler objects that combine the roles of traditional optimizers and
    schedulers together. Its main job is optimizing the target model parameters and scheduling the learning rate during
    training.

    In this toolkit, we combine traditional optimizers and schedulers into a single class: OptimScheduler. Each
    OptimScheduler object has one built-in member optimizer (torch.optim.Optimizer) which is initialized automatically
    by `optim_type` and `optim_conf` given in your configuration.

    """

    def __init__(self,
                 optim_type: str,
                 optim_conf: Dict[str, Any],
                 model: Model,
                 distributed: bool = False,
                 optim_loss: str = None,
                 updated_modules: List[str] = None,
                 step_per_update: int = 1,
                 use_amp: bool = True,
                 accum_grad: int = 1,
                 ft_factor: float = 1.0,
                 grad_clip: float = 1.0,
                 grad_norm_type: float = 2.0,
                 **sche_conf):
        """
        This initialization function initializes the general part shared by all OptimScheduler subclasses.
        At the end of this function, an interface function `sche_init()` is called to initialize the customized part of
        each OptimScheduler subclass.

        Args:
            # --- Arguments received from exp_cfg --- #
            model: speechain.model.abs.Model
                The pointer to the model whose parameters will be optimized by the built-in `torch.optim.Optimizer`.
            distributed: bool = False
                Whether the model to be optimized is distributed to multiple GPUs.
                If True, gradient accumulation will be done asynchronously in the DDP mode to speed up training.
            use_amp: bool = True
                Whether the Automatic Mixed Precision (AMP) technique is used during back-propagation.  
                If True, a built-in `torch.cuda.amp.GradScaler` will be initialized to calculate the gradients.
            accum_grad: int = 1
                The number of steps to accumulate gradients before optimization.
                The larger this argument is, the larger your virtual batches will be.
            ft_factor: float = 1.0
                The finetuning factor used to scale down the learning rates during training.
            # --- Arguments received from train_cfg --- #
            optim_type: str
                The optimizer query used to pick up the target Optimizer subclass from `torch.optim`
            optim_conf: Dict
                The optimizer configuration used to initialize the optimizer
            optim_loss: str = None
                The name of the target loss used in this _OptimScheduler_ object to calculate the gradients.
                If not given, the loss named `loss` will be used for optimization.
            updated_modules: str or List[str] = None
                This argument allows you to update only a part of parameters of the built-in model pointer.
                `updated_modules` indicate the names of your target modules (first-level module in the nested module
                tree) in the built-in model pointer.
                Its value can be either a string (only one target module) or a list (multiple target modules).
                If not given, the entire model will be updated.
            step_per_update: int = 1
                The optimization interval for the built-in optimizer.
                It means that the parameter optimization will be done once every `step_per_update` steps.
            **sche_conf:
                The arguments used to initialize the customized part of this OptimScheduler.
                Mainly used to decide the learning rate scheduling strategy.
        """
        # initialize the general part of the scheduler
        assert (isinstance(accum_grad, int) and accum_grad >= 1) and \
               (isinstance(step_per_update, int) and step_per_update >= 1), \
            f"Both of accum_grad and step_per_update should be an integer equal to or larger than 1, " \
            f"but got accum_grad={accum_grad} and step_per_update={step_per_update}."
        self.model = model
        self.distributed = distributed

        # gradient-related arguments (loaded from exp_cfg)
        self.accum_grad = accum_grad
        self.grad_clip = grad_clip
        self.grad_norm_type = grad_norm_type
        self.ft_factor = ft_factor

        # optimization-related arguments (loaded from train_cfg)
        assert isinstance(optim_loss, str) or optim_loss is None, \
            "Your input optim_loss must be a single string or None! If it's not given, the loss named 'loss' will be " \
            "used for optimization; If it's given as a string, the loss whose name matches your given string will be " \
            "used for optimization."
        self.optim_loss = optim_loss
        self.step_per_update = step_per_update

        # specific parameters are updated
        if updated_modules is not None:
            self.updated_modules = updated_modules if isinstance(updated_modules, List) else [updated_modules]
            _updated_modules = [self.model.__getattr__(module).parameters() for module in self.updated_modules]
            params = itertools.chain(*_updated_modules)
        # all parameters of the model are returned
        else:
            self.updated_modules = None
            params = self.model.parameters()

        # initialize the optimizer part
        optim_class = import_class('torch.optim.' + optim_type)
        self.optimizer = optim_class(params=params, **optim_conf)

        # Initialize the gradient scaler for AMP training
        self.scaler = GradScaler() if use_amp else None

        # initialize the customized part of the scheduler
        self.sche_init(**sche_conf)

    @abstractmethod
    def sche_init(self, **sche_conf):
        """
        This abstract interface function is the customized initialization function which decides how the learning rate
        is scheduled as the training goes.
        This interface is mandatory to be overridden.

        Args:
            **sche_conf: Dict
                The arguments used to initialize the customized part of this OptimScheduler.
                For more details about the learning rate scheduling strategy, please refer to the docstring of
                `sche_init()` of your target OptimScheduler subclass.

        """
        raise NotImplementedError

    def step(self, losses: Dict[str, torch.Tensor], time_func, optim_name: str, step_num: int, logger = None):
        """
        This function optimizes the target parameters of the built-in model pointer with the input training losses.

        Args:
            losses: Dict[str, torch.Tensor]
                The training loss Dict received from the `criterion_forward()` of the bulit-in model pointer.
            time_func:
                The context function used to record the consumed time during gradient back-propagation and parameter
                optimization.
            optim_name: str
                The name of the OptimScheduler object. This argument is used to identify the recorded consumed time
                information.
            step_num: int
                The number of the current training step.
                This argument is used to update the learning rate for the current step by `self.update_lr()`.
            logger:
                Lazily passed logger object. Used to record logging information during optimization.

        """
        # --- 0. Initial Preparation Part --- #
        # get the real step number based on accum_grad
        real_step = (step_num - 1) // self.accum_grad + 1

        # context function used when doing the loss backward for efficient gradient accumulation in the DDP mode
        backward_context = self.model.no_sync if self.distributed and step_num % self.accum_grad != 0 \
            else nullcontext

        # --- 1. Loss Backward Part --- #
        with time_func(["loss_backward_time", optim_name]):
            # back-propagate the loss only when the real step number meets the updating interval
            if real_step % self.step_per_update == 0:
                # pick up the target training loss
                if self.optim_loss is None:
                    assert 'loss' in losses.keys(), \
                        f"In this toolkit when optim_loss is set to None, the optimizer will automatically optimize " \
                        f"the input loss named 'loss'. Therefore, please name one training loss in the returned Dict " \
                        f"of your criterion_forward() implementation as 'loss'."
                    loss = losses['loss']
                else:
                    loss = losses[self.optim_loss]

                with backward_context():
                    # average the loss for accumulation
                    loss /= self.accum_grad
                    # backward the loss in either the amp mode or the normal mode
                    self.scaler.scale(loss).backward() if self.scaler is not None else loss.backward()

        # --- 2. Model Optimization Part --- #
        with time_func(["optim_time", optim_name]):
            # do optimization only when the real step number meets the updating interval
            if real_step % self.step_per_update == 0:
                # update the learning rate for the current step (scaled by the finetuning factor)
                curr_lr = self.update_lr(real_step=real_step)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.ft_factor * curr_lr

                # update the model parameters if the accumulation interval is met
                if step_num % self.accum_grad == 0:
                    # unscale the gradients in advance to enable gradient clipping in the amp setting
                    # refer: https://pytorch.org/docs/1.10/notes/amp_examples.html#working-with-unscaled-gradients
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)

                    # apply the gradient clipping right before updating the target parameters
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.grad_clip,
                        norm_type=self.grad_norm_type,
                    )

                    # optimize the target parameters only when the values of gradients are not infinite
                    if not torch.isfinite(grad_norm):
                        if logger is not None:
                            logger.info("The grad_norm in the current step is infinite! "
                                        "The parameters are not updated in this step.")
                        if self.scaler is not None:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                    else:
                        if self.scaler is not None:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()

                    # Turn the gradients of the target parameters of this optimizer to zero right after optimization
                    self.optimizer.zero_grad()

    @abstractmethod
    def update_lr(self, real_step: int) -> float:
        """
        This abstract interface function generates the learning rate by the input step number.

        Args:
            real_step: int
                The number of the real step for parameter optimization. Due to the existence of `self.accum_grad`,
                parameter optimization may not be done at each training step. The real step number here means the
                training steps where parameter optimization is done.

        Returns: float
            The learning rate used for parameter optimization in the current training step.

        """
        raise NotImplementedError

    def get_lr(self):
        """
        This function returns the current learning rate of the built-in `torch.optim.Optimizer` member.

        Returns: float
            The value of the learning rates obtained from `self.optimizer.param_groups`.

        """
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self) -> Dict:
        """
        This function returns the current status of the OptimScheduler object for checkpoint storage.

        Returns: Dict
            The status Dict containing the current status of the built-in `torch.optim.Optimizer` and the built-in
            `torch.cuda.amp.GradScaler` (if had).

        """
        return dict(
            optimizer=self.optimizer.state_dict(),
            scaler=self.scaler.state_dict() if self.scaler is not None else None
        )

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        This function loads the existing checkpoint information into the _OptimScheduler_ object as the starting status.

        Args:
            state_dict: Dict
                The status information loaded from the existing checkpoint.

        """
        # load the optimizer
        self.optimizer.load_state_dict(state_dict['optimizer'])
        # load the gradient scaler
        if state_dict['scaler'] is not None:
            self.scaler.load_state_dict(state_dict['scaler'])

    def __repr__(self):
        """
        This function returns the description string of the _OptimScheduler_ object.
        There is a general description part shared by all the _OptimScheduler_ subclasses.

        In this function, an interface hook function `extra_repr_fn()` will be called to generate the specific
        description part of each _OptimScheduler_ subclass.

        Returns: str
            The description string for the OptimScheduler object.

        """
        return f"{self.__class__.__name__}(" \
               f"optimizer={self.optimizer.__class__.__name__}, " \
               f"optim_loss={self.optim_loss}, " \
               f"updated_modules={self.updated_modules}, " \
               + self.extra_repr_fn() + ")"

    def extra_repr_fn(self) -> str:
        """
        This interface hook function returns the specific part of the description string of the OptimScheduler object.
        The original implementation in the base class returns an empty string.

        In principle, this interface hook function must be overridden by each _OptimScheduler_ subclass.
        But there won't be any errors if you don't override it in your implementation.

        Returns: str
            The specific part of the description string of the OptimScheduler object.


        """
        return ""
