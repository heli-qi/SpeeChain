"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import torch
import itertools
import warnings
from torch.optim.lr_scheduler import _LRScheduler
from abc import ABC, abstractmethod
from typing import Dict, List, Any
from collections import OrderedDict
from torch.cuda.amp import GradScaler
from contextlib import nullcontext

from speechain.model.abs import Model
from speechain.utilbox.import_util import import_class

class OptimScheduler(ABC):
    """
    OptimScheduler is the base class for all optimscheduler in this toolkit. The main job of the
    optimscheduler is optimizing the target model parameters and scheduling the learning rate during training. In
    this toolkit, we combine traditional optimizers and schedulers into a single class. Each optimeduler has one
    built-in optimizer member (torch.optim.Optimizer) which is initialized automatically by the 'optim_conf' given in
    your configuration.

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

        Args:
            optim_type: str
                The optimizer query used to pick up the target torch.optim.Optimizer from optim_class_dict
            optim_conf: Dict
                The optimizer configuration used to initialize the optimizer
            model: Model
                The model to be update.
            optim_loss: str
                The target loss used in this OptimScheduler to calculate the gradients.
                The value must be a string (only one loss is used for optimization).
                None means the input loss during will be used.
            updated_modules: str or List[str]
                The target modules to be updated in this optimscheduler.
                The value can be either a string (only one module) or a list (multiple modules).
                None means the entire model will be updated.
            accum_grad: int
                The number of steps for gradient accumulation.
                Received from the runner by exp_cfg.
            step_per_update: int
                The updating interval for the built-in optimizer.
                The parameter updating will be done once every step_per_update steps.
            step_num: int
                The initial step number.
            **sche_conf: Dict
                The customized arguments to initialize the scheduler part of this optimeduler.
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
        assert isinstance(optim_loss, (str, type(None))), "Your input optim_loss must be a single string!"
        self.optim_loss = optim_loss
        self.step_per_update = step_per_update
        self.updated_modules = updated_modules if isinstance(updated_modules, (List, type(None))) else [updated_modules]
        # all parameters of the model are returned
        if self.updated_modules is None:
            params = self.model.parameters()
        # specific parameters are be updated
        else:
            _updated_modules = [self.model.__getattr__(module).parameters() for module in self.updated_modules]
            params = itertools.chain(*_updated_modules)

        # initialize the optimizer part
        optim_class = import_class('torch.optim.' + optim_type)
        self.optimizer = optim_class(params=params, **optim_conf)

        # Initialize the gradient scaler for AMP training
        self.scaler = GradScaler() if use_amp else None

        # initialize the customized part of the scheduler
        self.sche_init(**sche_conf)


    @abstractmethod
    def sche_init(self, **sche_conf) -> List[str]:
        """
        The initialization function where the scheduler part of the optimscheduler is initialized.
        Mainly decide how the learning rate adjustment strategy works as the training goes.

        Args:
            **sche_conf: Dict
                The customized arguments used to initialize the scheduler part of this optimeduler.

        """
        raise NotImplementedError


    def step(self, losses: Dict[str, torch.Tensor], time_func, optim_name: str, step_num: int):
        """
        The function that updates the target parameters of the model with the input losses.

        Args:
            losses: Dict
                The input losses received from the model.
            step_num: int
                The number of the current step number. Received from the runner.

        """
        # --- Initial Preparation Part --- #
        # get the real step number based on accum_grad
        real_step = (step_num - 1) // self.accum_grad + 1

        # context function used when doing the loss backward for efficient gradient accumulation in the DDP mode
        backward_context = self.model.no_sync if self.distributed and step_num % self.accum_grad != 0 \
            else nullcontext


        # --- Loss Backward Part --- #
        with time_func(["loss_backward_time", optim_name]):
            # back-propagate the loss only when the real step number meets the updating interval
            if real_step % self.step_per_update == 0:
                # pick up the target training loss
                if self.optim_loss is None:
                    loss_keys = list(losses.keys())
                    assert len(loss_keys) == 1, f"An optimizer can only deal with one loss, but get {loss_keys}."
                    loss = losses[loss_keys[0]]
                else:
                    loss = losses[self.optim_loss]

                with backward_context():
                    # average the loss for accumulation
                    loss /= self.accum_grad
                    # backward the loss in either the amp mode or the normal mode
                    self.scaler.scale(loss).backward() if self.scaler is not None else loss.backward()


        # --- Model Optimization Part --- #
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
                        warnings.warn("The grad_norm in the current step is infinite! "
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
    def update_lr(self, real_step: int):
        """
        The function where the learning rate is adjusted according to the input step number.

        Note that the input step number must the one of the real step. The real step number means the times of
        updating parameters. For example, if accum_grad > 1, the step_num received from the runner is not the real
        step number.

        Args:
            real_step: int
                The real step number

        """
        raise NotImplementedError


    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


    def state_dict(self):
        return dict(
            optimizer=self.optimizer.state_dict(),
            scaler=self.scaler.state_dict() if self.scaler is not None else None
        )


    def load_state_dict(self, state_dict: Dict[str, Any]):
        # load the optimizer
        self.optimizer.load_state_dict(state_dict['optimizer'])
        # load the gradient scaler
        if state_dict['scaler'] is not None:
            self.scaler.load_state_dict(state_dict['scaler'])


    @abstractmethod
    def __repr__(self):
        raise NotImplementedError
