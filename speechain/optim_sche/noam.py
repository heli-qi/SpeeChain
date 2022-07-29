"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import torch
from speechain.optim_sche.abs import OptimScheduler

class Noamlr(OptimScheduler):
    """
    The OptimScheduler that schedule the learning rate as the way in
         Speech-transformer: a no-recurrence sequence-to-sequence model for speech recognition
            https://ieeexplore.ieee.org/abstract/document/8462506/

    This OptimScheduler is mainly used for Transformer-based models.

    """
    def sche_init(self,
                  d_model: int,
                  warmup_steps: int = 4000):
        """
        The scheduler in 'Attention is all you need'
        https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

        Args:
            d_model: int
                The dimension of the hidden vectors of your Transformer model.
            warmup_steps: int
                The number of warming up steps.

        Returns:
            A list of names of your customized member variables.

        """
        # para recording
        self.d_model = d_model
        self.init_lr = d_model ** (-0.5)
        self.warmup_steps = warmup_steps


    def update_lr(self, real_step: int):
        """

        Args:
            real_step: int
                The number of the current training step.
                Will be different from self.step_num when self.accum_grad is layer than 1.

        """
        # the learning rate of the current step for the optimizer
        return self.init_lr * min(real_step ** (-0.5), real_step * (self.warmup_steps ** (-1.5)))


    def __repr__(self):
        return f"{self.__class__.__name__}(" \
               f"optimizer={self.optimizer.__class__.__name__}, " \
               f"optim_losses={self.optim_losses}, " \
               f"updated_modules={self.updated_modules}, " \
               f"d_model={self.d_model}, " \
               f"warmup_steps={self.warmup_steps})"