"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
from speechain.optim_sche.abs import OptimScheduler


class Noamlr(OptimScheduler):
    """
    The OptimScheduler where the scheduling contains a LR warmup stage and a LR decay stage.
    In the warmup stage, the learning rate increases linearly to the peak.
    In the decay stage, the learning rate decreases in the level of square root.

    This OptimScheduler is mainly used for Transformer-based models.

    """

    def sche_init(self,
                  d_model: int = None,
                  warmup_steps: int = 4000):
        """
        The learning rate calculation is different depending on whether d_model is given or not.

        If d_model is given, the learning rate would be:
            (d_model ** -0.5) * min(step ** -0.5, real_step * warmup_steps ** -1.5)
        This calculation method is the original method proposed in 'Attention is all you need'.

        If d_model is not given, the learning rate would be:
            (optimizer.lr * warmup_steps ** 0.5) * min(real_step ** -0.5, step * warmup_steps ** -1.5)
        This calculation method makes sure that the learning rate reaches the maximum (optimizer.lr) right after
        all the warmup steps are finished.

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
        self.init_lr = d_model ** -0.5 if d_model is not None else self.get_lr() * warmup_steps ** 0.5
        self.warmup_steps = warmup_steps

    def update_lr(self, real_step: int, epoch_num: int) -> float:
        """

        Args:
            real_step: int
                The number of the current training step.
                Will be different from self.step_num when self.accum_grad is layer than 1.

        """
        # the learning rate of the current step for the optimizer
        return self.init_lr * min(real_step ** -0.5, real_step * (self.warmup_steps ** -1.5))

    def extra_repr_fn(self) -> str:
        return f"d_model={self.d_model}, " \
               f"warmup_steps={self.warmup_steps}"
