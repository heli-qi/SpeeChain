import math
from speechain.optim_sche.abs import OptimScheduler

class ExponentDecayLr(OptimScheduler):
    """
        ExponentDecayLr is a class that inherits from OptimScheduler and implements an exponential decay learning rate
        scheduler. It updates the learning rate based on the epoch number using the provided decay factor.
    """
    def sche_init(self, decay_factor: float = 0.999):
        """
        Initializes the exponential decay learning rate scheduler with the given decay factor.

        Args:
            decay_factor (float):
                The decay factor that will be used to exponentially decay the learning rate. Defaults to 0.999.
        """
        self.decay_factor = decay_factor

    def update_lr(self, real_step: int, epoch_num: int) -> float:
        """
        Updates the learning rate based on the current epoch number and the decay factor.

        Args:
            real_step (int):
                The current step in the optimization process.
                Not used in this implementation, but required for compatibility with the base class.
            epoch_num (int):
                The current epoch number.

        Returns:
            float: The updated learning rate.
        """
        return self.get_lr() * pow(self.decay_factor, epoch_num - 1)

    def extra_repr_fn(self) -> str:
        """
        Returns a string representation of the ExponentDecayLr object, including the decay factor.

        Returns:
            str: A string containing the class name and the decay factor value.
        """
        return f"decay_factor={self.decay_factor}"
