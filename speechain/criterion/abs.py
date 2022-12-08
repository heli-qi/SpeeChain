"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import torch

from abc import ABC, abstractmethod


class Criterion(ABC):
    """
    Criterion is a Callable object which is the base class for all criterion objects in this toolkit.
    It serves the role of evaluating the model forward calculation results.
    Its output can be either a loss function used for training or an evaluation metric used for validation.

    This base class has two abstract interface functions: criterion_init() for criterion initialization and __call__()
    for criterion forward calculation.
    1. __call__() must be overridden if you want to make your own Criterion implementation.
    2. criterion_init() is not mandatory to be overridden because some criteria can directly be applied to the input data
        without any initialization such as speechain.criterion.accuracy.Accuracy.

    """

    def __init__(self, **criterion_conf):
        """
        This initialization function is shared by all Criterion subclasses.
        Currently, the shared logic only contains calling the initialization function of the parent class.

        Args:
            **criterion_conf:
                The arguments used by criterion_init() for your customized Criterion initialization.
        """
        super(Criterion, self).__init__()
        self.criterion_init(**criterion_conf)

    def criterion_init(self, **criterion_conf):
        """
        Abstract interface function for customized initialization of each Criterion subclass.
        This interface function is not mandatory to be overridden by your implementation.

        Args:
            **criterion_conf:
                The arguments used for customized Criterion initialization.
                For more details, please refer to the docstring of your target Criterion subclass.

        """
        pass

    @abstractmethod
    def __call__(self, **kwargs):
        """
        This abstract interface function receives the model forward calculation results and ground-truth labels.
        The output is a scalar which could be either trainable for parameter optimization or non-trainable for
        information recording.

        This interface function is mandatory to be overridden by your implementation.

        Args:
            **kwargs:
                model forward calculation results and ground-truth labels.
                For more details, please refer to the docstring of __call__() of your target Criterion subclass.

        Returns:
            A trainable or non-trainable scalar.
            For more details, please refer to the docstring of __call__() of your target Criterion subclass.

        """
        raise NotImplementedError
