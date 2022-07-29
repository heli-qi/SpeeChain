"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import torch

from abc import ABC, abstractmethod

class Criterion(torch.nn.Module, ABC):
    """
    Criterion is the base class for all criterion objects in this toolkit.

    In principle, this base class is not necessary and all criterion objects can directly inherit torch.nn.Module.
    However, if we want to initialize member variables in our criteria, we need to type super(Criterion, self).__init__()
    every time when we make a new criterion. It's very troublesome, so I make this base class.

    This base class has only one abstract interface functions: forward() for output calculation. It must be overrode
    if you want to make your own criteria. criterion_init() is not mandatory to be overrode because some criteria can
    be applied to the input without any initialization such as Accuracy.

    """
    def __init__(self, **criterion_conf):
        super(Criterion, self).__init__()
        self.criterion_init(**criterion_conf)

    def criterion_init(self, **criterion_conf):
        pass

    @abstractmethod
    def forward(self, **kwargs):
        raise NotImplementedError