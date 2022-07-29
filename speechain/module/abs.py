"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import torch

from abc import ABC, abstractmethod

class Module(torch.nn.Module, ABC):
    """
    Model is the base class for all module objects in this toolkit.

    In principle, this base class is not necessary and all module objects can directly inherit torch.nn.Module.
    However, if we want to initialize member variables in our modules, we need to type super(Module, self).__init__()
    every time when we make a new module. It's very troublesome, so I make this base class.

    This base class has two abstract interface functions: module_init() and forward(). They must be overridden if you
    want to make your own modules. module_init() is for module initialization and forward() is for output calculation.

    There are two built-in variable members: input_size and output_size. These two variables are used to
    automatically construct a sequence of unit modules in a template module. input_size is the last dimension of the
    input tensors while output_size is the last dimension of the output tensors.

    You could utilize self.input_size in your module_init() to conveniently initialize your module and give the
    output data dimension to self.output_size. Note that the usage of these two variables is mandatory only when you
    want to initialize your unit module in a template module.

    """

    def __init__(self, input_size: int = None, **module_conf):
        """

        Args:
            input_size: int
            **module_conf:
        """
        super(Module, self).__init__()

        self.input_size = input_size
        self.output_size = None

        self.module_init(**module_conf)


    @abstractmethod
    def module_init(self, **module_conf):
        raise NotImplementedError


    @abstractmethod
    def forward(self, **kwargs):
        raise NotImplementedError