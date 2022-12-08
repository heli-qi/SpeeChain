"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import torch

from typing import Dict
from abc import ABC, abstractmethod


class Module(torch.nn.Module, ABC):
    """
    Module is the base class for all Module objects in this toolkit. For all the Model objects in this toolkit, their
    neural networks are constructed by many Module objects in a nested structure.
    Below is an example of the ASR model:
    ```
    ASR (Model)
        ---> ASREncoder (Module)
            ---> Speech2MelSpec (Module)
                ---> Speech2LinearSpec (Module)
                ---> LinearSpec2MelSpec (Module)
            ---> Conv2dPrenet (Module)
                ---> LinearPrenet (Module)
            ---> TransformerEncoder (Module)
                ---> PositionalEncoding (Module)
                ---> MultiHeadedAttention (Module)
                ---> PositionwiseFeedForward (Module)
        ---> ASRDecoder (Module)
            ---> EmbedPrenet (Module)
            ---> TransformerDecoder (Module)
                ---> PositionalEncoding (Module)
                ---> MultiHeadedAttention (Module)
                ---> PositionwiseFeedForward (Module)
            ---> TokenPostnet (Module)
    ```

    This base class has two required abstract interface functions that must be overriden by all Module subclasses:
    module_init() and forward(). module_init() is for module initialization and forward() is for output calculation.

    """

    def __init__(self, input_size: int = None, distributed: bool = False, **module_conf):
        """
        This initialization function is shared by all _Module_ subclasses.

        There are two built-in variable members: `input_size` and `output_size`. `input_size` is the last dimension of
        the input tensor while `output_size` is the last dimension of the output tensor.

        These two member variables serve as the socket and plug that are used to communicate with the front and back
        Module objects in a Model object.

        You could utilize `self.input_size` in your `module_init()` implement to initialize your module and give the
        output data dimension to `self.output_size`.
        Note: The usage of these two member variables is not mandatory, but it would be a convenient way for you to
        initialize your module.

        Args:
            input_size: int = None
                The last dimension of the tensor from the front _Module_ object.
                If not given, this argument would be None.
            distributed: bool = False
                Whether the _Model_ object this _Module_ object is belong to is distributed to multiple GPUs.
            **module_conf:
                The arguments used by `module_init()` for your customized _Module_ initialization.
        """
        super(Module, self).__init__()

        # shared general members
        self.input_size = input_size
        self.output_size = None
        self.distributed = distributed

        # customized initialization
        self.module_init(**module_conf)

    @abstractmethod
    def module_init(self, **module_conf):
        """
        Abstract interface function for customized initialization of each _Module_ subclass.
        This interface function is mandatory to be overridden by your implementation.

        Args:
            **module_conf:
                The arguments used for customized Module initialization.
                For more details, please refer to the docstring of your target Module subclass.

        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, **kwargs):
        """
        This abstract interface function is the customized implementation of `torch.nn.Module.forward()` used during
        model forward calculation. This interface function is mandatory to be overridden by your implementation.

        Args:
            **kwargs:
                The input arguments for module forward calculation.
                For more details, please refer to the docstring of `forward()` of your target _Module_ subclass.

        Returns:
            Module forward calculation results.
            For more details, please refer to the docstring of `forward()` of your target _Module_ subclass.

        """
        raise NotImplementedError

    def recover(self, **kwargs):
        """
        This interface function is used to recover the module forward calculation results back to the input data.
        It can be considered as the reverse process of `forward()`.
        This interface function is not mandatory to be overridden.

        Args:
            **kwargs:
                The input forward calculation results to be recovered.
                For more details, please refer to the docstring of `recover()` of your target _Module_ subclass.

        Returns:
            The recovered data or closely-recovered data (sometimes `forward()` may not be totally recoverable).
            For more details, please refer to the docstring of `recover()` of your target _Module_ subclass.

        """
        raise NotImplementedError

    def reset_parameters(self):
        """
        This abstract interface function is used to initialize the customized parameters in the _Module_ subclass if
        had. Some _Module_ subclasses have their customized parameters with specific initialization functions.

        If your _Module_ implementation has some customized parameters and you want to initialize them by yourself,
        please give the initialization logic in this interface function.

        This interface function is not mandatory to be overridden.
        Note: Don't forget to add `self.default_init_modules.append(YourModule)` in `model_init()` of your _Model_.

        """
        raise NotImplementedError

    def get_recordable_para(self) -> Dict or None:
        """
        This function returns the parameters of the module that you want to record as part of step information.

        If you want to record the value of the customized parameters of your module:
        1. when it is a leaf (no _Module_ members) in the nested _Module_ tree of the model, please override this
            function and return the parameter values in a _Dict_.
            For an example, you can refer to [${SPEECHAIN_ROOT}/speechain/module/transformer/pos_enc.py]().
        2. when it is a non-leaf (with _Module_ members) in the nested _Module_ tree of the model, please follow the
        pseudocode below:
         >>> class YourModule(Module):
         ...   def get_recordable_para(self) -> Dict or None:
         ...      output = dict()
         ...      # add the value of your target parameter into the output as key-value items
         ...      output.update(super(YourModule, self).get_recordable_para())
         ...      return output

        Returns: Dict or None
            For the leaf module, the default implementation returns None;
            For the non-leaf module, the default implementation returns a Dict containing names and recordable
            parameters of its member modules.


        """
        # for the leaf module, the default implementation returns None
        if sum([isinstance(module, Module) for module in self._modules.values()]) == 0:
            return None
        # for the non-leaf module, return a Dict containing names and recordable parameters of its member modules
        else:
            return {name: module.get_recordable_para()
                    for name, module in self._modules.items() if isinstance(module, Module)}
