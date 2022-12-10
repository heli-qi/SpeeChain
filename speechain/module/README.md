# Module

[*Module*](https://github.com/ahclab/SpeeChain/blob/main/speechain/module/abs.py) inherits `torch.nn.Module` and it is the base class for all _Module_ objects in this toolkit. 
The neural network parts of all the _Model_ objects in this toolkit are constructed by multiple _Module_ objects in a nested structure.  
Below is the nested _Module_ tree of an encoder-decoder ASR model:
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
This base class has two required abstract interface functions that must be overriden by all _Module_ subclasses: `module_init()` for module initialization and `forward()` for output calculation.

👆[Back to the handbook page](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#speechain-handbook)

## Table of Contents
1. [**Module Library**](https://github.com/ahclab/SpeeChain/tree/main/speechain/module#module-library)
2. [**API Document**](https://github.com/ahclab/SpeeChain/tree/main/speechain/module#api-document)

## Module Library
```
/speechain
    /module
        /abs.py             # Abstract class of Module. Base of all Module implementations.
        /frontend           # Acoustic feature extraction frontend modules
            /speech2linear.py   # Module implementation of speech-to-linear frontend. Used to transform the input speech waveforms into linear spectrogram.
            /linear2mel.py      # Module implementation of linear-to-mel frontend. Used to transform the input linear spectrogram into log-mel spectrogram.
            /speech2mel.py      # Module implementation of speech-to-mel frontend. Used to transform the input speech waveforms into log-mel spectrogram.
            /delta_feat.py      # Module implementation of delta frontend. Mainly used for ASR training when we want to take the first and second derivatives of log-mel spectrogram.
        /norm               # Normalization modules
            /feat_norm.py       # Module implementation of per-channel feature normalization.
        /augment            # Data augmentation modules
            /specaug.py         # Module implementation of SpecAugment. Mainly used for ASR training.
        /encoder            # Model encoder modules
            /asr.py             # Module implementation of ASR encoders. Used for ASR model construction.
            /tts.py             # Module implementation of TTS encoders. Used for TTS model construction.
        /decoder            # Model decoder modules
            /asr.py             # Module implementation of ASR autoregressive decoders. Used for autoregressive ASR model construction.
            /tts.py             # Module implementation of TTS autoregressive decoders. Used for autoregressive TTS model construction.
        /prenet             # Model prenet modules in front of encoders and decoders
            /conv1d.py          # Module implementation of 1D Convolutional prenet.
            /conv2d.py          # Module implementation of 2D Convolutional prenet.
            /embed.py           # Module implementation of token embedding prenet.
            /linear.py          # Module implementation of stacked linear prenet.
            /spk_embed.py       # Module implementation of speaker embedding prenet.
        /postnet            # Model postnet modules behind encoders and decoders
            /conv1d.py          # Module implementation of 1D Convolutional postnet.
            /token.py           # Module implementation of token prediction postnet.
        /transformer        # Transformer-related modules
            /encoder.py         # Module implementation of Transformer encoder layers. Used for decoder construction of ASR and TTS models.
            /decoder.py         # Module implementation of Transformer autoregressive decoder layers. Used for decoder construction of autoregressive ASR and TTS models.
            /pos_enc.py         # Module implementation of positional encoding layers.
            /attention.py       # Module implementation of multi-head attention layers.
            /feed_forward.py    # Module implementation of point-wise feed-forward layers.
```

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/speechain/module#table-of-contents)


## API Document
_Non-overridable backbone functions:_
1. [speechain.module.abs.Module.\_\_init__](https://github.com/ahclab/SpeeChain/tree/main/speechain/module#speechainmoduleabsmodule__init__self-input_size-distributed-module_conf)

_Overridable interface functions:_  
1. [speechain.module.abs.Module.module_init](https://github.com/ahclab/SpeeChain/tree/main/speechain/module#speechainmoduleabsmodulemodule_initself-module_conf)
2. [speechain.module.abs.Module.forward](https://github.com/ahclab/SpeeChain/tree/main/speechain/module#speechainmoduleabsmoduleforwardself-kwargs)
3. [speechain.module.abs.Module.recover](https://github.com/ahclab/SpeeChain/tree/main/speechain/module#speechainmoduleabsmodulerecoverself-kwargs)
4. [speechain.module.abs.Module.reset_parameters](https://github.com/ahclab/SpeeChain/tree/main/speechain/module#speechainmoduleabsmodulereset_parametersself)
5. [speechain.module.abs.Module.get_recordable_para](https://github.com/ahclab/SpeeChain/tree/main/speechain/module#speechainmoduleabsmoduleget_recordable_paraself)

### speechain.module.abs.Module.\_\_init__(self, input_size, distributed, **module_conf)
* **Description:**  
  This initialization function is shared by all _Module_ subclasses. 
  There are two built-in variable members: `input_size` and `output_size`. 
  `input_size` is the last dimension of the input tensor while `output_size` is the last dimension of the output tensor.  
  These two member variables serve as the socket and plug that are used to communicate with the front and back _Module_ objects in a _Model_ object.
  You could utilize `self.input_size` in your `module_init()` implement to initialize your module and give the output data dimension to `self.output_size`.  
  **Note:** The usage of these two member variables is not mandatory, but it would be a convenient way for you to initialize your module.
* **Arguments:**
  * _**input_size:**_ int = None  
    The last dimension of the tensor from the front _Module_ object. If not given, this argument would be None.
  * _**distributed:**_ bool = False  
    Whether the _Model_ object this _Module_ object is belong to is distributed to multiple GPUs.
  * _****module_conf:**_  
    The arguments used by `module_init()` for your customized _Module_ initialization.

👆[Back to the API list](https://github.com/ahclab/SpeeChain/tree/main/speechain/module#api-document)

### speechain.module.abs.Module.module_init(self, **module_conf)
* **Description:**  
  Abstract interface function for customized initialization of each _Module_ subclass. 
  This interface function is mandatory to be overridden by your implementation.
* **Arguments:**
  * _****module_conf:**_  
    The arguments used for customized Module initialization.
    For more details, please refer to the docstring of your target Module subclass.

👆[Back to the API list](https://github.com/ahclab/SpeeChain/tree/main/speechain/module#api-document)

### speechain.module.abs.Module.forward(self, **kwargs)
* **Description:**  
    This abstract interface function is the customized implementation of `torch.nn.Module.forward()` used during model forward calculation. 
    This interface function is mandatory to be overridden by your implementation.
* **Arguments:**
  * _****kwargs:**_  
    The input arguments for module forward calculation.  
    For more details, please refer to the docstring of `forward()` of your target _Module_ subclass.
* **Return:**  
  Module forward calculation results.  
  For more details, please refer to the docstring of `forward()` of your target _Module_ subclass.

👆[Back to the API list](https://github.com/ahclab/SpeeChain/tree/main/speechain/module#api-document)

### speechain.module.abs.Module.recover(self, **kwargs)
* **Description:**  
  This abstract interface function is used to recover the module forward calculation results back to the input data. 
  It can be considered as the reverse process of `forward()`.  
  This interface function is not mandatory to be overridden.
* **Arguments:**
  * _****kwargs:**_  
    The input forward calculation results to be recovered. 
    For more details, please refer to the docstring of `recover()` of your target _Module_ subclass.
* **Return:**  
  The recovered data or closely-recovered data (sometimes `forward()` may not be totally recoverable).  
  For more details, please refer to the docstring of `recover()` of your target _Module_ subclass.

👆[Back to the API list](https://github.com/ahclab/SpeeChain/tree/main/speechain/module#api-document)

### speechain.module.abs.Module.reset_parameters(self)
* **Description:**  
  This abstract interface function is used to initialize the customized parameters in the _Module_ subclass if had. 
  Some _Module_ subclasses have their customized parameters with specific initialization functions.  
  If your _Module_ implementation has some customized parameters and you want to initialize them by yourself, 
  please give the initialization logic in this interface function.  
  This interface function is not mandatory to be overridden.  
  **Note:** Don't forget to add `self.default_init_modules.append(YourModule)` in `model_init()` of your _Model_.

👆[Back to the API list](https://github.com/ahclab/SpeeChain/tree/main/speechain/module#api-document)

### speechain.module.abs.Module.get_recordable_para(self)
* **Description:**  
  This function returns the parameters of the module that you want to record as part of step information.  
  If you want to record the value of the customized parameters of your module:
  1. when it is a leaf (no _Module_ members) in the nested _Module_ tree of the model, 
     please override this function and return the parameter values in a _Dict_.  
     For an example, you can refer to [${SPEECHAIN_ROOT}/speechain/module/transformer/pos_enc.py]().
  2. when it is a non-leaf (with _Module_ members) in the nested _Module_ tree of the model, 
     please follow the pseudocode below:  
     ```python
     class YourModule(Module):
        def get_recordable_para(self) -> Dict or None:
           output = dict()
           # add the value of your target parameter into the output as key-value items
           output.update(super(YourModule, self).get_recordable_para())
           return output
     ```
* **Return:** Dict or None  
  For the leaf module, the default implementation returns None;  
  For the non-leaf module, the default implementation returns a Dict containing names and recordable parameters of its member modules.

👆[Back to the API list](https://github.com/ahclab/SpeeChain/tree/main/speechain/module#api-document)

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/speechain/module#table-of-contents)
