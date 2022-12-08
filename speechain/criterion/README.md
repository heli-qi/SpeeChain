# Criterion

[*Criterion*](https://github.com/ahclab/SpeeChain/blob/main/speechain/criterion/abs.py)  is a Callable object which is the base class for all criterion objects in this toolkit. 
It serves the role of evaluating the model forward calculation results. 
Its output can be either a loss function used for training or an evaluation metric used for validation.

This base class has two abstract interface functions: `criterion_init()` for criterion initialization and `__call__()` for criterion forward calculation.
1. `__call__()` must be overridden if you want to make your own Criterion implementation.
2. `criterion_init()` is not mandatory to be overridden because some criteria can directly be applied to the input data without any initialization such as [speechain.criterion.accuracy.Accuracy]().

👆[Back to the handbook page](https://github.com/ahclab/SpeeChain#the-speechain-toolkit)

## Table of Contents
1. [**Criterion Library**]()
2. [**API Document**]()

## Criterion Library
```
/speechain
    /criterion          
        /abs.py             # Abstract class of Criterion. Base of all Criterion implementations.
        /accuracy.py        # Criterion implementation of classification accuracy. Mainly used for ASR teacher-forcing accuracy.
        /bce_logits.py      # Criterion implementation of binary cross entropy. Mainly used for TTS stop flag.
        /cross_entropy.py   # Criterion implementation of cross entropy. Mainly used for ASR seq2seq loss function.
        /error_rate.py      # Criterion implementation of word and char error rate. Mainly used for ASR evaluation.
        /fbeta_score.py     # Criterion implementation of F_beta score. Mainly used for evaluation of TTS stop flag.
        /least_error.py     # Criterion implementation of mean square error and mean absolute error. Mainly used for TTS seq2seq loss function.
```

## API Document
1. [speechain.criterion.abs.Criterion.\_\_init__]()
2. [speechain.criterion.abs.Criterion.criterion_init]()
3. [speechain.criterion.abs.Criterion.\_\_call__]()

### speechain.criterion.abs.Criterion.\_\_init__(self, **criterion_conf)
* **Description:**  
    This initialization function is shared by all _Criterion_ subclasses.
    Currently, the shared logic only contains calling the initialization function of the parent class.
* **Arguments:**
  * _****criterion_conf:**_  
    The arguments used by `criterion_init()` for your customized Criterion initialization.

### speechain.criterion.abs.Criterion.criterion_init(self, **criterion_conf)
* **Description:**  
    Abstract interface function for customized initialization of each _Criterion_ subclass.
    This interface function is not mandatory to be overridden by your implementation.
* **Arguments:**
  * _****criterion_conf:**_  
    The arguments used for customized _Criterion_ initialization.
    For more details, please refer to the docstring of your target _Criterion_ subclass.

### speechain.criterion.abs.Criterion.\_\_call__(self, **kwargs)
* **Description:**  
    This abstract interface function receives the model forward calculation results and ground-truth labels.
    The output is a scalar which could be either trainable for parameter optimization or non-trainable for information recording.  
    This interface function is mandatory to be overridden by your implementation.
* **Arguments:**
  * _****kwargs:**_  
    model forward calculation results and ground-truth labels.  
    For more details, please refer to the docstring of `__call__()` of your target _Criterion_ subclass.
* **Return:**
  A trainable or non-trainable scalar.  
  For more details, please refer to the docstring of `__call__()` of your target _Criterion_ subclass.