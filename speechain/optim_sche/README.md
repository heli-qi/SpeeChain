# OptimScheduler
[*OptimScheduler*](https://github.com/ahclab/SpeeChain/blob/main/speechain/optim_sche/abs.py) is the base class of all _OptimScheduler_ objects that combine the roles of traditional optimizers and schedulers together. 
Its main job is optimizing the target model parameters and scheduling the learning rate during training.  
In this toolkit, we combine traditional optimizers and schedulers into a single class: OptimScheduler. 
Each _OptimScheduler_ object has one built-in member optimizer (`torch.optim.Optimizer`) which is initialized automatically by the `optim_type` and `optim_conf` given in your configuration.

👆[Back to the home page](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#speechain-handbook)

## Table of Contents
1. [**Configuration File Format**](https://github.com/ahclab/SpeeChain/tree/main/speechain/optim_sche#configuration-file-format)
2. [**OptimScheduler Library**](https://github.com/ahclab/SpeeChain/tree/main/speechain/optim_sche#optimscheduler-library)
3. [**API Document**](https://github.com/ahclab/SpeeChain/tree/main/speechain/optim_sche#api-document)
4. [**How to Construct Multiple Optimizers on Multiple Losses**](https://github.com/ahclab/SpeeChain/tree/main/speechain/optim_sche#how-to-construct-multiple-optimizers-on-multiple-losses)
5. [**How to Simulate Large Batch Training with Limited GPUs**](https://github.com/ahclab/SpeeChain/tree/main/speechain/optim_sche#how-to-simulate-large-batch-training-with-limited-gpus)
6. [**How to Perform Fine-tuning**](https://github.com/ahclab/SpeeChain/tree/main/speechain/optim_sche#how-to-perform-fine-tuning)

## Configuration File Format
The configuration of *OptimScheduler* is given in the `optim_sches` tag of *train_cfg*. 
The configuration format is shown below.
```
optim_sches:
    type: {file_name}.{class_name}
    conf:
        optim_type: {class_name}
        optim_conf:
            ...
        # general optimscheduler configuration
        optim_loss:
        updated_modules:
        step_per_update:
        # customized optimscheduler configuration
        ...
```
* The first-level key must be **optim_sches** to notify the framework of the optimscheduler configuration.
  1. **type** is a second-level key that indicates your optimscheduler type. 
  The value of this key is used as the query to pick up your target *OptimScheduler* subclass for initialization. 
  Your given query should be in the form of `file_name.class_name` to indicate the place of your target subclass.  
  For example, `noam.NoamLr` means the class `NoamLr` in `./speechain/optim_sche/noam.py`.
  3. **conf** is a second-level key that indicates your optimscheduler configuration. 
  The value of this key is a _Dict_ whose configuration is as following:
      1. **optim_type** is a query that indicates the type of the built-in `torch.optim.Optimizer` in this optimscheduler. 
      Your given query should be in the form of `class_name` to indicate your target subclass in `torch.optim`.  
      For example, `Adam` means the class `torch.optim.Adam`.
      2. **optim_conf** contains all the configuration used to initialize the built-in optimizer.  
        For more details, please refer to the _PyTorch_ document of your target `torch.optim.Optimizer` subclass.
      3. **optimscheduler general configuration** is shared by all _OptimScheduler_ subclasses. 
         1. optim_loss
         2. updated_modules
         3. step_per_update
      4. **optimscheduler customized configuration** is used to initialize the customized part of each optimscheduler subclass. 
      This part defines the scheduling strategy to adjust the learning rates during training.
      Please refer to the docstrings of your target *OptimScheduler* subclass for more details.
      
👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/speechain/optim_sche#table-of-contents)

## OptimScheduler Library
```
/speechain
    /optim_sche
        /abs.py     # Abstract class of OptimScheduler. Base of all OptimScheduler implementations.
        /noam.py    # OptimScheduler implementation of the Noam scheduler. Mainly used for Transformer training.
```

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/speechain/optim_sche#table-of-contents)

## API Document
_Non-overridable backbone functions:_
1. [speechain_optim_sche.abs.OptimScheduler.\_\_init__](https://github.com/ahclab/SpeeChain/tree/main/speechain/optim_sche#speechain_optim_scheabsoptimscheduler__init__self-optim_type-optim_conf-model-distributed-optim_loss-updated_modules-step_per_update-use_amp-accum_grad-ft_factor-grad_clip-grad_norm_type-sche_conf)
2. [speechain.optim_sche.abs.OptimScheduler.step](https://github.com/ahclab/SpeeChain/tree/main/speechain/optim_sche#speechain_optim_scheabsoptimschedulersteplosses-time_func-optim_name-step_num)
3. [speechain.optim_sche.abs.OptimScheduler.get_lr](https://github.com/ahclab/SpeeChain/tree/main/speechain/optim_sche#speechain_optim_scheabsoptimschedulerget_lrself)
4. [speechain.optim_sche.abs.OptimScheduler.state_dict](https://github.com/ahclab/SpeeChain/tree/main/speechain/optim_sche#speechain_optim_scheabsoptimschedulerstate_dictself)
5. [speechain.optim_sche.abs.OptimScheduler.load_state_dict](https://github.com/ahclab/SpeeChain/tree/main/speechain/optim_sche#speechain_optim_scheabsoptimschedulerload_state_dictself-state_dict)
6. [speechain.optim_sche.abs.OptimScheduler.\_\_repr__](https://github.com/ahclab/SpeeChain/tree/main/speechain/optim_sche#speechain_optim_scheabsoptimscheduler__repr__self)

_Overridable interface functions:_  
7. [speechain.optim_sche.abs.OptimScheduler.sche_init](https://github.com/ahclab/SpeeChain/tree/main/speechain/optim_sche#speechain_optim_scheabsoptimschedulersche_initsche_conf)
8. [speechain.optim_sche.abs.OptimScheduler.update_lr](https://github.com/ahclab/SpeeChain/tree/main/speechain/optim_sche#speechain_optim_scheabsoptimschedulerupdate_lrself-real_step)
9. [speechain.optim_sche.abs.OptimScheduler.extra_repr_fn](https://github.com/ahclab/SpeeChain/tree/main/speechain/optim_sche#speechain_optim_scheabsoptimschedulerextra_repr_fnself)

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/speechain/optim_sche#table-of-contents)

### speechain_optim_sche.abs.OptimScheduler.\_\_init__(self, optim_type, optim_conf, model, distributed, optim_loss, updated_modules, step_per_update, use_amp, accum_grad, ft_factor, grad_clip, grad_norm_type, **sche_conf)
* **Description:**  
    This initialization function initializes the general part shared by all _OptimScheduler_ subclasses.
    At the end of this function, an interface function `sche_init()` is called to initialize the customized part of each _OptimScheduler_ subclass.
* **Arguments:**  

    _Arguments received from `exp_cfg`_:
  * _**model:**_ speechain.model.abs.Model  
    The pointer to the model whose parameters will be optimized by the built-in `torch.optim.Optimizer`.
  * _**distributed:**_ bool = False  
    Whether the model to be optimized is distributed to multiple GPUs.   
    If True, gradient accumulation will be done asynchronously in the DDP mode to speed up training.
  * _**use_amp:**_ bool = True  
    Whether the Automatic Mixed Precision (AMP) technique is used during back-propagation.  
    If True, a built-in `torch.cuda.amp.GradScaler` will be initialized to calculate the gradients and optimize the parameters.
  * _**accum_grad:**_ int = 1  
    The number of steps to accumulate gradients before optimization. 
    The larger this argument is, the larger your virtual batches will be.
  * _**ft_factor:**_ float = 1.0  
    The finetuning factor used to scale down the learning rates during training.
  
  _Arguments received from `train_cfg`_:
  * _**optim_type:**_ str  
    The optimizer query used to pick up the target _Optimizer_ subclass from `torch.optim`.
  * _**optim_conf:**_ Dict  
    The configuration used to initialize the built-in `torch.optim.Optimizer`.
  * _**optim_loss:**_ str = None  
    The name of the target loss used in this _OptimScheduler_ object to calculate the gradients. 
    If not given, the loss named `loss` will be used for optimization.
  * _**updated_modules:**_ str or List[str]  
    This argument allows you to update only a part of parameters of the built-in model pointer. 
    `updated_modules` indicate the names of your target modules (first-level module in the nested module tree) in the built-in model pointer.  
    Its value can be either a string (only one target module) or a list (multiple target modules).  
    If not given, the entire model will be updated.
  * _**step_per_update:**_ int = 1  
    The optimization interval for the built-in optimizer.
    It means that the parameter optimization will be done once every `step_per_update` steps.
  * _****sche_conf:**_  
    The arguments used to initialize the customized part of this _OptimScheduler_.   
    Mainly used to decide the learning rate scheduling strategy.

### speechain_optim_sche.abs.OptimScheduler.step(losses, time_func, optim_name, step_num)
* **Description:**  
    This function optimizes the target parameters of the built-in model pointer with the input training losses.
* **Arguments:**
  * _**losses:**_ Dict[str, torch.Tensor]  
    The training loss Dict received from the `criterion_forward()` of the bulit-in model pointer.
  * _**time_func:**_  
    The context function used to record the consumed time during gradient back-propagation and parameter optimization.
  * _**optim_name:**_ str  
    The name of the _OptimScheduler_ object. 
    This argument is used to identify the recorded consumed time information.
  * _**step_num:**_ int  
    The number of the current training step. 
    This argument is used to update the learning rate for the current step by `self.update_lr()`.

### speechain_optim_sche.abs.OptimScheduler.get_lr(self)
* **Description:**  
    This function returns the current learning rate of the built-in `torch.optim.Optimizer` member.
* **Return:** float  
  The value of the learning rates obtained from `self.optimizer.param_groups`.

### speechain_optim_sche.abs.OptimScheduler.state_dict(self)
* **Description:**  
    This function returns the current status of the OptimScheduler object for checkpoint storage.
* **Return:** Dict  
  The status Dict containing the current status of the built-in `torch.optim.Optimizer` and the built-in `torch.cuda.amp.GradScaler` (if had).

### speechain_optim_sche.abs.OptimScheduler.load_state_dict(self, state_dict)
* **Description:**  
  This function loads the existing checkpoint information into the _OptimScheduler_ object as the starting status.
* **Arguments:**
  * _**state_dict:**_ Dict  
    The status information loaded from the existing checkpoint.

### speechain_optim_sche.abs.OptimScheduler.\_\_repr__(self)
* **Description:**  
    This function returns the description string of the _OptimScheduler_ object. 
    There is a general description part shared by all the _OptimScheduler_ subclasses.  
    In this function, an interface hook function `extra_repr_fn()` will be called to generate the specific description part of each _OptimScheduler_ subclass.
* **Return:** str  
    The description string for the OptimScheduler object.

### speechain_optim_sche.abs.OptimScheduler.sche_init(**sche_conf)
* **Description:**  
    This abstract interface function is the customized initialization function which decides how the learning rate is scheduled as the training goes.  
    This interface is mandatory to be overridden.
* **Arguments:**
  * _****sche_conf:**_  
    The arguments used to initialize the customized part of this OptimScheduler.
    For more details about the learning rate scheduling strategy, please refer to the docstring of `sche_init()` of your target OptimScheduler subclass.

### speechain_optim_sche.abs.OptimScheduler.update_lr(self, real_step)
* **Description:**  
    This abstract interface function generates the learning rate by the input step number.
* **Arguments:**
  * _**real_step:**_ int  
    The number of the real step for parameter optimization. 
    Due to the existence of `self.accum_grad`, parameter optimization may not be done at each training step. 
    The real step number here means the training steps where parameter optimization is done.
* **Return:** float  
    The learning rate used for parameter optimization in the current training step.

### speechain_optim_sche.abs.OptimScheduler.extra_repr_fn(self)
* **Description:**  
    This interface hook function returns the specific part of the description string of the _OptimScheduler_ object. 
    The original implementation in the base class returns an empty string.  
    In principle, this interface hook function must be overridden by each _OptimScheduler_ subclass. 
    But there won't be any errors if you don't override it in your implementation.
* **Return:** str  
    The specific part of the description string of the _OptimScheduler_ object. 

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/speechain/optim_sche#table-of-contents)

## How to Construct Multiple Optimizers on Multiple Losses
The cooperation of multiple optimizers is handled by 3 arguments: _optim_losses_, _updated_modules_, and _step_per_update_. 
1. _optim_losses_ means the training loss used to calculate the gradients for the optimizer. 
2. _update_modules_ means the target module in your where that you would like the optimizer to update the parameters.
3. _step_per_update_ means the updating frequency of the optimizer (i.e. the parameter optimization can be done once per _step_per_update_ steps).

In the example below, there are two optimschedulers for optimizing the parameters of an Encoder-Decoder model. 
_encoder_optim_ optimizes the encoder part using the training loss called _encoder_loss_ while _decoder_optim_ optimizes the decoder part using the training loss called _decoder_loss_. 
The encoder optimization is done once every 2 steps while the decoder optimization is done once every step.
```
optim_sches:
    encoder_optim:
        type: noam.NoamLr
        conf:
            optim_type: Adam
            optim_conf:
                ...
            optim_losses: encoder_loss
            updated_modules: encoder
            step_per_update: 2

    decoder_optim:
        type: noam.NoamLr
        conf:
            optim_type: Adam
            optim_conf:
                ...
            optim_losses: decoder_loss
            updated_modules: decoder
```

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/speechain/optim_sche#table-of-contents)

## How to Simulate Large Batch Training with Limited GPUs
We provide a method called gradient accumulation (implemented by the argument `accum_grad` in _exp_cfg_) to train your model with large batches that are beyond the memory of your GPUs. 
The basic idea is to accumulate the gradients calculated in several small batches and update the model with the accumulated gradients to mimic a large batch. 
So, the actual batch size becomes `accum_grad * batch_size`.

The pseudo-code of gradient accumulation is like this:
```python
for step in range(max_step):
    loss /= accum_grad
    loss.backward()
    if step % accum_grad == 0:
        # real_step = (step - 1) // accum_grad + 1
        optimizer.step()
        optimizer.zero_grad()
```
Let me show you an intuitive example. 
Suppose we want to calculate the mean value of 1, 2, ..., 9, 10 but we cannot directly divide the sum by 10 because our calculator is not powerful enough. 
Instead, we can calculate the mean value of two sub-groups: 1, 2, .., 5 and 6, 7, ..., 10. 
We get two sub-mean values: 3 and 8. 
The overall mean value can be calculated by taking the mean value of these two sub-mean values: (3 + 8) / 2 = 5.5

Unfortunately, gradient accumulation is not identical to large batch training. 
Since small batches are used to calculate the gradients of each step, some calculations of large batch training cannot be simulated (e.g. _BatchNorm_ and _FeatureNormalization_). 
Therefore, the performance of the model trained by gradient accumulation may be slightly different from the one trained by the actual large batches. 

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/speechain/optim_sche#table-of-contents)

## How to Perform Fine-tuning
In the normal setting, we need to scale down the learning rates by a factor of 10 to 100 for fine-tuning a pretrained model. 
In this toolkit, the learning rates can be easily scaled down by the input argument `ft_factor` in _exp_cfg_ without changing the scheduling configuration of your optimscheduler. 
It's no longer necessary for you to redesign the scheduler configuration for fine-tuning!

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/speechain/optim_sche#table-of-contents)
