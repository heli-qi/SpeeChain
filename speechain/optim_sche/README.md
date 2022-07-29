# Parameter Optimization Part
Parameter Optimization is done by only one class: *OptimScheduler*.

[*OptimScheduler*]() combines the roles of traditional optimizers and schedulers together. 
The job of updating the model parameters is done by the built-in `torch.optim.Optimizer` of this class. 
*OptimScheduler* also defines the strategy of adjusting the learning rates of the built-in `torch.optim.Optimizer` during training. 

👆[Back to the home page]()

## Table of Contents
1. [**Configuration File Format**]()
2. [**Abstract Interfaces Description**]()
    1. [OptimScheduler]()
4. [**How to Construct Multiple Optimizers on Multiple Losses**]()
5. [**How to Simulate Large Batch Training with Limited GPUs**]()
6. [**How to Perform Fine-tuning**]()

## Configuration File Format
The configuration of *OptimScheduler* is given in the *optim_sches* tag of *train_cfg*. 
The configuration format is shown below. If you would like to see some examples, please go to the following sections.
```
optim_sches:
    optim_sche_name1:
        type: file_name.class_name
        conf:
            optim_type: class_name
            optim_conf:
                ...
            ...
    optim_sche_name2:
        type: file_name.class_name
        conf:
            optim_type: class_name
            optim_conf:
                ...
            ...
    ...
```
1. **optim_sche_name** is the first-level key that indicates the name of an optimscheduler. 
Currently, these names are not used in codes but only for logging information during training. 
You can give whatever you want as your optimscheduler names. 
2. **type** is the second-level key that indicates your optimscheduler type. 
The value of this key is used as the query to pick up the target *OptimScheduler* class. 
Your given query should be in the form of `file_name.class_name` to indicate the place of your target class. 
For example, `noam.NoamLr` means the class `NoamLr` in `./speechain/optim_sche/noam.py`.
4. **conf** is the second-level key that indicates your optimscheduler configuration. 
The values of this key are as following:
    1. **optim_type** indicates the type of the built-in optimizer in this optimscheduler. 
    The value will be used as the query to pick up the target *torch.optim.Optimizer* class. 
    Your given query should be in the form of `class_name` to indicate your target class in `torch.optim`. 
    For example, `Adam` means the class `torch.optim.Adam`.
    2. **optim_conf** contains all the configuration used to initialize the built-in optimizer.
    3. **optimscheduler general configuration**. 
    These configurations are used to initialize the general part shared by all optimschedulers. 
    Please refer to the docstrings of [*OptimScheduler*]() for more details.
    4. **optimscheduler customized configuration**. 
    These configurations are used to initialize the customized part of a specific optimscheduler. 
    This part defines the scheduling strategy to adjust the learning rates during training.
    Please refer to the docstrings of your target *OptimScheduler* class for more details.


## Abstract Interfaces Description
### OptimScheduler
1. **sche_init()**: 
The function where the scheduler part of the optimscheduler is initialized. 
Mainly decide how the learning rate adjustment strategy works as the training goes. 
2. **update_lr()**: 
The function where the learning rate is adjusted by the strategy in the optimscheduler.
3. **\__repr\__()**: 
This function returns the string that describes the information of your optimscheduler. 
Mainly used when logging the experiment information into the log file.

For more details, please refer to [*OptimScheduler*]().

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
Since small batches are used to calculate the gradients of each step, some calculations of large batch training cannot be simulated (e.g. _BatchNorm_ and _LayerNorm_). 
Therefore, the performance of the model trained by gradient accumulation may be slightly worse than the one trained by the actual large batches. 

## How to Perform Fine-tuning
In the normal setting, we need to scale down the learning rates by a factor of 10 to 100 for fine-tuning a pretrained model. 
In this toolkit, the learning rates can be easily scaled down by the input argument `ft_factor` in _exp_cfg_ without changing the scheduling configuration of your optimscheduler. 
It's no longer necessary for you to redesign the scheduler configuration for fine-tuning!
