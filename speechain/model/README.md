# Model Calculation Part
Model calculation is done by three classes: *Model*, *Module*, and *Criterion*.

[*Model*](https://github.com/ahclab/SpeeChain/blob/main/speechain/model/abs.py) is the framework of this part where different *Module* and *Criterion* can be freely assembled to create a new model.
This framework encapsulates the general model-related services and provides sufficient interfaces. 
By overriding the interfaces, you can easily customize your own implementations to meet your specific research needs. 

[*Module*](https://github.com/ahclab/SpeeChain/blob/main/speechain/module/abs.py) is the unit of the main body of your model. It only has the function of forwarding the input data. 
The input batch is processed by all *Module* in a *Model* sequentially to become the model prediction.

[*Criterion*](https://github.com/ahclab/SpeeChain/blob/main/speechain/criterion/abs.py) serves the role of evaluating the model predictions. Its output can be either a training loss or a validation metric.

👆[Back to the home page](https://github.com/ahclab/SpeeChain#the-speechain-toolkit)

## Table of Contents
1. [**Configuration File Format**](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#configuration-file-format)
2. [**Abstract Interfaces Description**](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#abstract-interfaces-description)
    1. [Module](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#module)
    2. [Criterion](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#criterion)
    3. [Model](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#model)
3. [**How to Construct a Model by Available Modules**](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#how-to-construct-a-model-by-available-modules)
4. [**How to Freeze a Specific Part of your Model**](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#how-to-freeze-a-specific-part-of-your-model)
5. [**How to Initialize your Model by the Pretrained Model**](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#how-to-initialize-your-model-by-the-pretrained-model)
6. [**How to Perform Multi-Task Training**](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#how-to-perform-multi-task-training)

## Configuration File Format
The configuration of your model is given in the *model* tag of *train_cfg*. 
The configuration format is shown below. 
If you would like to see some examples, please go to the following sections.
```
model:
    model_type: file_name.class_name
    model_conf:
        init: xxx
        frozen_modules:
          - xxx
          - xxx
          ...
        pretrained_model:
            - path: xxx
              mapping: 
                src_name: tgt_name
                ...
            - path: xxx
              mapping: 
                src_name: tgt_name
                ...
            ...
        customize_conf: 
            xxx: xxx
            ...
    module_conf:
        module1:
            type: module_name.file_name.class_name
            conf:
                ...
        module2:
            type: module_name.file_name.class_name
            conf:
                ...
        ...
    criterion_conf:
        criterion1:
            type: file_name.class_name
            conf:
                ...
        criterion2:
            type: file_name.class_name
            conf:
                ...
        ...    
```
1. **model_type** is the query used to pick up your target *Model* class for automatic model initialization.
Your given query should be in the form of `file_name.class_name` to indicate the place of your target class. 
For example, `asr.ASR` means the class `ASR` in `./speechain/model/asr.py`.

2. **model_conf** contains the general configuration of your model. It is made up of the following 4 parts:
    1. **init** indicates the method used to initialize the parameters of your model before training.
    2. **frozen_modules** contains the names of the modules that don't need to be updated during training. 
    If a list of module names is given, all those modules will be frozen.
    3. **pretrained_model** contains pretrained models you would like to load into your model as the initial parameters. 
    If a list of pretrained models is given, all those pretrained models can be used to initialize your model. 
    4. **customize_conf** contains the configuration used to initialize the customized part of your model. 
    These configurations are used in the *model_customize()* interface.
    
3. **module_conf** contains all the information about the module initialization. 
Each module corresponds to a key-value pair here and the value of each pair has the same structure as below:
    1. **type** is the query used to pick up the target *Module* class.
    Your given query should be in the form of `subfolder_name.file_name.class_name` to indicate the place of your target class. 
    For example, `transformer.encoder.TransformerEncoder` means the class `TransformerEncoder` in `./speechain/module/transformer/encoder.py`.
    2. **conf** contains all the configurations used to initialize this module.

4. **criterion_conf** contains all the information about the criterion initialization. 
Each criterion corresponds to a key-value pair here and the value of each pair has the same structure as below:
    1. **type** is the query used to pick up the target *Criterion* class.
    Your given query should be in the form of `file_name.class_name` to indicate the place of your target class. 
    For example, `cross_entropy.CrossEntropy` means the class `CrossEntropy` in `speechain/criterion/cross_entropy.py`.
    2. **conf** contains all the configurations used to initialize this criterion.


## Abstract Interfaces Description
### Module
1. **module_init()**:
The initialization function of your module implementation.
2. **forward()**:
This function decides how the input data is processed in the module.

For more details, please refer to [*Module*](https://github.com/ahclab/SpeeChain/blob/main/speechain/module/abs.py).

### Criterion
1. **criterion_init()**:
The initialization function of your criterion implementation. 
For a criterion, *criterion_init()* is not mandatory to be overridden.
2. **forward()**:
This function decides how the input model prediction and the target label are used to calculate the evaluation metric.

For more details, please refer to [*Criterion*](https://github.com/ahclab/SpeeChain/blob/main/speechain/criterion/abs.py).

### Model
1. **model_customize()**:
The customized part of your model is initialized in this function.  
Note: *model_customize()* is not mandatory to be overridden.
2. **batch_preprocess()**: 
This function preprocesses the input batch before feeding it to the built-in modules. 
*batch_preprocess()* is not mandatory to be overridden.
3. **model_forward()**: 
The function where you decide how your model outputs the prediction. 
The prediction is returned in the form of *Dict*, so multiple predictions can be included and each one corresponds to a key-value pair.
4. **loss_calculation()**: 
The function where you decide how the training losses are calculated based on the target labels and the model predictions. 
The losses are returned in the form of a *Dict*, so multiple losses can be included and each one corresponds to a key-value pair.
5. **metrics_calculation()**: 
The function where you decide how the evaluation metrics are calculated based on the target labels and the model predictions. 
The metrics are returned in the form of a *Dict*, so multiple metrics can be included and each one corresponds to a key-value pair.
6. **aver_metrics_across_procs()**:
This function averages the input metrics across all processes in the multi-GPU distributed training setting.  
Note: *aver_metrics_across_procs()* doesn't need to be overridden if you are doing single-dataloader supervised training.
7. **inference()**:
The function where you decide the way how your model outputs the inference results based on the input test data. 
The results are returned in the form of a *Dict* and everything in this *Dict* will be saved to the disk. 
So, you can return any results you would like to see about the performance of your model.

For more details, please refer to [*Model*](https://github.com/ahclab/SpeeChain/blob/main/speechain/model/abs.py).

## How to Construct a Model by Available Modules
We provide two granularity of modules for you to construct your model. 
The module is either a *fine-grained unit module* or a *coarse-grained template module*.

### Fine-Grained Unit Module
_Fine-grained unit module_ is the smallest unit to build a model. Each unit module has only one specific purpose. 
An example of the module configuration of a Transformer-based ASR model is shown below. 
In this example, the ASR model is made up of 5 modules and each module corresponds to a key-value pair in _module_conf_. 
```
module_conf:
    frontend:
      type: frontend.speech2mel.Speech2MelSpec
      conf:
        ...

    encoder_prenet:
      type: prenet.conv2d.Conv2dPrenet
      conf:
        ...

    encoder:
      type: transformer.encoder.TransformerEncoder
      conf:
        ...

    decoder_prenet:
      type: prenet.embed.EmbedPrenet
      conf:
        vocab_size: 31
        ...

    decoder:
      type: transformer.decoder.TransformerDecoder
      conf:
        ...

    decoder_postnet:
      type: postnet.token.TokenPostnet
      conf:
        vocab_size: 31
        ...
```
The advantage is that there is a lot of freedom to construct your model and design a complicated model forward function. 
But the disadvantage is that both the configuration and codes will become a little untidy and you may need to give some redundant arguments in your configuration 
(e.g. _vocab_size_ needs to be given _decoder_prenet_ and _decoder_postnet_ twice.).

### Coarse-Grained Template Module
_Coarse-grained template module_ is the module that holds several built-in unit modules. 
With template modules, the model construction becomes easier and tidier. 
The example of constructing the same Transformer-based ASR model above by template modules is shown below.
```
module_conf:
    encoder:
      type: encoder.asr.ASREncoder
      conf:
        frontend:
          type: feat_frontend.speech2mel.Speech2MelSpec
          conf:
            ...

        prenet:
          type: prenet.conv2d.Conv2dPrenet
          conf:
            ...

        encoder:
          type: transformer.encoder.TransformerEncoder
          conf:
            ...

    decoder:
      type: decoder.asr.ASRDecoder
      conf:
        vocab_size: 31

        prenet:
          type: prenet.embed.EmbedPrenet
          conf:
            ...

        decoder:
          type: transformer.decoder.TransformerDecoder
          conf:
            ...

        postnet:
          type: postnet.token.TokenPostnet
          conf:
            ...
```
In this example, the ASR model is made up of two template modules: _encoder_ and _decoder_. 
In each template module, several unit modules are chosen to initialize its built-in unit models. 
In this way, the configuration becomes neater and tidier (we only need to give _vocal_size_ to _decoder_ once). 

In the contrast, the disadvantage is that the built-in unit modules are fixed for each template module. 
It may not be able to meet some specific research needs.

## How to Freeze a Specific Part of your Model
Parameter freezing can be done simply by giving the name of the module you want to freeze in _frozen_modules_. 
In the example below, the encoder of the ASR model will be frozen while other modules are still trainable.
```
model:
    model_type: asr.ASR
    model_conf:
        frozen_modules: encoder
```
If you want to freeze multiple modules, you can give their names as a list in _frozen_modules_. 
In the example below, the prenets of both the encoder and decoder will be frozen.
```
model:
    model_type: asr.ASR
    model_conf:
        frozen_modules:
          - encoder.prenet
          - decoder.prenet
```
The parameter freezing granularity can be very fine if you specify the module name by a series of dots. 
In the example below, the convolution layers of the prenet of the encoder will be frozen.
```
model:
    model_type: asr.ASR
    model_conf:
        frozen_modules: 
            - encoder.prenet.conv
```

## How to Initialize your Model by the Pretrained Model
Pretrained model loading can be easily done by giving the model path in _pretrained_model_. 
In the example below, the entire ASR model will be initialized by the given _best_accuracy.mdl_ model.
```
model:
    model_type: asr.ASR
    model_conf:
        pretrained_model:
            path: ./recipe/speech/librispeech/train_clean_100/asr/transformer/exp/sup_warmup4k/accuracy_best.mdl
```
If you only want to initialize a part of your model, you can use the _mapping_ argument in _pretrained_model_. 
The parameter name mismatch can also be solved by the _mapping_ argument. 
In the example below, only the encoder of the ASR model will be initialized by the given pretrained model. 
Even though the pretrained model is constructed by unit modules, it can still be loaded into the ASR model constructed by template modules by aligning their module names.
```
model:
    model_type: asr.ASR
    model_conf:
        pretrained_model:
            path: ./recipe/speech/librispeech/train_clean_100/asr/transformer/exp/sup_warmup4k/accuracy_best.mdl
            mapping: 
              encoder_prenet: encoder.prenet
              encoder: encoder.encoder
```
There could be multiple pretrained models in _pretrained_model_ that are used to initialize your model. 
In the example below, the encoder and decoder of the ASR model are initialized by different pretrained models.

Note that if there are overlapping modules between the _mapping_ arguments of different pretrained models, 
the module will be initialized by the pretrained models at the back of the list.
```
model:
    model_type: asr.ASR
    model_conf:
        pretrained_model:
            - path: ./recipe/speech/librispeech/train_clean_100/asr/transformer/exp/sup_warmup4k/accuracy_best.mdl
              mapping:
                encoder_prenet: encoder.prenet
                encoder: encoder.encoder
            - path: ./recipe/speech/librispeech/train_clean_100/asr/transformer/exp/sup_warmup4k/5_accuracy_average.mdl
              mapping:
                decoder_prenet: decoder.prenet
                decoder: decoder.decoder
                decoder_postnet: decoder.postnet
```

## How to Perform Multi-Task Training
In the normal setting, both the training loss, validation metric, and testing metric are given by the _type&conf_ configuration. 
The example below is the criterion configuration for an ASR model. 
_cross_entropy_ is the training loss, _accuracy_ is the validation metric, and _error_rate_ is the testing metric.
```
criterion_conf:
    cross_entropy:
      type: cross_entropy.CrossEntropy
      conf:
        ...

    accuracy:
      type: accuracy.Accuracy
      conf:
        ...

    error_rate:
      type: error_rate.ErrorRate
      conf:
        ...
```
Sometimes, you may need to combine multiple losses to a weighted sum in the multi-task setting. 
In this case, you can give the configuration of the combined loss like the example below. 
This example is the criterion configuration of a semi-supervised ASR model. 
The training loss _cross_entropy_ is calculated by two losses: _sup_ loss and _unsup_ loss. 
The weights of these two losses are both 0.5.

This one is just an example. The number of component losses can be more than 2. 
The names of the component losses (_sup_ and _unsup_ in this case) are used in _loss_calculation()_ to calculate the combined loss.
```
criterion_conf:
    cross_entropy:
      sup:
        weight: 0.5
        type: cross_entropy.CrossEntropy
        conf:
          ...
      unsup:
        weight: 0.5
        type: cross_entropy.CrossEntropy
        conf:
          ...
    
    accuracy:
      type: accuracy.Accuracy
      conf:
        ...

    error_rate:
      type: error_rate.ErrorRate
      conf:
        ...
```
