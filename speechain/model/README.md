# Model Calculation Part
Model calculation is done by three classes: *Model*, *Module*, and *Criterion*.

[*Model*](https://github.com/ahclab/SpeeChain/blob/main/speechain/model/abs.py) is the hub of this part where different *Module* and *Criterion* can be freely assembled to create a new model.
This framework encapsulates the general model-related services and provides sufficient interfaces. 
By overriding the interfaces, you can easily customize your own implementations to meet your specific research needs. 

[*Module*](https://github.com/ahclab/SpeeChain/blob/main/speechain/module/abs.py) is the unit of the main body of your model. 
It only has the function of forwarding the input data. 
The input batch is processed by all *Module* in a *Model* sequentially to become the model output.

[*Criterion*](https://github.com/ahclab/SpeeChain/blob/main/speechain/criterion/abs.py) serves the role of evaluating the model predictions. Its output can be either a training loss or a validation metric.

👆[Back to the home page](https://github.com/ahclab/SpeeChain#the-speechain-toolkit)

## Table of Contents
1. [**Configuration File Format**](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#configuration-file-format)
2. [**Abstract Interfaces Description**](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#abstract-interfaces-description)
    1. [Module](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#module)
    2. [Criterion](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#criterion)
    3. [Model](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#model)
3. [**How to Construct a Model**](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#how-to-construct-a-model)
    1. [ASR]()
    2. [TTS]()
4. [**How to Freeze a Specific Part of your Model**](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#how-to-freeze-a-specific-part-of-your-model)
5. [**How to Initialize your Model by the Pretrained Model**](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#how-to-initialize-your-model-by-the-pretrained-model)

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
            ...
        criterion2:
            ...
        ...    
```
1. **model_type** is the query used to pick up your target *Model* class for automatic model initialization.
Your given query should be in the form of `file_name.class_name` to indicate the place of your target class in `./speechain/model/`. 
For example, `asr.ASR` means the class `ASR` in `./speechain/model/asr.py`.

2. **model_conf** contains the general configuration of your model. It is made up of the following 4 parts:
    1. **init** indicates the method used to initialize the parameters of your model before training.
    2. **frozen_modules** contains the names of the modules that don't need to be updated during training. 
    If a list of module names is given, all those modules will be frozen.
    3. **pretrained_model** contains pretrained models you would like to load into your model as the initial parameters. 
    If a list of pretrained models is given, all those pretrained models can be used to initialize your model. 
    4. **customize_conf** contains the configuration used to initialize the customized part of your model. 
    These configurations will be used in the *model_construction()* interface.
    
3. **module_conf** contains all the information about the module initialization. 
Each module corresponds to a key-value pair here and the value of each pair has the same structure as below:
    1. **type** is the query used to pick up the target *Module* class.
    The way how to give the '_type_' arguments is defined in the *model_construction()* of each model.
    2. **conf** contains all the configurations used to initialize this module.

4. **criterion_conf** contains all the information about the criterion initialization. 
You don't need to separate 'type' and 'conf' here but directly give the configuration under the criterion name. 
All the criteria will be initialized explicitly in the *model_construction()* of each model.


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
1. **model_construction()**:
The function where the specific model is constructed. 
This function receives _customize_conf_, _module_conf_, and _criterion_conf_ as the input. 
The construction is at least two parts: built-in module initialization and built-in criterion initialization. 
Some models have their customized part that needs to be initialization in this function, e.g. the tokenizer of ASR and TTS models.

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

    **Note**: *aver_metrics_across_procs()* doesn't need to be overridden if you are doing single-dataloader supervised training.

7. **inference()**:
The function where you decide how your model outputs the inference results based on the input test data. 
The results should be returned in the form of *Dict[str, Dict]* and everything in this *Dict* will be saved to the disk. 
Each key-value pair in the _Dict_ corresponds to a file where the key is the file name and the value is the file content.

For more details, please refer to [*Model*](https://github.com/ahclab/SpeeChain/blob/main/speechain/model/abs.py).

## How to Construct a Model
### ASR
**model_type:**
   1. [asr.ASR]()  
      * Encoder-Decoder ASR model.  
      * Receive one set of speech-text paired data (_feat_, _feat_len_, _text_, _text_len_) in _model_forward()_.  
      * Return a single cross-entropy loss calculated on the supervised data in _loss_calculation()_.
   2. [asr.SemiASR]() (_under development_)   
      * Semi-supervised Encoder-Decoder ASR model.  
      * Receive multiple sets of speech-text paired data  (_feat_, _feat_len_, _text_, _text_len_) in _model_forward()_. 
      One of them is the real speech-text paired data and the others are pseudo speech-text paired data.
      * Return multiple cross-entropy losses calculated on all the paired data sets in _loss_calculation()_. 
      The loss named _loss_ is the trainable overall loss that is the linear combination of all losses. 
      The component loss calcualated on each set of paired set is converted into a non-trainable metric value for recording. 

### TTS
Coming soon~~~

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
mdl_root: ./recipe/speech/librispeech/train_clean_100/asr/transformer/exp/sup_warmup4k
model:
    model_type: asr.ASR
    model_conf:
        pretrained_model:
            path: !ref <mdl_root>/accuracy_best.mdl
```
If you only want to initialize a part of your model, you can use the _mapping_ argument in _pretrained_model_. 
The parameter name mismatch can also be solved by the _mapping_ argument. 
In the example below, only the encoder of the ASR model will be initialized by the given pretrained model. 
Even though the pretrained model is constructed by unit modules, it can still be loaded into the ASR model constructed by template modules by aligning their module names.
```
mdl_root: ./recipe/speech/librispeech/train_clean_100/asr/transformer/exp/sup_warmup4k
model:
    model_type: asr.ASR
    model_conf:
        pretrained_model:
            path: !ref <mdl_root>/accuracy_best.mdl
            mapping: 
              encoder_prenet: encoder.prenet
              encoder: encoder.encoder
```
There could be multiple pretrained models in _pretrained_model_ that are used to initialize your model. 
In the example below, the encoder and decoder of the ASR model are initialized by different pretrained models.

Note that if there are overlapping modules between the _mapping_ arguments of different pretrained models, 
the module will be initialized by the pretrained models at the back of the list.
```
mdl_root: ./recipe/speech/librispeech/train_clean_100/asr/transformer/exp/sup_warmup4k
model:
    model_type: asr.ASR
    model_conf:
        pretrained_model:
            - path: !ref <mdl_root>/accuracy_best.mdl
              mapping:
                encoder_prenet: encoder.prenet
                encoder: encoder.encoder
            - path: !ref <mdl_root>/5_accuracy_average.mdl
              mapping:
                decoder_prenet: decoder.prenet
                decoder: decoder.decoder
                decoder_postnet: decoder.postnet
```

