# Model
[*Model*](https://github.com/ahclab/SpeeChain/blob/main/speechain/model/abs.py) is the hub of this part where different *Module* and *Criterion* objects can be freely assembled to create a model.
_Model_ encapsulates the general model-related services and provides sufficient interface functions for you to override to customize your own models. 

👆[Back to the handbook page](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#speechain-handbook)

## Table of Contents
1. [**Configuration File Format**](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#configuration-file-format)
2. [**Model Library**](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#model-library)
3. [**API Document**](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#api-document)
4. [**Supported Models**](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#supported-models)
5. [**How to Freeze a Specific Part of your Model**](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#how-to-freeze-a-specific-part-of-your-model)
6. [**How to Initialize your Model by the Pretrained Models**](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#how-to-initialize-your-model-by-the-pretrained-model)

## Configuration File Format
The configuration of your model is given in *train_cfg*. 
The configuration format is shown below.
```
model:
    model_type: {file_name}.{class_name}
    model_conf:
        init: {init_function}
        frozen_modules:
            - {frozen_module1}
            - {frozen_module2}
            - ...
        pretrained_model:
            - path: {model_path1}
              mapping: 
                {src_name1}: {tgt_name1}
                {src_name2}: {tgt_name2}
                ...
            - path: {model_path2}
            - ...
        visual_infer_conf:
            ...
        customize_conf: 
            {customize_arg1}: {arg_value1}
            {customize_arg2}: {arg_value2}
            ...
    module_conf:
        ...
    criterion_conf:
        ...    
```
* The first-level key must be **model** to notify the framework of the model configuration.
  1. **model_type** is used as the query to pick up your target *Model* subclass in `{SPEECHAIN_ROOT}/speechain/model/` for model initialization.
  Your given query should be in the form of `{file_name}.{class_name}`, e.g., `asr.ASR` means the subclass `ASR` in `{SPEECHAIN_ROOT}/speechain/model/asr.py`.
  
  2. **model_conf** contains the general configuration of your model. It is made up of the following 4 parts:
      1. **init** indicates the function used to initialize the parameters of your model before training. 
      The available initialization functions are shown in the keys of the built-in dictionary `init_class_dict`.  
      For more details about the available initialization functions, please refer to the built-in dictionary `init_class_dict`.
      2. **frozen_modules** contains the names of the modules that don't need to be updated during training. 
      If a list of module names is given, all those modules will be frozen.
      3. **pretrained_model** contains the pretrained models you would like to load into your model as the initial parameters. 
      If a list of pretrained models is given, all those pretrained models will be used to initialize your model.  
         1. **path** indicates where the pretrained model file is placed.
         2. **mapping** is a dictionary used to solve the mismatch between the parameter names of the pretrained model and the model you want to train. 
         Each key-value item solves a name mismatch where the key is the name in the pretrained model and the value is the name in the model to be trained.
      4. **visual_infer_conf** contains the inference configuration you want to use for model visualization during training. 
         This argument is default to be an empty dictionary which means the default inference configuration of each model will be used.  
         For more details, please refer to the docstring of `inference()` of each _Model_ subclass.
      5. **customize_conf** will be used to initialize the main body of the model in the interface function *module_init()*.  
         For more details about the argument setting, please refer to the README.md of each _Model_ subclass.
    
  3. **module_conf** contains all the configuration about the module initialization. 
  These configuration arguments will be used to initialize the network structure of the model in the interface function *module_init()*.  
  For more details about the argument setting, please refer to the README.md of each _Model_ subclass.

  4. **criterion_conf** contains all the information about the criterion initialization. 
  These configuration arguments will be used to initialize all the criteria of the model in the interfance function *criterion_init()*.  
  For more details about the argument setting, please refer to the README.md of each _Model_ subclass.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#table-of-contents)

## Model Library
```
/speechain
    /model
        /abs.py     # Abstract Model class. Base of all Model implementations.
        /asr.py     # Initial Model implementation of autogressive ASR. Base of all advanced ASR models.
        /tts.py     # Initial Model implementation of autogressive TTS. Base of all advanced TTS models.
```

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#table-of-contents)


## API Document
1. [speechain.model.abs.Model](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#speechainmodelabsmodel)  
   _Non-overridable backbone functions:_
   1. [\_\_init\_\_](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#__init__self-args-device-model_conf-module_conf-criterion_conf)
   2. [batch_to_cuda](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#batch_to_cudaself-data)
   3. [forward](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#forwardself-batch_data-epoch-kwargs)
   4. [aver_metrics_across_procs](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#aver_metrics_across_procsself-metrics-batch_data)
   5. [evaluate](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#evaluateself-test_batch-infer_conf)
   
   _Overridable interface functions:_  
   1. [bad_cases_selection_init_fn](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#bad_cases_selection_init_fn)
   2. [module_init](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#module_initself-kwargs)
   3. [criterion_init](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#criterion_initself-criterion_conf)
   4. [batch_preprocess_fn](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#batch_preprocess_fnself-batch_data)
   5. [module_forward](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#module_forwardself-batch_data)
   6. [criterion_forward](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#criterion_forwardself-kwargs)
   7. [visualize](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#visualizeself-epoch-sample_index-valid_sample)
   8. [inference](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#inferenceself-infer_conf-kwargs)

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#table-of-contents)

### speechain.model.abs.Model
_speechain.model.abs.Model_ is the base class for all models in this toolkit. 
The main job of a model includes: 
1. (optional) preprocess the input batch data to the trainable format
2. calculate the model prediction results by the _Module_ members
3. evaluate the prediction results by the _Criterion_ members
 
Each model has several built-in _Module_ members that make up the neural network structure of the model. 
These _Module_ members will be initialized by the `module_conf` given in your configuration.

There are a built-in dictionary named `init_class_dict` and a built-in list named `default_init_modules` in the base class.
`init_class_dict` contains all the available initialization functions of the model parameters while `default_init_modules` includes the network layers that have their own initialization functions.

#### \_\_init__(self, args, device, model_conf, module_conf, criterion_conf)
* **Description:**  
    In this initialization function, there are two parts of initialization: model-specific customized initialization and model-independent general initialization.

    1. Model-specific customized initialization is done by two interface functions: `module_init()` and `criterion_init()`. 
    `module_init()` initializes the neural network structure of the model while `criterion_init()` initializes the criteria used to optimize (loss functions) and evaluate (validation metrics) the model.

    2. After the customized initialization, there are 3 steps for general initialization shared by all _Model_ subclasses:
       1. Pretrained parameters will be loaded into your model if the key `pretrained_model` is given. 
       Multiple pretrained models can be specified and each of them can be loaded into different parts of your model. 
       The mismatch between the names of pretrained parameters and the parameters of your model is handled by the key `mapping`. 
       The value of the key `mapping` is a dictionary where each key-value item corresponds to a mapping of parameter names. 
       The key is the parameter name in the pretrained parameters while the value is the parameter name of your model.

       2. If `pretrained_model` is not given, the parameters of your model will be initialized by the function that matches your input query `init`. 
       For more details about the available initialization functions, please refer to the built-in dictionary `init_class_dict`. 
       If `init` is not given, the default initialization function `torch.nn.init.xavier_normal_` will be used to initialize your model.

       3. Finally, the specified parts of your model will be frozen if `frozen_modules` is given. 
       If there is only one frozen module, you can directly give the string of its name to `frozen_modules` like `frozen_modules: {module_name}`. 
       If there are multiple modules you want to freeze, you can give their names in a list as
          ```
          frozen_modules:
           - {module_name1}
           - {module_name2}
           - ...
          ```
       Moreover, the frozen granularity depends on your input `frozen_modules`. For example,
       1. If you give `frozen_modules: encoder_prenet`, all parameters of the prenet of your encoder will be frozen
       2. If you give `frozen_modules: encoder_prenet.conv`, only the convolution layers of the prenet of your encoder will be frozen
       3. If you give `frozen_modules: encoder_prenet.conv.0`, only the first convolution layer of the prenet of your encoder will be frozen
       4. If you give `frozen_modules: encoder_prenet.conv.0.bias`, only the bias vector of the first convolution layer of the prenet of your encoder will be frozen
* **Arguments:**
  * _**args:**_ argparse.Namespace  
    Experiment pipeline arguments received from the `Runner` object in `runner.py`.
  * _**device:**_ torch.device  
    The computational device used for model calculation in the current GPU process.
  * _**model_conf:**_ Dict  
    The model configuration used for general model initialization.
  * _**module_conf:**_ Dict  
    The module configuration used for network structure initialization.
  * _**criterion_conf:**_ Dict = None  
    The criterion configuration used for criterion (loss functions and evaluation metrics) initialization.

👆[Back to the API list](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#api-document)

#### batch_to_cuda(self, data)
* **Description:**  
    The recursive function that transfers the batch data to the specified device in the current process.
* **Arguments:**
  * _**data:**_ Dict or torch.Tensor  
    The input batch data. It should be either a Tensor or a Dict of Tensors. 
    For the Dict input, the function itself will be called once by each Tensor element.
* **Return:** Dict or torch.Tensor  
    If the input is a Dict, the returned output will also be a Dict of Tensors transferred to the target device;  
    If the input is a Tensor, the returned output will be its copy on the target device.

👆[Back to the API list](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#api-document)

#### forward(self, batch_data, epoch, **kwargs)
* **Description:**  
    The general model forward function shared by all the _Model_ subclasses. This forward function has 3 steps:
    1. preprocess and transfer the batch data to GPUs
    2. obtain the model prediction results
    3. calculate the loss function and evaluate the prediction results
    
    For each step above, we provide interface functions for you to override and make your own implementation.
* **Arguments:**
  * _**batch_data:**_ Dict  
    The input batch data received from the `train` or `valid` dataloader object in the experimental pipeline. 
    The batch is in the form of a Dict where the key is the data name and the value is the data content. 
  * _**epoch:**_ int = None  
    The number of the current epoch. Used for real-time model visualization and model prediction.
  * _****kwargs:**_  
    The additional arguments for real-time model visualization. If given, the code will go through the model visualization branch.
* **Return:**  
    In the training branch, the loss functions and evaluation metrics will be returned each of which is in the form of a _Dict_.  
    In the validation branch, only the evaluation metrics will be returned.  
    In the visualization branch, the model snapshots on the given validation instance will be returned.

👆[Back to the API list](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#api-document)

#### aver_metrics_across_procs(self, metrics, batch_data)
* **Description:**  
    This function averages the evaluation metrics across all GPU processes in the DDP mode for model distribution.
* **Arguments:**
  * _**metrics:**_ Dict[str, torch.Tensor]  
    The evaluation metrics to be averaged across all GPU processes. 
  * _**batch_data:**_ Dict  
    The input batch data used to calculate the batch size for averaging evaluation metrics.
* **Return:** Dict[str, torch.Tensor]  
    The evaluation metrics _Dict_ after averaging. The key names remain the same.

👆[Back to the API list](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#api-document)

#### evaluate(self, test_batch, infer_conf)
* **Description:**  
    The shared evaluation function by all _Model_ subclasses. This evaluation function has 2 steps:
    1. preprocess and transfer the batch data to GPUs
    2. calculate the inference results
  
    For each step above, we provide interface functions for you to override and make your own implementation.
* **Arguments:**
  * _**test_batch:**_ Dict  
    The input batch data received from the `test` dataloader object in the experimental pipeline.
  * _**infer_conf:**_ Dict  
    The configuration used for model inference.
* **Return:** Dict  
    A Dict of the inference results where each key-value item corresponds to one evaluation metric you want to save to the disk.

👆[Back to the API list](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#api-document)

#### bad_cases_selection_init_fn()
* **Description:**  
    This hook function returns the default bad case selection method of each _Model_ object. 
    This default value will be referred by the _Runner_ to present the top-N bad cases. 
    The original hook implementation in the base Model class returns None which means no default value.
* **Return:** List[List[str or int]]  
    The returned default value should be a list of tri-list where each tri-list is in the form of [`selection_metric`, `selection_mode`, `case_number`].  
    For example, ['wer', 'max', 50] means 50 testing waveforms with the largest WER will be selected.

👆[Back to the API list](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#api-document)

#### module_init(self, **kwargs)
* **Description:**  
    The interface function that initializes the _Module_ members of the model. These _Module_ members make up the neural network structure of the model. 
    Some models have their customized part that also needs to be initialization in this function, e.g. the tokenizer of ASR and TTS models.  
    **Note:** This interface function must be overridden for each _Model_ subclass.
* **Arguments:**
  * _****kwargs:**_  
    The combination of the arguments in your `module_conf` and `model_conf['customize_conf']`.

👆[Back to the API list](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#api-document)

#### criterion_init(self, **criterion_conf)
* **Description:**  
    The interface function that initializes the _Criterion_ members of the model. 
    These _Criterion_ members can be divided into two parts: the loss functions used for training and the evaluation metrics used for validation.  
    **Note:** This interface function must be overridden for each _Model_ subclass.
* **Arguments:**
  * _****criterion_conf:**_  
    The arguments in your given `criterion_conf`.

👆[Back to the API list](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#api-document)

#### batch_preprocess_fn(self, batch_data)
* **Description:**
  This hook function does the preprocessing for the input batch data before using them in `self.model_forward()`. 
  This function is not mandatory to be overridden and the original implementation in the base _Model_ class does nothing but return the input `batch_data`.  
  **Note:** the key names in the returned Dict should match the argument names in `self.model_forward()`.
* **Arguments:**
  * _**batch_data:**_ Dict  
    Raw data of the input batch to be preprocessed in this hook function.
* **Return:** Dict  
    Processed data of the input batch that is ready to be used in `self.model_forward()`.

👆[Back to the API list](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#api-document)

#### module_forward(self, **batch_data)
* **Description:**  
    This interface function forwards the input batch data by all _Module_ members.  
    **Note:** 
    1. This interface function must be overridden for each _Model_ subclass.
    2. The argument names should match the key names in the returned Dict of `self.batch_preprocess_fn()`.
    3. The key names in the returned Dict should match the argument names of `self.loss_calculation()` and `self.metrics_calculation()`.
* **Arguments:**
  * _****batch_data:**_  
    Processed data of the input batch received from `self.batch_preprocess_fn()`.
* **Return:** Dict  
    Prediction results (logits) of the model on the input batch data. 
    Some intermediate results (e.g., attention matrices) can also be returned for later use.

👆[Back to the API list](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#api-document)

#### criterion_forward(self, **kwargs)
* **Description:**  
    This interface function is activated after `self.model_forward()`. 
    It receives the model prediction results from `self.model_forward()` and input batch data from `self.batch_preprocess_fn()`.  
    **Note:** This interface function must be overridden for each _Model_ subclass.
* **Arguments:**
  * _****kwargs:**_ 
    The combination of the returned arguments from `self.batch_preprocess_fn()` and `self.model_forward()`.
* **Return:** (Dict[str, torch.Tensor], Dict[str, torch.Tensor]) or Dict[str, torch.Tensor]  
    The returned values should be different for the training and validation branches.
    1. For training, two Dict[str, torch.Tensor] should be returned where the first one contains all the trainable training losses for optimization and the second one contains all the non-trainable evaluation metrics used to record the training status.
    2. For validation, only one Dict[str, torch.Tensor] should be returned which contains all the non-trainable evaluation metrics used to record the validation status.

👆[Back to the API list](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#api-document)

#### visualize(self, epoch, sample_index, **valid_sample)
* **Description:**
* **Arguments:**
* **Return:**

👆[Back to the API list](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#api-document)

#### inference(self, infer_conf, **kwargs)
* **Description:**  
    This function receives the test data and test configuration. 
    The inference results will be packaged into a _Dict[str, Dict]_ which is passed to the _TestMonitor_ object for disk storage. 
    The returned _Dict_ should be in the form of
    ```python
    dict(
        {file_name}=dict(
            format={file_format},
            content={file_content}
        )
    )
    ```
    The first-level key is used to decide the name of the meta file as `idx2{file_name}`. 
    Its value is also a _Dict_ and there must be two keys in this sub-_Dict_: `format` and `content`. 
    The configuration of the sub-_Dict_ is different for different file formats:
    1. For pure text metadata files, the value of `format` must be `txt` and the value of `content` must be a _List_ of Python built-in data type (i.e.,. int, float, str, bool, ...).
    Each line of the file `idx2{file_name}` will be made up of the index of a test data instance and its metadata value in the `content` _List_ which are separated by a blank.  
    For example, `dict(cer=dict(format='txt', content=[0.1, 0.2, 0.3]))` will create a pure text file named `idx2cer` which looks like  
        ```
        {test_index1} 0.1
        {test_index2} 0.2
        {test_index3} 0.3
        ```
        **Note:** if the first-level key ends with _.md_, there will not be _'idx2'_ attached at the beginning of the file name.

    2. For audio files, the value of `format` must be either `wav` or `flac` and the value of `content` must be a _List_ of array-like data type (e.g. numpy.ndarry, torch.Tensor, ...).
    Moreover, there must be an additional key named `sample_rate` to indicate the sampling rate of the waveforms to be saved in audio files.
    There will be a folder named `{file_name}` that contains all the audio files and a pure text file named `idx2{file_name}` that contains the absolute paths of all the saved audio files.  
    For example, `dict(wav=dict(format='flac', content=[np_arr1, np_arr2, np_arr3]))` will create a folder named `wav` and a pure text file named `idx2wav` in the same directory. 
    The file `idx2wav` looks like:
         ```
         {test_index1} /x/xx/wav/{test_index1}.flac
         {test_index2} /x/xx/wav/{test_index2}.flac
         {test_index3} /x/xx/wav/{test_index3}.flac
         ```
         where `/x/xx/` is your result path given in your `exp_cfg`.

    3. For binary files, the value of `format` in the sub-_Dict_ must be `npy` and the value of `content` must be a _List_ of numpy.ndarry (torch.Tensor is not supported).
    There will be a folder named `{file_name}` that contains all the _.npy_ files and a pure text file named `idx2{file_name}` that contains the absolute paths of all the saved binary files.  
    For example, `dict(feat=dict(format='npy', content=[np_arr1, np_arr2, np_arr3]))` will create a folder named `feat` and a pure text file named `idx2feat`. 
    The `idx2feat` file is like:
       ```
       {test_index1} /x/xx/feat/{test_index1}.npy
       {test_index2} /x/xx/feat/{test_index2}.npy
       {test_index3} /x/xx/feat/{test_index3}.npy
       ```
       where `/x/xx/` is your result path given in your `exp_cfg`.
* **Arguments:**
  * _**infer_conf:**_ Dict  
    The configuration Dict used for model inference.
  * _****kwargs:**_  
    The testing data loaded from `test` dataloader object in the experimental pipeline. 
* **Return:** Dict[str, Dict[str, str or List]]  
    The model inference results to be saved on the disk.

👆[Back to the API list](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#api-document)

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#table-of-contents)

## Supported Models
1. [ASR](https://github.com/ahclab/SpeeChain/tree/main/recipes/asr#automatic-speech-recognition-asr)
   1. [asr.ASR](https://github.com/ahclab/SpeeChain/blob/main/speechain/model/asr.py#L29)  
      * **Structure:** Encoder-Decoder ASR model.  
      * **Input:** One tuple of speech-text paired data (_feat_, _feat_len_, _text_, _text_len_) in _model_forward()_.  
      * **Output:** One ASR loss calculated on the input data tuple in _criterion_calculation()_.
   2. [asr.SemiASR](https://github.com/ahclab/SpeeChain/blob/main/speechain/model/asr.py#L658) 
      * **Structure:** Semi-supervised Encoder-Decoder ASR model.  
      * **Input:** Multiple tuples of speech-text paired data  (_feat_, _feat_len_, _text_, _text_len_) in _model_forward()_. 
      Each of them is generated by a dataloader object.
      * **Output:** Multiple ASR losses calculated on all the input data tuples in _criterion_calculation()_. 
      A loss named _loss_ is also returned which is the trainable overall loss calculated by all ASR losses. 
2. [TTS](https://github.com/ahclab/SpeeChain/tree/main/recipes/tts#text-to-speech-synthesis-tts)
   1. [tts.TTS](https://github.com/ahclab/SpeeChain/blob/main/speechain/model/tts.py#L36)
      * **Structure:** Encoder-Decoder TTS model.  
      * **Input:** One tuple of speech-text paired data (_feat_, _feat_len_, _text_, _text_len_) in _model_forward()_.  
      * **Output:** One TTS loss calculated on the input data tuple in _criterion_calculation()_.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#table-of-contents)

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

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#table-of-contents)

## How to Initialize your Model by the Pretrained Model
Pretrained model loading can be easily done by giving the model path in _pretrained_model_. 
In the example below, the entire ASR model will be initialized by the given _best_accuracy.pth_ model.
```
mdl_root: recipe/asr/librispeech/train-clean-100/exp/{exp_name}/models
model:
    model_type: asr.ASR
    model_conf:
        pretrained_model:
            path: !ref <mdl_root>/accuracy_best.pth
```
If you only want to initialize a part of your model, you can use the _mapping_ argument in _pretrained_model_. 
The parameter name mismatch can also be solved by the _mapping_ argument. 
In the example below, only the encoder of the ASR model will be initialized by the given pretrained model. 
Even though the pretrained model is constructed by unit modules, it can still be loaded into the ASR model constructed by template modules by aligning their module names.
```
mdl_root: recipe/asr/librispeech/train-clean-100/exp/{exp_name}/models
model:
    model_type: asr.ASR
    model_conf:
        pretrained_model:
            path: !ref <mdl_root>/accuracy_best.pth
            mapping: 
              encoder_prenet: encoder.prenet
              encoder: encoder.encoder
```
There could be multiple pretrained models in _pretrained_model_ that are used to initialize your model. 
In the example below, the encoder and decoder of the ASR model are initialized by different pretrained models.

Note that if there are overlapping modules between the _mapping_ arguments of different pretrained models, 
the module will be initialized by the pretrained models at the back of the list.
```
mdl_root: recipe/asr/librispeech/train-clean-100/exp/{exp_name}/models
model:
    model_type: asr.ASR
    model_conf:
        pretrained_model:
            - path: !ref <mdl_root>/accuracy_best.pth
              mapping:
                encoder_prenet: encoder.prenet
                encoder: encoder.encoder
            - path: !ref <mdl_root>/10_accuracy_average.pth
              mapping:
                decoder_prenet: decoder.prenet
                decoder: decoder.decoder
                decoder_postnet: decoder.postnet
```

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#table-of-contents)