# SpeeChain Handbook
Our documentation is organized by different roles in this toolkit. 
You can start the journey of SpeeChain by your current position.

👆[Back to the home page](https://github.com/ahclab/SpeeChain#speechain-a-pytorch-based-speechlanguage-processing-toolkit-for-the-machine-speech-chain)

## Table of Contents
1. [**For those who just discovered SpeeChain**](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#for-those-who-just-discovered-speechain)
   1. [How to dump a dataset to your machine](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#how-to-dump-a-dataset-to-your-machine)
   2. [How to prepare a configuration file](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#how-to-prepare-configuration-files)
   3. [How to train and evaluate a model](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#how-to-train-and-evaluate-a-model)
   4. [How to interpret the files generated in the _exp_ folder](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#how-to-interpret-the-files-generated-in-the-exp-folder)
2. [**For those who want to use SpeeChain for research**](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#for-those-who-want-to-use-speechain-for-research)
   1. [SpeeChain file system](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#speechain-file-system)
   2. [How to customize my own data loading and batching strategy](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#how-to-customize-my-own-data-loading-and-batching-strategy)
   3. [How to customize my own model](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#how-to-customize-my-own-model)
   4. [How to customize my own learning rate scheduling strategy](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#how-to-customize-my-own-learning-rate-scheduling-strategy)
3. [**For those who want to contribute to SpeeChain**](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#for-those-who-want-to-contribute-to-speechain)
   1. [Contribution specifications](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#contribution-specifications)


## For those who just discovered SpeeChain
In SpeeChain toolkit, a basic research pipeline has 5 steps:
1. Dump a dataset from the Internet to your disk.
2. Prepare experimental configuration files.
3. Train a model.
4. Evaluate the trained model.
5. Analyse the evaluation results.

The following subsections will explain how to execute the steps above one by one.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#table-of-contents)

### How to dump a dataset to your machine
In our toolkit, the datasets are grouped by their data types. 
Each available dataset corresponds a specific folder in `${SPEECHAIN_ROOT}/datasets`:

SpeeChain follows the all-in-one dumping style by a bash script named `data_dumping.sh` where the procedure of dataset dumping is divided into individual steps and each step is executed by a specific script.

We provide an executable script named `run.sh` in each dataset folder under `${SPEECHAIN_ROOT}/datasets`. 
Please refer to [**here**](https://github.com/ahclab/SpeeChain/tree/main/datasets#how-to-dump-a-dataset-on-your-machine) before starting the dumping pipeline.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#table-of-contents)


### How to prepare configuration files
In order to avoid messy and unreadable configuration setting in the terminal, SpeeChain provides the following services to simplify the configuration setting.

#### Flexible Path Parsing Services
In SpeeChain, the path arguments can be given in 3 ways:
1. **Absolute Path:** You can indicate an absolute path by beginning the path with a slash '/', e.g., `/x/xx/xxx/speechain/runner.py`.
2. **General Relative Path:** If your input path begins with `.` or `..`, it will be converted to the corresponding absolute path in our framework.  
    **Note:** The relative path will be parsed by the directory where you execute the script rather than the directory where the executable script is placed!
3. **In-toolkit Relative Path:**  
    The path arguments can be given as the relative location under the toolkit root, i.e., `${SPEECHAIN_ROOT}`. 
    The toolkit root `${SPEECHAIN_ROOT}` is created by the bash script `envir_preparation.sh`.  
    For example, `speechain/runn.py` will be parsed to to `${SPEECHAIN_ROOT}/speechain/runner.py`. 
    If you would like to specify a place outside the toolkit root, you can directly give its absolute path with a slash `/` at the beginning to notify the framework of an absolute path, e.g., `/x/xx/xxx/speechain/runner.py`.

#### Convertable Arguments in the Terminal
Conventionally, it's hard for us to assign the values of _List_ and _Dict_ arguments in the terminal. 
In SpeeChain, our framework provides a convenient way to convert your entered strings in the specified format into the corresponding _List_ or _Dict_ variables.

1. For the _List_ variables, your entered string should be surrounded by a pair of square brackets and each element inside the brackets should be split by a comma.
   The structure can be nested to initialize sub-_List_ in the return _List_ variable.  
   For example, the string `[a,[1,2,[1.1,2.2,3.3],[h,i,j,k]],c,[d,e,[f,g,[h,i,j,k]]]]` will be parsed to
   ```
   - 'a'
   - - 1
     - 2
     - - 1.1
       - 2.2
       - 3.3
     - - 'h'
       - 'i'
       - 'j'
       - 'k'
   - 'c'
   - - 'd'
     - 'e'
     - - 'f'
       - 'g'
       - - 'h'
         - 'i'
         - 'j'
         - 'k'
   ```
2. For the _Dict_ variables, the key and its value should be split by a colon. 
   The value should be surrounded by a pair of braces if it's a sub-_Dict_. 
   The structure can be nested to initialize sub-_Dict_ in the return _Dict_ variable.  
   For example, the string `a:{b:12.3,c:{d:123,e:{g:xyz}}},g:xyz` will be parsed to
    ```
    a:
        b: 12.3
        c:
            d: 123
            e:
                g:xyz
    g: xyz
    ```
   Moreover, the _List_ string can also be nested into the _Dict_ string like `a:[1,2,3]` will be parsed as
   ```
   a:
   - 1
   - 2
   - 3
   ```

#### Concise Configuration File
As the number of arguments increases, it would be hard for us to given all the arguments one by one in the terminal. 
As a frequently-used file format for configuration, _.yaml_ has been popular in many well-known toolkits. 

In SpeeChain, we wrap the conventional _.yaml_ file and provides some advanced !-suffixed _.yaml_ representers to further simplify its layout and improve the readability:
  1. **!str** allows you to cast a numerical value into a string by replacing `key_name: 10` with `key_name: !str 10`. 
     In this scenario, the value of `key_name` will be a string '10' instead of an integer 10.
  2. **!list** allows you to compress the configuration of a list into one line from  
      ```
      key_name: 
      - - a
        - b
        - c
      - - d
        - e
        - f
      ```
      to
      ```
      key_name:
      - !list [a,b,c]
      - !list [d,e,f]
      ```
      **Note:** 
      1. The elements should be separated by commas ',' and surrounded by a pair of angle brackets '[]'. 
      2. Nested structures like `key_name: !list [!list [a,b,c],!list [d,e,f]]` are not supported yet.
  3. **!tuple** allows you to create tuples in your configuration. 
      The statement
      ```
      key_name: 
      - a
      - b
      - c
      ```
      can only give us a list, but sometimes we may need to create a tuple. Instead, we can use `key_name: !tuple (a,b,c)` to create a tuple.  
      **Note:** The elements should be separated by commas ',' and surrounded by a pair of brackets '()'. 
  4. **!ref** allows you to reuse the values you have already created by replacing
      ```
      key_name1: abc/def/ghi/jkl
      key_name2: abc/def/ghi/jkl/mno
      key_name3: abc/def/ghi/jkl/mno/pqr
      ```
      with
      ```
      key_name1: abc/def/ghi/jkl
      key_name2: !ref <key_name1>/mno
      key_name3: !ref <key_name2>/pqr
      ```
      In this scenario, the value of `key_name1` will be reused to create `key_name2` which will be further reused to create `key_name3`.  
      **Note:** 
      1. Nested structures like
          ```
          key_name1: abc/def/ghi/jkl
          key_name2: !ref <key_name1>/mno
          key_name3: !list [!ref <key_name1>,!ref <key_name2>]
          ```
          are not supported yet.
      2. Different !ref representers must be used in order. The following usage is invalid:
          ```
          key_name1: abc/def/ghi/jkl
          key_name3: !ref <key_name2>/pqr
          key_name2: !ref <key_name1>/mno
          ```
  
👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#table-of-contents)


#### Hierarchical Configuration File
The configuration files in this toolkit are divided into the following 4 parts to improve their reusability:
1. **Data loading and batching configuration** `data_cfg`:  
    `data_cfg` defines how the SpeeChain framework fetches the raw data from the disk and organizes them into individual batches for model training or testing. 
    These configuration files can be shared by different models in the folder of each dataset setting, i.e., `SPEECHAIN_ROOT/recipes/{dataset_name}/{setting_name}/data_cfg`.  
    For more details about the arguments, please refer to the [_Iterator_ README.md](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#configuration-file-format) and [_Dataset_ README.md](https://github.com/ahclab/SpeeChain/tree/main/speechain/dataset#configuration-file-format).
2. **Model construction and optimization configuration** `train_cfg`:  
    `train_cfg` defines how the SpeeChain framework constructs the model and optimizes its parameters during training. 
    These configuration files are placed in the same folder as `data_cfg`, i.e., `SPEECHAIN_ROOT/recipes/{dataset_name}/{setting_name}/train_cfg`. 
    This configuration is made up of two parts: `model` for model construction configuration and `optim_sche` for model optimization configuration.  
    For more details about the arguments, please refer to the [_Model_ README.md](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#configuration-file-format) and [_OptimScheduler_ README.md](https://github.com/ahclab/SpeeChain/tree/main/speechain/optim_sche#configuration-file-format).

    * _**infer_cfg:**_ str = None  
    The configuration for model inference during model testing. This argument is required for model testing.  
    There could be only one inference configuration or multiple configurations in *infer_cfg*:  
      1. If _infer_cfg_ is not given, the default inference configuration will be used for model inference.
      2. If you only want to give one inference configuration, please give it by either a string or a _Dict_.
         1. **String:** The string indicates where the inference configuration file is placed. For example, 
            `infer_cfg: config/infer/asr/greedy_decoding.yaml` means the configuration file `${SPEECHAIN_ROOT}/config/infer/asr/greedy_decoding.yaml` will be used for model inference. 
            In this example, the evaluation results will be saved to a folder named `greedy_decoding`.  
            If there are many arguments you need to give in the configuration, we recommend you to give them by a configuration file for concision. 
         2. **Dict:** The _Dict_ indicates the content of your inference configuration. For example.
            ```
            infer_cfg:
                beam_size: 1
                temperature: 1.0
            ```
            means that `beam_size=1` and `temperature=1.0` will be used for ASR decoding. 
            In this example, the evaluation results will be saved to a folder named `beam_size=1_temperature=1.0` which is decided by the keys and values in your given _Dict_.  
            If there are not so many arguments in your configuration, we recommend you to give them by a _Dict_ to avoid messy configuration files on your disk.
      3. If you want to give multiple inference configuration in *infer_cfg*, please give them by either a _List_ or a _Dict_.
         1. **List:** Each element in the _List_ could be either a string or a _Dict_.  
            * The string indicates the file paths of a given inference configuration. For example,
              ```
              infer_cfg:
                - config/infer/asr/greedy_decoding.yaml
                - config/infer/asr/beam_size=16.yaml
              ```
              means that both `greedy_decoding.yaml` and `beam_size=16.yaml` in `${SPEECHAIN_ROOT}/config/infer/asr/` will be used for ASR decoding.  
            * The _Dict_ indicates the content of a given inference configuration. For example,
              ```
              infer_cfg:
                - beam_size: 1
                  temperature: 1.0
                - beam_size: 16
                  temperature: 1.0
              ```
              could be used and two folders `beam_size=1_temperature=1.0` and `beam_size=16_temperature=1.0` will be created to place their evaluation results.  
            * Of course, strings and *Dict*s can be mixed in _infer_cfg_ like
              ```
              infer_cfg:
                - config/infer/asr/greedy_decoding.yaml
                - beam_size: 16
                  temperature: 1.0
              ```
         2. **Dict:** There must be two keys in the _Dict_: `shared_args` and `exclu_args`.  
            `shared_args` (short of 'shared arguments') is a _Dict_ which contains the arguments shared by all the configurations in the _Dict_.  
            `exclu_args` (short of 'exclusive arguments') is a _List[Dict]_ where each element contains the exclusive arguments for each configuration.  
            For example,
            ```
              infer_cfg:
                shared_args:
                    beam_size: 16
                exclu_args:
                    - temperature: 1.0
                    - temperature: 1.5
            ```
            means that there will be two configurations used for model inference:
            ```
            beam_size: 16
            temperature: 1.0
            ```
            and
            ```
            beam_size: 16
            temperature: 1.5
            ```
            Their evaluation results will be saved to `beam_size=16_temperature=1.0` and `beam_size=16_temperature=1.5`.  
            If your configurations don't contain too many arguments and you only want to change one or two arguments for each of them, we recommend you to give your configurations in this way.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#table-of-contents)


### How to train and evaluate a model
We provide two levels of executable bash scripts:
1. All-in-one executable `run.sh` in `${SPEECHAIN_ROOT}/recipes/`. This bash script is task-independent and can be called everywhere to run an experimental job. 
    This all-in-one script exposes some frequently-used arguments to you as follows:
   1. (optional) dry_run
   2. (optional) no_optim
   3. (optional) resume
   4. (optional) train_result_path
   5. (optional) test_result_path
   6. (required) exp_cfg
   7. (optional) data_cfg
   8. (optional) train_cfg
   9. (optional) infer_cfg
   10. (optional) ngpu
   11. (optional) gpus
   12. (optional) num_workers
   13. (required) train
   14. (required) test
   
   For other arguments, please give them in an overall configuration file by `--exp_cfg`.
2. Low-level `run.sh` designed for each subset of each task folder in `${SPEECHAIN_ROOT}/recipes/`. Those scripts are used to run the experiments of the specific subset for the specific task.

The execution hierarchy of the scripts is like:
```
${SPEECHAIN_ROOT}/recipes/{task_name}/{dataset_name}/{subset_name}/run.sh
    --->${SPEECHAIN_ROOT}/recipes/run.sh
        --->${SPEECHAIN_ROOT}/speechain/runner.py
```
For more details about those bash scripts, please move to their directories and use the command `bash run.sh --help` or `bash run.sh -h` to see the help message.

By the way, you can also directly use the command `${SPEECHAIN_PYTHON} ${SPEECHAIN_ROOT}/speechain/runn.py` in your terminal or your own bash script to run your experimental jobs. 
Before doing so, we recommend you to first use the command `${SPEECHAIN_PYTHON} ${SPEECHAIN_ROOT}/speechain/runn.py --help` to familiarize yourself with the involved arguments.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#table-of-contents)



### How to interpret the files generated in the _exp_ folder

Please refer to [${SPEECHAIN_ROOT}/recipes/README.md](https://github.com/ahclab/SpeeChain/tree/main/recipes#experimental-file-system) for more details.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#table-of-contents)


## For those who want to use SpeeChain for research
### SpeeChain file system
#### Configuration Folder
This folder contains off-the-shelf configuration files that can be shared across different tasks, models, or datasets. 
Each type of configuration corresponds to a specific sub-folder where each category of configuration corresponds to a specific sub-sub-folder.

Folder architecture is shown below:
```
/config
    /feat       # Configuration for acoustic feature extraction
        /log_mel    # Configuration files for log-Mel spectrogram extraction
            /...
        /mfcc       # Configuration files for MFCC extraction
            /...
    /infer      # Configuration for model inference
        /asr        # Configuration files for ASR inference
            /...
        /tts        # Configuration files for TTS inference
            /...
```
For more details about the configuration files in `${SPEECHAIN_ROOT}/config/feat/`, please refer to the docstring of [${SPEECHAIN_ROOT}/datasets/pyscripts/feat_extractor.py](https://github.com/ahclab/SpeeChain/blob/main/datasets/pyscripts/feat_extractor.py).

For more details about the configuration files in `${SPEECHAIN_ROOT}/config/infer/`, please refer to the docstring of the corresponding inference function in [${SPEECHAIN_ROOT}/speechain/infer_func/](https://github.com/ahclab/SpeeChain/tree/main/config/infer).

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#table-of-contents)


#### Dataset Folder
This folder contains off-the-shelf processing scripts to dump datasets into your machine. 
Each type of datasets corresponds to a specific sub-folder where each dataset corresponds a specific sub-sub-folder.

Folder architecture is shown below:
```
/datasets
    /speech_text        # Datasets that are made up of speech and text data
        /librispeech        # Processing scripts for the LibriSpeech dataset
            /...
        /libritts           # Processing scripts for the LibriTTS dataset
            /...
        /ljspeech           # Processing scripts for the LJSpeech dataset
            /...
        /data_dumping.sh    # all-in-one speech-text dataset dumping script
```
For more details, please refer to the README.md of each type of dataset in [${SPEECHAIN_ROOT}/datasets/](https://github.com/ahclab/SpeeChain/tree/main/datasets).

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#table-of-contents)


#### Recipe Folder
This folder contains our recipes for all tasks on the available datasets. 
Each task corresponds to a specific sub-folder where each dataset corresponds a specific sub-sub-folder.
In the dataset folder, there may be some sub-folders corresponding to different settings of model training where a sub-sub-folder `/data_cfg` contains all the configuration files of data loading that are shared by all the model sub-sub-folders.

Folder architecture is shown below:
```
/recipes
    /asr                    # Recipes for the ASR task
        /librispeech            # Recipes for ASR models on the LibriSpeech dataset
            ...                     # different ASR settings for LibriSpeech
        /libritts               # Recipes for ASR models on the LibriTTS dataset
            ...                     # different ASR settings for LibriTTS
        /libritts+librispeech   # Recipes for ASR models on the 16khz-downsampled LibriTTS and LibriSpeech datasets
            ...                     # different ASR settings for 16khz-downsampled LibriTTS and LibriSpeech
    /tts                    # Recipes for the TTS task
        /libritts               # Recipes for TTS models on the LibriTTS dataset
            ...                     # different TTS settings for LibriTTS
        /ljspeech               # Recipes for TTS models on the LJSpeech dataset
            ...
    /offline_tts2asr        # Recipes for the offline TTS-to-ASR chain
        /libritts_librispeech   # Recipes for TTS trained on LibriTTS and ASR trained on LibriSpeech
            ...                     # different TTS-to-ASR settings for LibriSpeech and LibriTTS
    /offline_asr2tts        # Recipes for the offline ASR-to-TTS chain
        /libritts                # Recipes for ASR and TTS trained on LibriTTS 
            ...                     # different ASR-to-TTS settings for LibriTTS
```
For more details, please refer to [${SPEECHAIN_ROOT}/recipes/README.md](https://github.com/ahclab/SpeeChain/tree/main/recipes#recipes-folder-of-the-speechain-toolkit).

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#table-of-contents)

    
#### Main Folder
The folder `/speechain` is the core part of our toolkit where each sub-folder corresponds to a specific part of an experimental pipeline. 
In each sub-folder, there is a .py file named `abs.py` that declares the abstract class of the corresponding pipeline part. 
Based on the abstract class, many implementation classes are included in the same sub-folder with the name like `xxx.py`.

```
/speechain
    # Sub-folders for all specific parts of an experimental pipeline
    /criterion
        ...
    /dataset
        ...
    /infer_func
        /beam_search.py     # Inference function of the beam searching. Mainly used for ASR models.
        /tts_decoding.py    # Inference function of the autoregressive TTS decoding.
        ...
    /iterator
        ...
    /model
        ...
    /module
        ...
    /optim_sche
        ...
    /tokenizer
        ...
    # General part of the pipeline
    /run.py             # The entrance of SpeeChain toolkit for both model training and testing.
    /monitor.py         # The training and testing monitors. Used to record and regulate the training and testing process.
    /snapshooter.py     # The figure snapshooter. Used to transform the input snapshotting materials into the visible figures.
```

For more details about `/speechain/criterion`, please refer to [${SPEECHAIN_ROOT}/speechain/criterion/README.md](https://github.com/ahclab/SpeeChain/tree/main/speechain/criterion#criterion).  
For more details about `/speechain/dataset`, please refer to [${SPEECHAIN_ROOT}/speechain/dataset/README.md](https://github.com/ahclab/SpeeChain/tree/main/speechain/dataset).  
For more details about `/speechain/iterator`, please refer to [${SPEECHAIN_ROOT}/speechain/iterator/README.md](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#iterator).  
For more details about `/speechain/model`, please refer to [${SPEECHAIN_ROOT}/speechain/model/README.md](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#model).  
For more details about `/speechain/module`, please refer to [${SPEECHAIN_ROOT}/speechain/module/README.md](https://github.com/ahclab/SpeeChain/tree/main/speechain/module#module).  
For more details about `/speechain/optim_sche`, please refer to [${SPEECHAIN_ROOT}/speechain/optim_sche/README.md](https://github.com/ahclab/SpeeChain/tree/main/speechain/optim_sche#optimscheduler).  
For more details about `/speechain/tokenizer`, please refer to [${SPEECHAIN_ROOT}/speechain/tokenizer/README.md](https://github.com/ahclab/SpeeChain/tree/main/speechain/tokenizer#tokenizer).

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#table-of-contents)


### How to customize my own data loading and batching strategy
For how to customize your own data loading strategy, please refer to the [API document](https://github.com/ahclab/SpeeChain/tree/main/speechain/dataset#api-document) of `/speechain/dataset`.  

For how to customize your own data batching, please refer to the [API document](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#api-document) of `/speechain/iterator`.  

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#table-of-contents)


### How to customize my own model
For how to customize your own model, please refer to the [API document](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#api-document) of `/speechain/model`.

If the existing _Module_ implementations in `/speechain/module`, you can refer to the [API document](https://github.com/ahclab/SpeeChain/tree/main/speechain/module#api-document) of `/speechain/module` for the instructions about how to customize your own modules.

For the model involving text tokenization like ASR and TTS, if the existing _Tokenizer_ implementations cannot satisfy your needs, you can refer to the [API document](https://github.com/ahclab/SpeeChain/tree/main/speechain/tokenizer#api-document) of `/speechain/tokenizer` for the instructions about how to customize your own tokenizers.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#table-of-contents)


### How to customize my own learning rate scheduling strategy
For how to customize your own optimization strategy, please refer to the [API document](https://github.com/ahclab/SpeeChain/tree/main/speechain/optim_sche#api-document) of `/speechain/optim_sche`.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#table-of-contents)


## For those who want to contribute to SpeeChain
### Contribution specifications
We have some specifications for you to standardize your contribution:

1. **Documentation**: We will appreciate it a lot if you could provide enough documents for your contribution. 
    * Please give a header string at the top of the code file you made in the following format to notify others of your contribution:
        ```
            Author: Heli Qi
            Affiliation: NAIST (abbreviation is OK)
            Date: 2022.08 (yyyy.mm is enough)
        ```
    * We recommend you to use the Google-style function docstring. 
    If you are using PyCharm, you can set the docstring style in File→Setting→Tools→Python Integrated Tools→Docstrings→Docstring format.
    
        As for argument explanation in the docstring, we recommend you to write the argument type after the colon and give its description below with a tab retract as follows.
        ```
            Args:
                d_model: int
                    The dimension of the input feature sequences.
        ```
        If the argument type is `torch.Tensor` or `numpy.array`, please replace the type with its shape as follows.
        ```
            Args:
                emb_feat: (batch_size, seq_len, d_model)
                    Embedded input feature sequences
        ```
    * For in-line comments, we recommend you start a new line every time you want to comment (it's better not to type a long comment after the code). 
    The codes are better to be divided into several code blocks by their roles with an in-line comment right above the block as follows.
      ```
        # member registration
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.layernorm_first = layernorm_first
      ```
    
2. **Naming**: We have several recommendations for class names and variable names.
    * For class names, we recommend you to name your class in the CamelCase style. The names are better to be in the form of "What is it made up of" + "What is it". 
    
        For example, `SpeechTextDataset` means a dataset class that returns speech-text paired data during training. 
        `Conv2dPrenet` means a prenet module that is made up of Conv2d layers.
    * For long variable names, please make some abbreviations. For the abbreviations, we recommend the following 2 frequently-used strategies:
        * **Tail-Truncating**: delete the letters from the tail and only retain the part before the second vowel. 
        For example, '*convolution*' -> '*conv*', '*previous*' -> '*prev*'.
        * **Vowel-Omitting**: directly delete all vowels and some trivial consonants behind each vowel. 
        For example, '*transformer*' -> '*trfm*', '*source*' -> '*src*', '*target*' -> '*tgt*'.
    * For the temporary variables only used to register data for a short period, please add an underline at the beginning of the name to notify other users.
    For example, '*_tmp_feat_dim*' means the temporary variable used to register the intermediate value of the feature dimension. 

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#table-of-contents)

