# SpeeChain Handbook
Our documentation is organized by different roles in this toolkit. 
You can start the journey of SpeeChain by your current position.

👆[Back to the toolkit README.md](https://github.com/ahclab/SpeeChain#speechain-a-pytorch-based-speechlanguage-processing-toolkit-for-the-machine-speech-chain)

## Table of Contents
1. [**For those who just discovered SpeeChain**](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#for-those-who-just-discovered-speechain)
   1. [How to dump a dataset to your machine](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#how-to-dump-a-dataset-to-your-machine)
   2. [How to prepare a configuration file](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#how-to-prepare-configuration-files)
   3. [How to train and evaluate a model](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#how-to-train-and-evaluate-a-model)
   4. [How to interpret the files generated in the _exp_ folder](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#how-to-interpret-the-files-generated-in-the-exp-folder)
2. [**For those who want to use SpeeChain for research**](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#for-those-who-want-to-use-speechain-for-research)
   1. [SpeeChain file system](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#speechain-file-system)
   2. [SpeeChain architecture workflow](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#speechain-architecture-workflow)
   3. [How to customize my own data loading and batching strategy](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#how-to-customize-my-own-data-loading-and-batching-strategy)
   4. [How to customize my own model](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#how-to-customize-my-own-model)
   5. [How to customize my own learning rate scheduling strategy](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#how-to-customize-my-own-learning-rate-scheduling-strategy)
3. [**For those who want to contribute to SpeeChain**](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#for-those-who-want-to-contribute-to-speechain)
   1. [Contribution specifications](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#contribution-specifications)


## For those who just discovered SpeeChain
In SpeeChain toolkit, a basic research pipeline has 5 steps:
1. Dump a dataset from the Internet to your disk.
2. Prepare experimental configuration files.
3. Train your model.
4. Evaluate the trained model.
5. Analyse the evaluation results.

The following subsections will explain how to execute the steps above one by one.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#table-of-contents)

### How to dump a dataset to your machine
In our toolkit, the datasets are grouped by their data types. 
Each data type corresponds a sub-folder in `/datasets`. 
In the sub-folder of each data type, each dataset has a second-level sub-folder.

SpeeChain follows the all-in-one dumping style by a bash script named `data_dumping.sh` where the procedure of dataset dumping is divided into individual steps and each step is executed by a specific script file.
We provide several dumping templates named `run.sh` under the second-level sub-folder of each dataset in `/datasets`. 
The dumping procedure is slightly different for each data type, so please refer to the **README.md** in the sub-folder of each data type before starting the dumping pipeline.

Current available datasets: (click the hyperlinks below and jump to the **README.md** of the corresponding data type)
* [Speech-Text](https://github.com/ahclab/SpeeChain/tree/main/datasets/speech_text#speech-text-datasets)
  1. LJSpeech
  2. LibriSpeech
  3. LibriTTS
* Speech-Speaker

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#table-of-contents)


### How to prepare configuration files
#### Concise Configuration Format
In order to avoid messy and redundant configuration file layout, SpeeChain provides the following services to simplify the configuration setting: 
* In-toolkit path assignment. The file path can be given by their relative location under `{SPEECHAIN_ROOT}` created by the bash script `envir_preparation.sh`. 
  For example, `speechain/runn.py` corresponds to `{SPEECHAIN_ROOT}/speechain/runner.py`.  
  If you want to specify a place outside the toolkit, you can start the file path by a slash '/' to notify the framework of an absolute path, e.g., `/abc/def/speechain/runner.py`.
* Additional !-suffixed _.yaml_ representers:
  1. **!str** allows you to cast a numerical value into a string by replacing `key_name: 10` with `key_name: !str 10`. 
     In this scenario, the variable `key_name` will be a string whose value is '10'.
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


#### Hierarchical Configuration
The configuration in this toolkit is divided into the following 4 parts to improve their reusability:
1. **Data loading and batching configuration** `data_cfg`:  
    `data_cfg` defines how the SpeeChain framework fetches the raw data from the disk and organizes them into individual batches for model training or testing. 
    These configuration files can be shared by different models in the folder of each dataset setting, i.e., `SPEECHAIN_ROOT/recipes/{dataset_name}/{setting_name}/data_cfg`.  
    For more details about the arguments, please refer to the [_Iterator_ README.md]() and [_Dataset_ README.md]().
2. **Model construction and optimization configuration** `train_cfg`:  
    `train_cfg` defines how the SpeeChain framework constructs the model and optimizes its parameters during training. 
    These configuration files are placed in the same folder as `data_cfg`, i.e., `SPEECHAIN_ROOT/recipes/{dataset_name}/{setting_name}/train_cfg`. 
    This configuration is made up of two parts: `model` for model construction configuration and `optim_sche` for model optimization configuration.  
    For more details about the arguments, please refer to the [_Model_ README.md]() and [_OptimScheduler_ README.md]().
3. **Model inference configuration** `infer_cfg`:  
    The arguments in the inference configuration are different for different models. 
    For more details, please refer to the [API document]() of each _Model_ subclass.
4. **Experimental environment configuration** `exp_cfg`:  
    `exp_cfg` is the high-level configuration given to `SPEECHAIN_ROOT/speechain/runner.py` for experimental environment initialization. 
    In `exp_cfg`, all the low-level configurations `data_cfg`, `train_cfg`, and `infer_cfg` need to be specified for model training and testing. 
    These configuration files are placed in the same folder as `data_cfg` and `train_cfg`, i.e., `SPEECHAIN_ROOT/recipes/{dataset_name}/{setting_name}/exp_cfg`.  
    The arguments that can be given in `exp_cfg` are introduced below: (The same information can also be learned by `${SPEECHAIN_PYTHON} ${SPEECHAIN_ROOT}/speechain/runner.py --help`).  
    _Group 0: Summary Configuration_
    * _**config:**_ str = None  
    The path of the all-in-one experiment configuration file. 
    You can write all the arguments in this all-in-one file instead of giving them to `{SPEECHAIN_ROOT}/speechain/runner.py` by command lines.
   
    _Group 1: Calculation and System Backend_
    * _**seed:**_ int = 0  
    Initial random seed for the experiment.
    * _**cudnn_enabled:**_ bool = True  
    Whether to activate `torch.backends.cudnn`.
    * _**cudnn_benchmark:**_ bool = False  
    Whether to activate `torch.backends.cudnn.benchmark`. 
    If True, the process of model training will be speed up and the model performance may improve somewhat.
    But your results will become less reproducible.
    * _**cudnn_deterministic:**_ bool = True  
    Whether to activate `torch.backends.cudnn.deterministic`.
    If True, it will improve the reproducibility of your experiment.
    * _**num_workers:**_ int = 1  
    The number of worker processes in the `torch.utils.data.DataLoader` of each epoch.  
    If you have complicated logic of data loading and data augmentation in the memory before passing the data to the model (e.g., speech speed perturbation, environmental noise addition, ...), 
    raising this argument may improve the speed of data loading and pre-augmentation. 
    But the choice of the argument value should be within your machine capability (i.e., the number of CPU cores).  
    If you want to debug your programs, we recommend you to set this argument to 0.
    * _**pin_memory:**_ bool = False  
    Whether to activate `pin_memory` for the Dataloader of each epoch. 
    If True, the pinned memory in the dataloaders will be activated and the data loading will be further speed up. 
    `pin_memory=True` is often used together with `non_blocking=True`. **Note** that this combination requires a large amount of memory and CPU cores.
    * _**non_blocking:**_ bool = False  
    Whether to activate `non_blocking` when transferring data from the memory to GPUs. 
    If True, the process of model training will be speed up. 
    `non_blocking=True` is often used together with `pin_memory=True`. 
    **Note** that this combination requires a large amount of memory and CPU cores.
   
    _Group 2: Gradient Calculation and Back-Propagation_
    * _**use_amp:**_ bool = True  
    Whether activate AMP (Automatic Mixed Precision) during the back-propagation. 
    If True, the GPU consumption of your model will be smaller so that you can include more data instances in a single batch.
    * _**grad_clip:**_ float = 5.0  
    Gradient clipping threshold during the back-propagation.
    * _**grad_norm_type:**_ float = 2.0  
    Normalization type used when clipping the gradients.
    * _**accum_grad:**_ int = 1  
    The number of gradient accumulation steps.  
    To mimic the gradients calculated by large batches with only a small amount of GPUs, please raise this argument. 
    The virtual batch size will become (accum_grad * the actual batch size).  
    **Note** that the model trained by accum_grad is not identical to the one actually trained by large batches because of the different randomness in each training step and the existence of BatchNorm.
    * _**ft_factor:**_ float = 1.0  
    The finetuing factor used to scale down learning rates during the parameter optimization. 
    If `ft_factor` is smaller than 1.0, the learning rates will be proportionally decreased without changing its scheduling strategy. 
    Usually, ft_factor could be set from 0.1 to 0.5 depending on your finetuning scenarios.
   
    _Group 3: Multi-GPU Distribution_
    * _**dist_backend:**_ str = 'nccl'  
    Communication backend for multi-GPU distribution. 
    If you are using NVIDIA GPUs, we recommend you set this argument to 'nccl'.
    * _**dist_url:**_ str = 'tcp://127.0.0.1'  
    Communication URL for multi-GPU distribution. 
    The default value is `tcp://127.0.0.1` for single-node distributed training and an idle port will be "automatically selected. 
    The port number cannot be set manually, which means that the argument `tcp://127.0.0.1:xxxxx` will have the same effect with `tcp://127.0.0.1`.  
    If you want to train your model on multiple nodes, please set `dist_url=env://` (**Note:** multi-node model distribution is still in beta). "
    In this case, env values of `MASTER_PORT`, `MASTER_ADDR`, `WORLD_SIZE`, and `RANK` are referred in the environment of your terminal.
    * _**world_size:**_ int = 1  
    The number of nodes for model distribution. 
    This argument is fixed to 1. Currently, we don't recommend you to modify its value. 
    If you want to conduct multi-node model distribution, please give `world_size` by `WORLD_SIZE=XXX` in your terminal 
    (**Note:** multi-node model distribution is still in beta).
    * _**rank:**_ int = 0  
    The global rank of the current node for model distribution. 
    This argument is fixed to 0. Currently, we don't recommend you to modify its value. 
    If you want to conduct multi-node model distribution, please give `rank` by `RANK=XXX` in your terminal 
    (**Note:** multi-node model distribution is still in beta)."
    * _**ngpu:**_ int = 1  
    The number of GPUs used to run your experiment. 
    If `ngpu` is larger than 1, multi-GPU model distribution will be activated.
    * _**gpus:**_ str = None  
    This argument specifies the GPUs used to run your experiment. 
    If you want to specify multiple GPUs, please give this argument in the form of 'x,x,x' where different GPUs are separated by a comma (please don't end this argument with ','). 
    Of course, you could also specify your target GPUs by `CUDA_VISIBLE_DEVICES` in the terminal.  
    If this argument is not given, the framework will automatically select `ngpu` idle GPUs. 
    * _**same_proc_seed:**_ bool = False  
    Whether to set the same initial random seed for all the GPU processes in DDP mode. 
    The different random seeds can prevent model distribution from the process homogeneity, 
    e.g., different GPU processes may have the same on-the-fly data augmentation strategy (noise addition, SpecAugment, ...) if they have the same initial random seed.  
    **Note:** please set this argument to True if you want to use random data selection for your dataloaders in the DDP mode.
   
    _Group 4: Model Training_    
    * _**train_result_path:**_ str = None  
    Where to place all the result files generated during model training. 
    If not given, `train_result_path` wil be automatically initialized to the same directory with your input `config`.  
    For example, if your input `config` is `{SPEECHAIN_ROOT}/recipes/asr/librispeech/train-960/exp_cfg/XXXXX.yaml`, 
    your `train_result_path` will be automatically initialized to `{SPEECHAIN_ROOT}/recipes/asr/librispeech/train-960/exp/XXXXX`.
    * _**train:**_ bool = False  
    Whether to go through the model training branch. 
    * _**dry_run:**_ bool = False  
    Whether to turn on the dry-running mode. 
    In this mode, only the data loading will be done to see its speed and robustness. 
    Model calculation and parameter optimization will be skipped.   
    * _**no_optim:**_ bool = False  
    Whether to turn on the no-optimization mode. 
    In this mode, only the data loading and model calculation will be done to see their speed, robustness, and memory consumption. 
    **Note:** `dry_run` has the higher priority than `no_optim`. It means that the model calculation will be skipped if you give both `--dry_run True` and `--no_optim True` in the terminal.
    * _**resume:**_ bool = False  
    Whether to resume your model training or testing experiment from the checkpoints. 
    If True, there must be .pth checkpoint files of your existing experiment in `train_result_path` or `test_result_path`. 
    This argument is shared by the training and testing branches.  
    * _**start_epoch:**_ int = 1  
    The starting epoch of your experiments. This argument will be automatically initialized by your checkpoint files if `--resume` is given. 
    * _**num_epochs:**_ int = 1000  
    The maximum number of training epochs of your experiments.
    * _**valid_per_epochs:**_ int = 1  
    The interval of going through the validation phase during training. 
    If not given, validation will be done right after parameter optimization in each epoch.
    * _**report_per_steps:**_ int = 0  
    The interval of reporting step information logs during model training or testing.  
    Positive integers mean the absolute reporting intervals that a step report will be made after each `report_per_steps` steps.  
    Negative integers mean the relative reporting intervals that there will be `-report_per_steps` reports in each epoch.  
    If not given, there will be default 10 reports in each epoch. 
    * _**best_model_selection:**_ List = None  
    The ways of selecting the best models. This argument should be given as a list of quad-tuples, i.e., (`metric_group`, `metric_name`, `metric_mode`, `model_number`).  
    `metric_group` can be either _'train'_ or _'valid'_ which indicates the group the metric belongs to;  
    `metric_name` is the name of the metric you select; 
    `metric_mode` can be either _'min'_ or _'max'_ which indicates how to select the models by this metric;  
    `model_number` indicates how many best models will be saved by this metric.   
    **Note:** the metric of the first tuple in the list will be used to do early-stopping for model training."
    * _**early_stopping_patience:**_ int = 10  
    The maximum number of epochs when the model doesn't improve its performance before stopping the model training.
    * _**early_stopping_threshold:**_ float = 0.005  
    The threshold to refresh the early-stopping status in the monitor during model training.  
    Positive float numbers in (0.0, 1.0) mean the relative threshold over the current best performance.  
    Negative float numbers main the absolute threshold over the current best performance.  
    `early_stopping_threshold=0` means no early-stopping threshold is applied to the current best performance when deciding whether to refresh the status.
    * _**last_model_number:**_ int = 10  
    The number of models saved for the last several epochs. 
    Usually, it's better to set this argument to the same value with `early_stopping_patience`.
   
    _Group 5: Real-time Model Visualization Snapshotting_
    * _**monitor_snapshot_conf:**_ Dict = {} (emtpy dictionary)  
    The configuration given to `matploblib.plot()` in `{SPEECHAIN_ROOT/speechain/snapshooter.py}` to plot curve figures for real-time model visualization during model training. 
    This argument should be given in the form of a _Dict_.
    * _**visual_snapshot_number:**_ int = 0  
    The number of the validation data instances used to make snapshots made during model visualization. 
    This argument should be smaller than the number of your validation data instances.
    * _**visual_snapshot_interval:**_ int = 5  
    The snapshotting interval of model visualization during model training. 
    This argument should be a positive integer which means that model visualization will be done once in every `visual_snapshot_interval` epochs.
   
    _Group 6: Model Testing_
    * _**test_result_path:**_ str = None  
    Where to place all the result files generated during model testing. 
    If not given, `train_result_path` wil be automatically initialized by your input `train_result_path` and `test_model`.  
    For example, if your `train_result_path` is `{SPEECHAIN_ROOT}/recipes/asr/librispeech/train-960/exp/XXXXX`, and `test_model` is `MMMMM`, 
    then your `test_result_path` will be automatically initialized to `{SPEECHAIN_ROOT}/recipes/asr/librispeech/train-960/exp/XXXXX/MMMMM/`.
    * _**test:**_ bool = False  
    Whether to go through the model testing branch.  
    * _**test_model:**_ str = None  
    The names of the model you want to evaluate during model testing. 
    If not given, `{train_result_path}/model/{test_model}.pth` will be used to initialize the parameters of the _Model_ object.  
    If you want to evaluate multiple models in one job, please give the strings of their names in a List.
    * _**bad_cases_selection:**_ List = None  
    The selection methods of the top-N bad cases during model testing. 
    This argument should be given as a list of tri-tuples (`selection_metric`, `selection_mode`, `case_number`).  
    For example, ('wer', 'max', 50) means 50 testing waveforms with the largest WER will be selected. 
    Multiple tuples can be given to present different sets of top-N bad cases.  
    If not given, the default value of your selected _Model_ subclass will be used to present top-N bad cases.
   
    _Group 7: Experiment .yaml Configuration File_
    * _**data_cfg:**_ str = None  
    The path of the configuration file for data loading and batching.  
    This argument is required for both model training and testing.
    * _**train_cfg:**_ str = None  
    The path of the configuration file for model construction and parameter optimization.  
    This argument is required for both model training (both `model` and `optim_sche` need to be given) and testing (only `model` needs to be given)."
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

Please refer to [${SPEECHAIN_ROOT}/recipes/README.md]() for more details.

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
For more details about the configuration files in `${SPEECHAIN_ROOT}/config/feat/`, please refer to the docstring of [${SPEECHAIN_ROOT}/datasets/pyscripts/feat_extractor.py]().

For more details about the configuration files in `${SPEECHAIN_ROOT}/config/infer/`, please refer to the docstring of the corresponding inference function in [${SPEECHAIN_ROOT}/speechain/infer_func/]().

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
    /speech_spk         # Datasets that are made up of speech and speaker data
```
For more details, please refer to the README.md of each type of dataset in [${SPEECHAIN_ROOT}/datasets/]().

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
    /tts                    # Recipes for the TTS task
        /libritts               # Recipes for TTS models on the LibriTTS dataset
            ...                     # different TTS settings for LibriTTS
        /ljspeech               # Recipes for TTS models on the LJSpeech dataset
            ...
    /offline_tts2asr        # Recipes for the offline TTS-to-ASR chain
        /libritts_librispeech   # Recipes for TTS trained on LibriTTS and ASR trained on LibriSpeech
            ...                     # different TTS-TO-ASR settings for LibriSpeech and LibriTTS
    /offline_asr2tts        # Recipes for the offline ASR-to-TTS chain
        /libritts   # Recipes for ASR and TTS trained on LibriTTS 
            ...                     # different ASR-to-TTS settings for LibriTTS
```
For more details, please refer to [${SPEECHAIN_ROOT}/recipes/README.md]().

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

For more details about `/speechain/criterion`, please refer to [${SPEECHAIN_ROOT}/speechain/criterion/README.md]().  
For more details about `/speechain/dataset`, please refer to [${SPEECHAIN_ROOT}/speechain/dataset/README.md]().  
For more details about `/speechain/iterator`, please refer to [${SPEECHAIN_ROOT}/speechain/iterator/README.md]().  
For more details about `/speechain/model`, please refer to [${SPEECHAIN_ROOT}/speechain/model/README.md]().  
For more details about `/speechain/module`, please refer to [${SPEECHAIN_ROOT}/speechain/module/README.md]().  
For more details about `/speechain/optim_sche`, please refer to [${SPEECHAIN_ROOT}/speechain/optim_sche/README.md]().  
For more details about `/speechain/tokenizer`, please refer to [${SPEECHAIN_ROOT}/speechain/tokenizer/README.md]().


👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#table-of-contents)


## For those who want to use SpeeChain for research
### SpeeChain architecture workflow

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#table-of-contents)


### How to customize my own data loading and batching strategy

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#table-of-contents)


### How to customize my own model

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#table-of-contents)


### How to customize my own learning rate scheduling strategy

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
    * For variable names, we recommend that the name length is better to be no more than 20 characters and the name contains no more than 3 underlines *inside* it. 
    Otherwise, it will be very hard for us to understand the meanings of your variables.
    
        If the names are longer than 20 characters, please make some abbreviations. For the abbreviations, we recommend the following 2 frequently-used strategies:
        * **Tail-Truncating**: delete the letters from the tail and only retain the part before the second vowel. 
        For example, '*convolution*' -> '*conv*', '*previous*' -> '*prev*'.
        * **Vowel-Omitting**: directly delete all vowels and some trivial consonants behind each vowel. 
        For example, '*transformer*' -> '*trfm*', '*source*' -> '*src*', '*target*' -> '*tgt*'.
    * For the temporary variables only used to register data for a short period, please add an underline at the beginning of the name to notify other users.
    For example, '*_tmp_feat_dim*' means the temporary variable used to register the intermediate value of the feature dimension. 

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#table-of-contents)











[//]: # (## Toolkit Architecture)

[//]: # (Our toolkit is consisted of the following 5 parts: )

[//]: # (1. Data preparation part)

[//]: # (2. User interaction part &#40;green&#41; )

[//]: # (3. Data loading part &#40;blue&#41;)

[//]: # (4. Model calculation part &#40;red&#41;)

[//]: # (5. Parameter optimization part &#40;brown&#41;.)

[//]: # ()
[//]: # (### [Data Preparation Part]&#40;https://github.com/ahclab/SpeeChain/tree/main/datasets&#41;)

[//]: # (We follow the ESPNET-style data preparation pipeline and provide all-in-one _.sh_ script. )

[//]: # (This script separates the entire data preparation pipeline into several individual steps, each of which is done by a specific .sh or .py file. )

[//]: # (The all-in-one _.sh_ script acts as intermediary that glues those processing scripts together. )

[//]: # (Users can easily customize their own data preparation pipeline by designing their own processing scripts for either an existing dataset or a new dataset.)

[//]: # ()
[//]: # (### User Interaction Part)

[//]: # (This part interacts with both the user and disk. The architectures of this part are different in train-valid and test branches as shown in the figures below.  )

[//]: # (**Train-valid branch:**)

[//]: # (![image]&#40;user_interaction_arch_train.png&#41;)

[//]: # (**Test branch:**)

[//]: # (![image]&#40;user_interaction_arch_test.png&#41;)

[//]: # ()
[//]: # (There are three members in this part: _Runner_, _Monitor_, and _Snapshooter_:)

[//]: # (* **Runner** is the entry of this toolkit where the experiment pipeline starts. )

[//]: # (It receives all the necessary experiment configurations from the user and distributes these configurations to the other parts of this toolkit like a hub. )

[//]: # (In each step, the runner receives the model calculation results from the model calculation part and passes the results to the monitors for recording.)

[//]: # ()
[//]: # (* **Monitor** is the processor of the model calculation results received from the runner. )

[//]: # (The raw results are converted into human-readable logs and disk-savable data files. )

[//]: # (_ValidMonitor_ also encapsulates the logic of monitoring the training process, such as best model recording, early-stopping checking, and so on. )

[//]: # (Also, a monitor keeps the connection with a _Snapshooter_ and constantly packages snapshotting materials for it to capture the intermediate information.)

[//]: # ()
[//]: # (* **Snapshooter** possesses a queue and constantly receives snapshotting materials by dequeuing the queue elements. )

[//]: # (For the program efficiency, it doesn't run in the main process of model training but a new process. )

[//]: # (It communicates with _Monitor_ by _multiprocessing.Queue_ and _multiprocessing.Event_.)

[//]: # ()
[//]: # (👆[Back to the table of contents]&#40;https://github.com/ahclab/SpeeChain/blob/main/handbook.md#table-of-contents&#41;)

[//]: # ()
[//]: # (### [Data Loading Part]&#40;https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#data-loading-part&#41;)

[//]: # (This part plays the role of extracting the raw data on the disk and providing the model with trainable batches. )

[//]: # (The architecture of the data loading part of this toolkit is shown in the figure below.)

[//]: # (![image]&#40;data_loading_arch.png&#41;)

[//]: # ()
[//]: # (There are two members in this part: _Iterator_, _Dataset_:)

[//]: # (* **Iterator** is the hub of this part. )

[//]: # (It possesses a built-in _Dataset_ and batch views that decide its accessible samples. )

[//]: # (In each epoch, it produces a _Dataloader_ that provides the built-in _Dataset_ with the sample indices to be extracted from the disk. )

[//]: # (After receiving the preprocessed vectors of all the data samples, the _Dataloader_ packages them into a trainable batch.)

[//]: # ()
[//]: # (* **Dataset** is the frontend that accesses the data files on the disk. )

[//]: # (It stores the necessary information of the data samples of the specified dataset &#40;e.g. physical addresses for the waveform files, strings for the text files&#41;. )

[//]: # (After receiving the sample index from the _Dataloader_, it loads the chosen data sample from the disk and preprocesses it into a machine-friendly vector. )

[//]: # ()
[//]: # (👆[Back to the table of contents]&#40;https://github.com/ahclab/SpeeChain/blob/main/handbook.md#table-of-contents&#41;)

[//]: # ()
[//]: # (### [Model Calculation Part]&#40;https://github.com/ahclab/SpeeChain/tree/main/speechain/model#model-calculation-part&#41;)

[//]: # (This part receives the batch from the data loading part and outputs the training losses or evaluation metrics. )

[//]: # (The architecture of the model calculation part of this toolkit is shown in the figure below.)

[//]: # (![image]&#40;model_calculation_arch.png&#41;)

[//]: # ()
[//]: # (There are three members in this part: _Model_, _Module_, _Criterion_:)

[//]: # (* **Model** is the wrapper of all the model-related operations. )

[//]: # (It encapsulates the general model services such as model construction, pretrained parameter loading, parameter freezing, and so on. )

[//]: # (It possesses several built-in _Module_ members and several built-in _Criterion_ members.)

[//]: # (After receiving the batches from the data loading part, users can choose to do some _Model_ preprocessing to fit the model's customized requirements. )

[//]: # (Then, the processed batches will be forwarded through multiple _Module_ members to obtain the model outputs. )

[//]: # (Finally, the model outputs will go through different branches for different usage.)

[//]: # ()
[//]: # (* **Module** is the unit of model forward. )

[//]: # (The job of model forward is actually done by a series of module sub-forward. )

[//]: # (It inherits `torch.nn.Module` and allows users to override its initialization and forward interfaces for customization usage.)

[//]: # ()
[//]: # (* **Criterion** does the job of converting the model output into a scalar. )

[//]: # (There are two kinds of criteria: _Train Criterion_ as the loss functions for model training and _Valid Criterion_ as the validation metrics used for model validation. )

[//]: # (The criterion and module are independent of each other, which allows any combinations of criteria in a single model.)

[//]: # (Users can create their customized criteria by overriding its initialization and forward interfaces.)

[//]: # ()
[//]: # (👆[Back to the table of contents]&#40;https://github.com/ahclab/SpeeChain/blob/main/handbook.md#table-of-contents&#41;)

[//]: # ()
[//]: # (### [Parameter Optimization Part]&#40;https://github.com/ahclab/SpeeChain/tree/main/speechain/optim_sche#parameter-optimization-part&#41;)

[//]: # (This part does the job of updating the model parameters by the received losses. )

[//]: # (The architecture of the parameter optimization part of this toolkit is shown in the figure below.)

[//]: # (![image]&#40;parameter_optimization_arch.png&#41;)

[//]: # ()
[//]: # ()
[//]: # (Unlike the traditional scheme of two separate objects &#40;optimizer and scheduler&#41;, )

[//]: # (parameter optimization and learning rate scheduling are simultaneously done by _OptimScheduler_ in this toolkit.)

[//]: # ()
[//]: # (* **OptimScheduler** is the hub of this part which encapsulates the logic of scheduling learning rates. )

[//]: # (It possesses a built-in `torch.optim.Optimizer` member and provides the learning rate for the optimizer member in each training step. )

[//]: # (Users can override its interfaces to customize their personal strategies of scheduling the learning rates during training.)

[//]: # ()
[//]: # (👆[Back to the table of contents]&#40;https://github.com/ahclab/SpeeChain/blob/main/handbook.md#table-of-contents&#41;)

