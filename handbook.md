# SpeeChain Documentation
Our documentation is organized by different roles in this toolkit. 
You can start the journey of SpeeChain by your current position.

## Table of Contents
1. [**For those who just discovered SpeeChain**]()
   1. [How to dump a dataset to your machine]()
   2. [How to prepare a configuration file]()
   3. [How to train a model]()
   4. [How to interpret the files generated during training]()
   5. [How to evaluate a trained model]()
   6. [How to analyse a trained model by the files generated during evaluation]()
2. [**For those who want to use SpeeChain for research**]()
   1. [What does the SpeeChain framework work during model training and testing]()
   2. [How to customize my own data loading strategy]()
   3. [How to customize my own model]()
   4. [How to customize my parameter optimization strategy]()
   5. [Are there any specifications for my contribution]()


## For those who just discovered SpeeChain
In SpeeChain toolkit, a basic research pipeline has 5 steps:
1. Dump a dataset from the Internet to your disk.
2. Prepare experimental configuration files.
3. Train your model.
4. Evaluate the trained model.
5. Analyse the evaluation results.

The following subsections will explain how to execute the steps above one by one.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)

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
* [Speech-Speaker]()

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)


### How to prepare configuration files
2. **Configuration Preparation**:
    1. Move to the sub-folder of your target dataset and task in `./recipes/`.
    2. Pick up an experiment configuration file you would like to run the experiment in _exp_cfg_.
    3. Replace the reference values on the top of the _exp_cfg_ file to the places on your machine.
    4. Don't forget to also change the reference values in the _data_cfg_ file used in your selected _exp_cfg_ file.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)


### How to train a model
3. **Model Training**:
    1. Move to the folder of one of the subset of your target dataset in *./recipe/*.
    2. Activate the environment *speechain* by the command `conda activate speechain`
    3. Open the template script by `vim run.sh` and change the arguments in it if you need.
    4. Start the experiment by running `./run.sh`.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)


### How to interpret the files generated during training

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)


### How to evaluate a trained model

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)


### How to analyse a trained model by the files generated after testing

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)


## For those who want to use SpeeChain for research
### File System
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

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)


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
For more details, please refer to the README.md of each type of dataset in `/datasets/`.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)


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
            ...                     # different ASR settings for LibriSpeech
    /offline_asr2tts        # Recipes for the offline ASR-to-TTS chain
        /librispeech_libritts   # Recipes for ASR trained on LibriSpeech and TTS trained on LibriTTS 
            ...                     # different ASR-to-TTS settings for LibriSpeech
```
For more details, please refer to [_/recipes/README.md_]().

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)

    
#### Main Folder
The folder `/speechain` is the core part of our toolkit where each sub-folder corresponds to a specific part of an experimental pipeline. 
In each sub-folder, there is a .py file named `abs.py` that declares the abstract class of the corresponding pipeline part. 
Based on the abstract class, many implementation classes are included in the same sub-folder with the name like `xxx.py`.

For the role of each part, please refer to the [toolkit architecture]() section below.

##### Folder Overview
```
/speechain
    # Sub-folders for all specific parts of an experimental pipeline
    /criterion
        ...
    /dataset
        ...
    /infer_func
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

##### Criterion Sub-folder
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

##### Dataset Sub-folder
```
/speechain
    /dataset
        /abs.py             # Abstract class of Dataset. Base of all Dataset implementations.
        /speech_text.py     # Dataset implementation of speech-text datasets. Mainly used for ASR and TTS models.
```

##### Inference Function Sub-folder
```
/speechain
    /infer_func
        /beam_search.py     # Inference function of the beam searching. Mainly used for ASR models.
        /tts_decoding.py    # Inference function of the autoregressive TTS decoding.
```

##### Iterator Sub-folder
```
/speechain
    /iterator
        /abs.py         # Abstract class of Iterator. Base of all Iterator implementations.
        /block.py       # Iterator implementation of the block strategy (variable utterances per batch). Mainly used for ASR and TTS training.
        /piece.py       # Iterator implementation of the piece strategy (fixed utterances per batch). Mainly used for ASR and TTS evaluation.
```

##### Model Sub-folder
```
/speechain
    /model
        /abs.py     # Abstract Model class. Base of all Model implementations.
        /asr.py     # Initial Model implementation of autogressive ASR. Base of all advanced ASR models.
        /tts.py     # Initial Model implementation of autogressive TTS. Base of all advanced TTS models.
```

##### Module Sub-folder
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

##### OptimScheduler Sub-folder
```
/speechain
    /optim_sche
        /abs.py     # Abstract class of OptimScheduler. Base of all OptimScheduler implementations.
        /noam.py    # OptimScheduler implementation of the Noam scheduler. Mainly used for Transformer training.
```

##### Tokenizer Sub-folder
```
/speechain
    /tokenizer
        /abs.py         # Abstract class of Tokenizer. Base of all Tokenizer implementations.
        /char.py        # Tokenizer implementation of the character tokenizer.
        /subword.py     # Tokenizer implementation of the subword tokenizer by SentencePiece package.
```

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)



### Development Specifications
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
        # para recording
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

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)











## Toolkit Architecture
Our toolkit is consisted of the following 5 parts: 
1. Data preparation part
2. User interaction part (green) 
3. Data loading part (blue)
4. Model calculation part (red)
5. Parameter optimization part (brown).

### [Data Preparation Part](https://github.com/ahclab/SpeeChain/tree/main/datasets)
We follow the ESPNET-style data preparation pipeline and provide all-in-one _.sh_ script. 
This script separates the entire data preparation pipeline into several individual steps, each of which is done by a specific .sh or .py file. 
The all-in-one _.sh_ script acts as intermediary that glues those processing scripts together. 
Users can easily customize their own data preparation pipeline by designing their own processing scripts for either an existing dataset or a new dataset.

### User Interaction Part
This part interacts with both the user and disk. The architectures of this part are different in train-valid and test branches as shown in the figures below.  
**Train-valid branch:**
![image](user_interaction_arch_train.png)
**Test branch:**
![image](user_interaction_arch_test.png)

There are three members in this part: _Runner_, _Monitor_, and _Snapshooter_:
* **Runner** is the entry of this toolkit where the experiment pipeline starts. 
It receives all the necessary experiment configurations from the user and distributes these configurations to the other parts of this toolkit like a hub. 
In each step, the runner receives the model calculation results from the model calculation part and passes the results to the monitors for recording.

* **Monitor** is the processor of the model calculation results received from the runner. 
The raw results are converted into human-readable logs and disk-savable data files. 
_ValidMonitor_ also encapsulates the logic of monitoring the training process, such as best model recording, early-stopping checking, and so on. 
Also, a monitor keeps the connection with a _Snapshooter_ and constantly packages snapshotting materials for it to capture the intermediate information.

* **Snapshooter** possesses a queue and constantly receives snapshotting materials by dequeuing the queue elements. 
For the program efficiency, it doesn't run in the main process of model training but a new process. 
It communicates with _Monitor_ by _multiprocessing.Queue_ and _multiprocessing.Event_.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)

### [Data Loading Part](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#data-loading-part)
This part plays the role of extracting the raw data on the disk and providing the model with trainable batches. 
The architecture of the data loading part of this toolkit is shown in the figure below.
![image](data_loading_arch.png)

There are two members in this part: _Iterator_, _Dataset_:
* **Iterator** is the hub of this part. 
It possesses a built-in _Dataset_ and batch views that decide its accessible samples. 
In each epoch, it produces a _Dataloader_ that provides the built-in _Dataset_ with the sample indices to be extracted from the disk. 
After receiving the preprocessed vectors of all the data samples, the _Dataloader_ packages them into a trainable batch.

* **Dataset** is the frontend that accesses the data files on the disk. 
It stores the necessary information of the data samples of the specified dataset (e.g. physical addresses for the waveform files, strings for the text files). 
After receiving the sample index from the _Dataloader_, it loads the chosen data sample from the disk and preprocesses it into a machine-friendly vector. 

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)

### [Model Calculation Part](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#model-calculation-part)
This part receives the batch from the data loading part and outputs the training losses or evaluation metrics. 
The architecture of the model calculation part of this toolkit is shown in the figure below.
![image](model_calculation_arch.png)

There are three members in this part: _Model_, _Module_, _Criterion_:
* **Model** is the wrapper of all the model-related operations. 
It encapsulates the general model services such as model construction, pretrained parameter loading, parameter freezing, and so on. 
It possesses several built-in _Module_ members and several built-in _Criterion_ members.
After receiving the batches from the data loading part, users can choose to do some _Model_ preprocessing to fit the model's customized requirements. 
Then, the processed batches will be forwarded through multiple _Module_ members to obtain the model outputs. 
Finally, the model outputs will go through different branches for different usage.

* **Module** is the unit of model forward. 
The job of model forward is actually done by a series of module sub-forward. 
It inherits `torch.nn.Module` and allows users to override its initialization and forward interfaces for customization usage.

* **Criterion** does the job of converting the model output into a scalar. 
There are two kinds of criteria: _Train Criterion_ as the loss functions for model training and _Valid Criterion_ as the validation metrics used for model validation. 
The criterion and module are independent of each other, which allows any combinations of criteria in a single model.
Users can create their customized criteria by overriding its initialization and forward interfaces.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)

### [Parameter Optimization Part](https://github.com/ahclab/SpeeChain/tree/main/speechain/optim_sche#parameter-optimization-part)
This part does the job of updating the model parameters by the received losses. 
The architecture of the parameter optimization part of this toolkit is shown in the figure below.
![image](parameter_optimization_arch.png)


Unlike the traditional scheme of two separate objects (optimizer and scheduler), 
parameter optimization and learning rate scheduling are simultaneously done by _OptimScheduler_ in this toolkit.

* **OptimScheduler** is the hub of this part which encapsulates the logic of scheduling learning rates. 
It possesses a built-in `torch.optim.Optimizer` member and provides the learning rate for the optimizer member in each training step. 
Users can override its interfaces to customize their personal strategies of scheduling the learning rates during training.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)

