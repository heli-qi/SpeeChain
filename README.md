# The SpeeChain Toolkit: A PyTorch-based Toolkit designed for ASR-TTS Joint Research
**_SpeeChain_** is an open-source PyTorch-based deep learning toolkit made by the [**_AHC lab_**](https://ahcweb01.naist.jp/en/) at Nara Institute of Science and Technology (NAIST). 
This toolkit is designed for simplifying the joint research on speech recognition and speech synthesis models. 

If you are interested in our previous research on the _Machine Speech Chain_, you can also find our research codes in this toolkit😀. 
The paper references are as follows: 
1. [Listening while speaking: Speech chain by deep learning](https://arxiv.org/pdf/1707.04879)
2. [Machine speech chain with one-shot speaker adaptation](https://arxiv.org/pdf/1803.10525)
3. [Machine speech chain](https://ieeexplore.ieee.org/iel7/6570655/8938144/09020132.pdf)
4. [Incremental machine speech chain towards enabling listening while speaking in real-time](https://arxiv.org/pdf/2011.02126)

_SpeeChain is currently in beta._ Contribution to this toolkit is warmly welcomed! 

If you find our toolkit helpful for your research, we would appreciate it a lot if you could give us a star⭐ (see the top-right corner of this page)! 
Anytime you meet a problem when using our toolkit, please don't hesitate to leave us an issue!

## Table of Contents
1. [**Toolkit Characteristics**](https://github.com/ahclab/SpeeChain#toolkit-characteristics)
    1. [General services](https://github.com/ahclab/SpeeChain#general-services)
    2. [Data loading services](https://github.com/ahclab/SpeeChain#data-loading-services)
    3. [Pseudo labeling services](https://github.com/ahclab/SpeeChain#pseudo-labeling-services)
    4. [Model services](https://github.com/ahclab/SpeeChain#model-services)
    5. [Optimization services](https://github.com/ahclab/SpeeChain#optimization-services)
    6. [User-friendly documents and interfaces](https://github.com/ahclab/SpeeChain#user-friendly-documents-and-interfaces)
2. [**File System**](https://github.com/ahclab/SpeeChain#file-system)
3. [**Toolkit Architecture**](https://github.com/ahclab/SpeeChain#toolkit-architecture)
    1. [User Interaction Part](https://github.com/ahclab/SpeeChain#user-interaction-part)
    2. [Data Loading Part](https://github.com/ahclab/SpeeChain#data-loading-part)
    3. [Model Calculation Part](https://github.com/ahclab/SpeeChain#model-calculation-part)
    4. [Parameter Optimization Part](https://github.com/ahclab/SpeeChain#parameter-optimization-part)
4. [**Support Models**](https://github.com/ahclab/SpeeChain#support-models)
    1. [Speech Recognition](https://github.com/ahclab/SpeeChain#speech-recognition)
    2. [Speech Synthesis](https://github.com/ahclab/SpeeChain#speech-synthesis)
    3. [Speaker Recognition](https://github.com/ahclab/SpeeChain#speaker-recognition)
5. [**Available Datasets**](https://github.com/ahclab/SpeeChain#available-datasets)
    1. [Speech-Text](https://github.com/ahclab/SpeeChain#speech-text)
    2. [Speech-Speaker](https://github.com/ahclab/SpeeChain#speech-speaker)
6. [**Get a Quick Start**](https://github.com/ahclab/SpeeChain#get-a-quick-start)
    1. [Installation](https://github.com/ahclab/SpeeChain#installation)
    2. [Usage](https://github.com/ahclab/SpeeChain#usage)
    3. [Customization](https://github.com/ahclab/SpeeChain#customization)
7. [**Contribution**](https://github.com/ahclab/SpeeChain#contribution)
    1. [Current Developers](https://github.com/ahclab/SpeeChain#current-developers)
    2. [Development Documentation](https://github.com/ahclab/SpeeChain#development-documentation)
    3. [Development Specifications](https://github.com/ahclab/SpeeChain#development-specifications)
8. [**Common Q&A**](https://github.com/ahclab/SpeeChain#common-q&a)

## Toolkit Characteristics
### General services
1. **On-the-fly acoustic feature extraction**
2. **On-the-fly SpecAugment**
3. **Single-node multi-GPU model distribution based on DDP (Distributed Data Parallel)**
4. **Tensorboard real-time model visualization**
5. **Model inference during training**
6. **Detailed model diagnosing during testing**

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)

### Data loading services
We provide various data loading services that enable users to easily conduct experiments across different datasets: 
1. **Customized combinations of training, validation, and testing datasets**.
The datasets used for model training, validation, and testing can be easily changed by your configuration. 
They don't have to come from the same data source.
This customization enables us to easily conduct some advanced experiments, such as semi-supervised learning, multimodal learning, and domain adaptation.
2. **Multiple dataloaders can be initialized for training your model.** 
This means that data samples from multiple datasets can be fetched into a single batch for training and validating your models.
These dataloaders are independent of each other, so they could hold their specific datasets and batch generation strategies.
3. **Multiple datasets can be used to initialize a single dataloader.** 
These datasets are mixed up to form a mixed dataset and the dataloader uniformly fetches data-label pairs from the mixed dataset.
4. **Data selection can be done in the dataset of each dataloader.** 
Sometimes we may only want to use a part of the dataset to train our models. 
Data selection enables us to dynamically initialize our dataloader with a part of data samples in its dataset. 
With data selection, we can be freed from the annoying work of segmenting and dumping partial datasets before model training. 
Off-the-shelf selection methods include: 
    1. random selection
    2. selection from the beginning or the end
    3. selection by metadata
      
👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)

### Pseudo labeling services
For the research on semi-supervised learning and domain adaptation, technical work about processing unlabeled data is inevitable 
(e.g. hypothesis transcripts for untranscribed speech or synthetic utterances for unspoken text).
We provide various pseudo-labeling services for unlabeled data to help researchers save more energy for their research:
1. **GPU-aid batch-level pseudo data generation.** 
As a necessary part of semi-supervised ASR and TTS models, pseudo data generation often consumes plenty of time for large-scale datasets (such as _LibriSpeech_ or _GigaSpeech_). 
The faster we get pseudo data, the more experiments we can conduct to realize our ideas. 
In our toolkit, this job is done at the batch level with the aid of multi-GPUs.
2. **Off-the-shelf correction functions for pseudo labels.** 
Pseudo-labeled data usually need to go through some corrections before being used to train our models. 
We provide some off-the-shelf correction functions to free our users from these annoying dirty jobs:
    1. Pseudo text correction:
        1. During-inference correction:
            1. language model joint decoding
            2. hypothesis length adjustment
            3. lexicon-restricted decoding
        2. After-inference correction:
            1. language model hypothesis reweighing
            2. n-gram looping phrase removal
            3. lexicon calibration
    2. Pseudo speech correction:
        1. silence removal
        2. non-speech sound removal
        3. synthesized artifact removal

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)

### Model services
The models in our toolkit are built with multiple independent modules. Independence offers the following merits:
1. **Simple model construction.** 
Users can easily build their models by assembling different modules. 
Since our toolkit supports dynamic class importing, model construction can be easily done merely by the model configuration file.
2. **Flexible pretrained model loading.** 
Multiple pretrained models can be used to initialize different parts of your models. 
Also, the mismatch between the parameter names of pretrained models and your models can be easily solved by the configuration.
3. **Flexible parameters freezing.** 
The parameter freezing can also be done independently for different parts of your models. 
The freezing granularity can be very fine (even to a specific neural layer) if you give the proper configuration.

Moreover, our toolkit enables our users to **perform model inference during training**, which visualizes the evolution of the model during training.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)

### Optimization services
We provide various optimization services that offer users the possibilities for more advanced model training schemes:
1. **Multi-losses training with multiple optimizers.** 
Multiple optimizers can be used to train your models. 
Each optimizer has specific target losses, target model parameters, learning rate scheduling strategy, and optimization interval. 
2. **Gradient accumulation for large-batch training by limited computing resources.** 
Our toolkit enables our users to accumulate the gradients from multiple small batches to mimic the one calculated by a large batch. 
The parameter optimization can be done after several steps of gradient accumulation. 
It becomes possible for you to enjoy training the large-batch models even though you are suffering from the lack of GPUs! 
3. **Simple finetuing setting.** 
In our toolkit, the learning rates can be easily scaled down without changing the scheduling hyperparameters. 
The troublesome jobs of making a new learning rate scheduler configuration have gone!

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)

### User-friendly documents and interfaces
We provide many interfaces for users to customize their experiment pipelines. 
Most of the interfaces return a *Dict* where users can freely design the contents of the returned results.

We also provide sufficient documents to explain the details of our toolkit. 
Please refer to the [Documentation](https://github.com/ahclab/SpeeChain#development-documentation) section for more details.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)

## File System
The description of the file system in this toolkit is shown below:
```
/config                 # the folder containing all the configuration files
    /feat                   # feature configuration files for manual feature extraction
        ...                     
    /infer                  # inference configuration files
        ...                     # each inference method has a specific sub-folder

/datasets                # the folder containing all the datasets and their processing scripts, each data-label type has a sub-folder
    /speech_text             # all the datasets made up of speech and text data
        /librispeech            # the data and scripts for LibriSpeech dataset
        /ljspeech               # the data and scripts for LJSpeech dataset
        ...
    ...

/speechain                # the main folder of the toolkit
    /criterion              # the sub-folder containing all the Criterion objects, each criterion object corresponds to a .py file
        /abs.py                 # the abstract class of Criterion, base class of all criteria.
        ...
    /dataset                # the sub-folder containing all the Dataset objects, each subfolder corresponds to a data type
        /abs.py                 # the abstract class of Dataset, base model of all datasets.
        /speech                 # the sub-folder containing all datasets class related to speech data.
            /speech_text.py         # the dataset fetching speech-text pairs
            ...
        ...
    /infer_func             # the sub-folder containing all the inference functions, each function corresponds to a specific .py file
    /iterator               # the sub-folder containing all the Iterator objects, each file corresponds to a batch generation strategy
        /abs.py                 # the abstract class of Iterator, base class of all iterators.
        ...
    /model                  # the sub-folder containing all the Model objects, each file corresponds to a model
        /abs.py                 # the abstract class of Model, base class of all models.
        ...
    /module                 # the sub-folder containing all the Module objects, each sub-folder corresponds to a category of modules
        /abs.py                 # the abstract class of Module, base class of all modules.
        /frontend               # the subfolder containing all frontend for acoustic feature extraction, each kind of acoustic feature corresponds to a .py file
            /speech2mel.py          # input is speech, output is log-mel spectrogram
            ...
        /transformer            # the subfolder containing all Transformer-related modules
            /pos_enc.py             # the different implementations of positional encoding layers 
            /attention.py           # the different implementations of multi-head attention layers 
            /feed_forward.py        # the different implementations of positionwise feedforward layers 
            /encoder.py             # transformer encoder layers 
            /decoder.py             # transformer decoder layers 
        ...
    /optim_sche         # the sub-folder containing all the OptimScheduler objects, each lr scheduling strategy corresponds to a .py file
        /abs.py                 # the abstract class of OptimScheduler, base class of all optimschedulers.
        ... 
    /tokenizer              # the sub-folder containing all the Tokenizer objects, each tokenization strategy corresponds to a .py file
        /abs.py                 # the abstract class of Tokenizer, base class of all tokenizer.
        ...
    /utilbox                # the utility functions used in the toolkit
        ...
    monitor.py              # the monitor
    snapshooter.py          # the snapshooter 
    runner.py               # the runner, entrance of this toolkit.

/recipes                # the folder containing the recipes for all tasks on the available datasets, each task has a specific sub-folder
    /asr                    # the sub-folder for ASR, each dataset has a sub-folder
        /librispeech            # the sub-folder for librispeech dataset, each subset has a sub-folder
            /train_clean_100        # the sub-folder for train_clean_100 subset, each model type has a sub-folder                
                /transformer            # the sub-folder for transformer-based ASR
                    /exp                    # the sub-folder containing all experimental results, each experiment has a specific sub-folder
                    /exp_cfg                # the sub-folder containing all experiment configuration files
                    /train_cfg              # the sub-folder containing all training configuration files
                ...     
                /data_cfg               # the sub-folder containing all data configuration files of the specific subset. These files are shared across all models.            
            ...
        ...
    ...
```

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)

## Toolkit Architecture
Our toolkit can be divided into 4 parts: **user interaction part** (green), **data loading part** (blue), **model calculation part** (red), **parameter optimization part** (brown).

### User Interaction Part
The different architectures of this part in train-valid and test branches are shown in the figures below.  
**Train-valid branch:**
![image](user_interaction_arch_train.png)
**Test branch:**
![image](user_interaction_arch_test.png)

This part interacts with the user and disk. There are three members in this part: _Runner_, _Monitor_, and _Snapshooter_:
* **Runner** is the entry of this toolkit where the experiment pipeline starts. 
It receives all the necessary experiment configurations from the user and distributes these configurations to the other parts of this toolkit like a hub. 
In each step, the runner receives the model calculation results from the model calculation part and passes the results to the monitors for later use.

* **Monitor** is the converter of the model calculation results received from the runner. 
The raw results are processed into human-readable logs and disk-savable data files. 
_ValidMonitor_ also encapsulates the logic of monitoring the training process, such as best model recording, early-stopping checking, and so on. 
Also, a monitor keeps the connection with a _Snapshooter_ and constantly packages snapshotting materials for it to capture the intermediate information.

* **Snapshooter** possesses a queue and constantly receives snapshotting materials by dequeuing the queue elements. 
For the program efficiency, it doesn't run in the main process of model training but a new process. 
It communicates with _Monitor_ by _multiprocessing.Queue_ and _multiprocessing.Event_.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)

### [Data Loading Part](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#data-loading-part)
The architecture of the data loading part of this toolkit is shown in the figure below.
![image](data_loading_arch.png)
This part plays the role of processing the raw data on the disk and providing the model with trainable batches. 
There are two members in this part: _Iterator_, _Dataset_:
* **Iterator** is the hub of this part. 
It holds a built-in _Dataset_ and batch views that decide its visible samples. 
In each epoch, it produces a _Dataloader_ that provides the built-in _Dataset_ with the sample indices to be extracted from the disk. 
After receiving the preprocessed vectors of all the data samples, the _Dataloader_ packages them into a trainable batch.

* **Dataset** is the frontend that can access the data files on the disk. 
It stores the necessary information of the data samples of the specified dataset (e.g. physical addresses for the waveform files, strings for the text files). 
After receiving the sample index from the _Dataloader_, it loads the chosen data sample from the disk and preprocesses it into a machine-friendly vector. 

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)

### [Model Calculation Part](https://github.com/ahclab/SpeeChain/tree/main/speechain/model#model-calculation-part)
The architecture of the model calculation part of this toolkit is shown in the figure below.
![image](model_calculation_arch.png)
This part receives the batch from the data loading part and outputs the training losses and evaluation metrics. 
There are three members in this part: _Model_, _Module_, _Criterion_:
* **Model** is the wrapper of all the model-related operations. 
It encapsulates the general model services such as model initialization, pretrained parameter loading, parameter freezing, and so on. 
It holds several built-in _Module_ members and several built-in _Criterion_ members.
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
The architecture of the parameter optimization part of this toolkit is shown in the figure below.
![image](parameter_optimization_arch.png)

This part does the job of updating the model parameters by the received losses. 
Unlike the traditional scheme of two separate objects (optimizer and scheduler), 
parameter optimization and learning rate scheduling are simultaneously done by _OptimScheduler_ in this toolkit.

* **OptimScheduler** is the hub of this part which encapsulates the logic of scheduling learning rates. 
It holds a built-in `torch.optim.Optimizer` member and provides the learning rate for the optimizer member in each training step. 
Users can override its interfaces to customize their personal strategies of scheduling the learning rates during training.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)

## Support Models
### [Speech Recognition](https://github.com/ahclab/SpeeChain/blob/main/speechain/model/asr.py)
* Speech-transformer ([**paper**](https://ieeexplore.ieee.org/abstract/document/8462506/))
* Conformer ([**paper**](https://arxiv.org/pdf/2005.08100))
### [Speech Synthesis]()
* Transformer-TTS ([**paper**](https://ojs.aaai.org/index.php/AAAI/article/view/4642/4520))
* FastSpeech2 ([**paper**](https://arxiv.org/pdf/2006.04558.pdf))
### [Speaker Recognition]()
* DeepSpeaker ([**paper**](https://arxiv.org/pdf/1705.02304.pdf))
* X-vector ([**paper**](https://danielpovey.com/files/2018_icassp_xvectors.pdf))

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)

## Available Datasets
### [Speech-Text](https://github.com/ahclab/SpeeChain/tree/main/datasets/speech_text#speech-text-datasets)
* LJSpeech ([**folder**](https://github.com/ahclab/SpeeChain/tree/main/datasets/speech_text/ljspeech), [**paper**](https://keithito.com/LJ-Speech-Dataset/))
* LibriSpeech ([**folder**](https://github.com/ahclab/SpeeChain/tree/main/datasets/speech_text/librispeech), [**paper**](http://ecargo.mn/storage/app/uploads/public/5b6/fde/8ca/5b6fde8ca80b0041081405.pdf))
* LibriTTS ([**paper**](https://arxiv.org/pdf/1904.02882))
* Libri-Light ([**paper**](https://arxiv.org/pdf/1912.07875))
* Libri-Adapt ([**paper**](https://arxiv.org/pdf/2009.02814))
* GigaSpeech ([**paper**](https://arxiv.org/pdf/2106.06909))
### [Speech-Speaker]()
* VoxCeleb ([**paper**](https://arxiv.org/pdf/1706.08612))
* VoxCeleb2 ([**paper**](https://arxiv.org/pdf/1806.05622))

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)

## Get a quick start
### Installation
We recommend you first install *Anaconda* into your machine before using our toolkit. 
After the installation of *Anaconda*, please follow the steps below to deploy our toolkit on your machine:
1. Clone this toolkit to your machine.
2. Move to the root path of this toolkit.
3. Run the command `conda env create -f environment.yaml` in the terminal 
(don't forget to change the _prefix_ inside this .yaml file to the place in your machine). 
Then, a virtual conda environment named *speechain* will be created for you.
4. Activate the environment *speechain* by the command `conda activate speechain`
5. Run the command `pip install -e .` to install our toolkit into the environment.

### Usage
The usage of this toolkit has 3 steps:
1. **Dataset Preparation**:
    1. Read [_./datasets/README.md_](https://github.com/ahclab/SpeeChain/tree/main/datasets#datasets-folder-of-the-speechain-toolkit) and familiarize yourself with how to dump the datasets.
    2. Move to the sub-folder of your target dataset in *./datasets/*.
    3. Run the processing scripts to dump the dataset on your machine by the command `./run.sh` in the sub-folder.
2. **Configuration Preparation**:
    1. Move to the sub-folder of your target dataset and task in *./recipe/*.
    2. Pick up an experiment configuration file you would like to run the experiment in _exp_cfg_.
    2. Replace all the absolute paths in the _exp_cfg_ file to the places on your machine.
    3. Don't forget to also change the paths in the _data_cfg_ file used in your selected _exp_cfg_ file. 
3. **Model Training**:
    1. Move to the root path of this toolkit.
    2. Activate the environment *speechain* by the command `conda activate speechain`
    3. Run the command `python speechain/runner.py --help` to see the meaning of each argument.
    4. Run the command `python speechain/runner.py --config 'the absolute path of your selected exp_cfg file'`

### Customization
Customization can be easily done by overriding the abstract interfaces of each base class. 
Please refer to the *README.md* in each sub-folder and the docstrings of *abs.py* in each sub-folder for more details.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)

## Contribution
### Current Developers
1. **[Heli Qi](https://scholar.google.com/citations?user=CH-rTXsAAAAJ&hl=zh-CN&authuser=1)**:
A master's student in the [AHC lab](https://ahcweb01.naist.jp/en/) at NAIST. Email: _qi.heli.qi9@is.naist.jp_  
(The names of contributors below are in alphabetical order. The order may change later depending on the contribution.)
2. Hao Wang
3. Sashi Novitasari
4. Yuta Nishikawa

### Development Documentation
There are 4 kinds of documents in this toolkit:
1. **Subfolder README.md**: There will be a README.md in each subfolder that explains the interfaces provided by the abstract base class and the corresponding configuration format.
2. **Function docstring**: There will be a docstring for each function that explains the details of input arguments and output results. We follow the Google Docstring format.
3. **In-line comments**: There will be some in-line comments in our codes to explain the role of each code block.
4. **Repository issues**: The future development plans will be given in the form of issues in this toolkit. 
Also, the issue part reports the problems when we are using and developing this toolkit.

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

## Common Q&A
