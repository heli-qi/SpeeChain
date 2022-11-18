# SpeeChain: A PyTorch-based Speech&Language Processing Toolkit for the Machine Speech Chain
_SpeeChain_ is an open-source PyTorch-based speech and language processing toolkit produced by the [_AHC lab_](https://ahcweb01.naist.jp/en/) at Nara Institute of Science and Technology (NAIST). 
This toolkit is designed for simplifying the pipeline of the research on the machine speech chain, 
i.e. the joint model of automatic speech recognition (ASR) and text-to-speech synthesis (TTS). 

_SpeeChain is currently in beta._ Contribution to this toolkit is warmly welcomed anywhere anytime! 

If you find our toolkit helpful for your research, we sincerely hope that you can give us a star⭐! 
Anytime when you encounter problems when using our toolkit, please don't hesitate to leave us an issue!

## Table of Contents
1. [**What is Machine Speech Chain?**](https://github.com/ahclab/SpeeChain#toolkit-overview)
2. [**Toolkit Overview**](https://github.com/ahclab/SpeeChain#toolkit-overview)
3. Show-off 
4. [**Get a Quick Start**](https://github.com/ahclab/SpeeChain#get-a-quick-start)
5. [**Documentation**](https://github.com/ahclab/SpeeChain#documentation)
6. [**Contribution**](https://github.com/ahclab/SpeeChain#contribution)


## What is Machine Speech Chain?

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)


## Toolkit Overview
### On-the-fly Data Processing
* **On-the-fly Feature Extraction:**
  * Acoustic features:
      * Linear Spectrogram
      * Log-Mel Spectrogram
      * MFCC (_under development_)
  * Self-supervised representations:
      * Wav2Vec2 (_under development_)
      * HuBERT (_under development_)
* **On-the-fly Data Augmentation:**
  * Time-domain speech augmentations:
      * Speed perturbation (_under development_)
      * Environmental noise addition (_under development_)
      * Reverberation addition (_under development_)
  * Frequency-domain speech augmentations:
      * SpecAugment
* **On-the-fly Data Pre-processing:**
  * Speech downsampling (_under development_)
  * Text normalization (_under development_)
    * Letter case handling
    * Punctuation Restoration
  * Per-channel feature normalization

### **Efficient and Transparent Training:**
* Multi-GPU model distribution based on DDP (Distributed Data Parallel) for both training and inference.
* Real-time status reporting by online _Tensorboard_ and offline _Matplotlib_.
* Real-time learning dynamics visualization by interpretability tools. Currently available interpretability tools:
  * Attention visualization
  * Integrated gradients (_under development_) 

### **Detailed and Informative Model Evaluation:**
* Multi-level _.md_ evaluation reports (overall-level, group-level model, and sample-level) without any layout misplacement. 
* Statistical distribution visualization of evaluation metrics
* TopN bad case analysis for better model diagnosis and adjustment.
* 
### **Easy-to-use Interfaces and Configuration:**
* Customizable pipeline with abundant interface functions and detailed documents. 
* Reusable and user-friendly configuration files that support dynamic class importing, dynamic datatype specification, and dynamic value reference. 

### **Flexible Data Loading:**
* Free combinations of training, validation, and testing sets from different corpora.
* On-the-fly dataset mixture for a single dataloader to uniformly fetch data samples from multiple corpora.
* On-the-fly data selection that changes the accessible data indices of a dataloader to filter the undesired data samples.
* Multi-dataloader batch generation to form training batches with static data composition. 
Different dataloaders can have different source datasets, augmentation functions, and data fetching strategies.

### **Off-the-shelf Pseudo Data Calibrations for Semi-supervised Learning:**
* ASR hypothesis transcript calibrations:
    * Joint decoding by CTC & language model (_under development_)
    * Hypothesis rescoring by CTC & language model (_under development_)
    * lexicon-restricted decoding (_under development_)
    * Controllable beam searching:
        * Hypothesis length penalty
        * Early-stopping hypothesis prevention
        * Softmax temperature adjustment
    * Criterion filtering:
        * Hypothesis confidence
        * Dropout-based uncertainty (_under development_)
    * Heuristic methods:
        * Repeating n-gram phrase removal (_under development_)
        * Lexicon calibration for out-of-vocabulary words (_under development_)
* TTS Synthesis utterance calibrations:
    * ASR evaluation
    * In-utterance silence removal (_under development_)
    * Repeating n-gram phrase removal (_under development_)
    * Non-speech sound removal (_under development_)

### **Convenient Model Optimization:**
* Model training can be done by multiple optimizers.
Each optimizer has a specific training loss, model parameters, learning rate scheduling strategy, and optimization interval. 
* Gradient accumulation for mimicking the large-batch gradients by the ones on several small batches.
* Easy-to-set finetuning factor to scale down the learning rates without any modification of the scheduler configuration. 

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)


## Get a Quick Start
We recommend you first install *Anaconda* into your machine before using our toolkit. 
After the installation of *Anaconda*, please follow the steps below to deploy our toolkit on your machine:
1. Find a path with enough disk memory space.
2. Clone our toolkit by `git clone https://github.com/ahclab/SpeeChain.git`.
3. Move to the root path of our toolkit by `cd SpeeChain`.
4. Run `conda env create -f environment.yaml` to create our virtual conda environment named *speechain*.
(don't forget to change the _prefix_ inside this .yaml file to the place in your machine). 
5. Activate the environment *speechain* by `conda activate speechain`
6. Run `pip install -e .` to install our toolkit into the environment.
7. Read the [handbook]() and start your journey in SpeeChain!

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)


## Documentation
There are 4 types of documents in our toolkit:
1. [**Overview Handbook**]():   
We provide a _handbook.md_ at the root path to present an overview of our toolkit to our users. 
This handbook will tell you the detailed learning path from a beginner to a contributor.
2. **Multi-level README.md**:  
Apart from the _README.md_ in the root path, we also provide a _README.md_ in each sub-folder where we think it's necessary to explain the usage such as interfaces, configuration formats, and so on.
We provide enough hyperlinks for you to easily jump between different README.md levels.
3. **Docstrings**:  
We provide detailed docstrings for the classes or functions that we think should be detailed-ly explained such as input arguments, returned results, and so on. 
For better readability, we follow the Google Docstring format.
4. **In-line comments**:  
Our codes are grouped by their functions to improve the readability. 
For each functional code block, we provide some in-line comments to explain the block such as its working flow and data shape transformation.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)


## Contribution
* Previous Contributors
  1. **[Sashi Novitasari](https://scholar.google.com/citations?user=nkkik34AAAAJ)**
* Current Developers
  1. **[Heli Qi](https://scholar.google.com/citations?user=CH-rTXsAAAAJ)** 

For those who are interested in cooperating with us, please feel free to email to _qi.heli.qi9@is.naist.jp_ while cc to _s-nakamura@is.naist.jp_.  
👆[Back to the table of contents](https://github.com/ahclab/SpeeChain#table-of-contents)
