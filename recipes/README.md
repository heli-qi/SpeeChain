# Recipes Directory of the SpeeChain toolkit
The SpeeChain toolkit organizes its recipes by task, each located within a dedicated sub-folder in /speechain/recipes/. 
Every task sub-folder hosts second-level sub-folders that pertain to individual datasets.

👆[Back to the handbook page](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#speechain-handbook)

## Table of Contents
1. [**Available Datasets**](https://github.com/ahclab/SpeeChain/tree/main/recipes#available-datasets)
   1. [ASR (Automatic Speech Recognition)](https://github.com/ahclab/SpeeChain/tree/main/recipes#automatic-speech-recognition-asr)
   2. [TTS (Text-To-Speech Synthesis)](https://github.com/ahclab/SpeeChain/tree/main/recipes#text-to-speech-synthesis-tts)
   3. [Offline TTS-to-ASR Chain](https://github.com/ahclab/SpeeChain/tree/main/recipes#offline-tts-to-asr-chain)
2. [**Experimental File System**](https://github.com/ahclab/SpeeChain/tree/main/recipes#experimental-file-system)


## Available Datasets
Each task comes with a dedicated README.md file found within its respective folder, providing further details and instructions. 
Follow the hyperlinks below to navigate to the README.md file for your target task.
### [Automatic Speech Recognition (ASR)](https://github.com/ahclab/SpeeChain/tree/main/recipes/asr#automatic-speech-recognition-asr)
Refer to this sample structure for an overview of how the ASR folder is organized:
```
/asr
    /librispeech            # ASR Recipes for the LibriSpeech dataset
        /train-clean-100        # Labeled data: train-clean-100
            /data_cfg               # Data loading configuration files
            /exp_cfg                # Experimental configuration files
        /train-clean-460        # Labeled data: train-clean-460 (train-clean-100 + train-clean-360)
            ...
        /train-960              # Labeled data: train-960 (train-clean-460 + train-other-500)
            ...
    /libritts_librispeech   # ASR Recipes for the joint dataset of LibriSpeech and 16khz-downsampled LibriTTS
        /train-960              # Labeled data: LibriSpeech_train-960 & 16khz-LibriTTS_train-960
            ...
```

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/recipes#table-of-contents)


### [Text-To-Speech Synthesis (TTS)](https://github.com/ahclab/SpeeChain/tree/main/recipes/tts#text-to-speech-synthesis-tts)
Refer to this sample structure for an overview of how the TTS folder is organized:
```
/tts
    /libritts               # TTS Recipes for the LibriTTS dataset
        /train-clean-100        # Labeled data: train-clean-100
            /data_cfg               # Data loading configuration files that are shared by different models
            /exp_cfg                # Experimental configuration files
        /train-clean-460        # Labeled data: train-clean-460 (train-clean-100 + train-clean-360)
            ...
        /train-960              # Labeled data: train-960 (train-clean-460 + train-other-500)
            ...
    /ljspeech               # TTS Recipes for the LJSpeech dataset, LJSpeech doesn't have the official subset division.
        ...
    /vctk                   # TTS Recipes for the VCTK dataset, VCTK doesn't have the official subset division.
```

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/recipes#table-of-contents)

### [Offline TTS-to-ASR Chain](https://github.com/ahclab/SpeeChain/tree/main/recipes/offline_tts2asr#offline-tts-to-asr-chain)
Refer to this sample structure for an overview of how the offline TTS-to-ASR Chain folder is organized:
```
/offline_tts2asr
    /librispeech   # LibriSpeech is used as labeled data to train ASR models
        /train-clean-100    # ASR Labeled data: LibriSpeech_train-clean-100
            /data_cfg               # Data loading configuration files
            /exp_cfg                # Experimental configuration files
        /train-960          # ASR Labeled data: LibriSpeech_train-960
            ...
```

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/recipes#table-of-contents)

## Experimental File System
The file structure of an experiment folder is as follows:
```
/data_cfg
/train_cfg
/exp_cfg
/exp                # 'exp' folder of each model
    /{exp_name}         # Name of a specific experiment
        /{test_cfg_name}    # Name of a testing configuration file
            /{test_model_name}  # Name of the model you want to test the performance
                /{test_set_name}    # Name of a test set
                    /figures                # Folder that contains all the distribution figures of each metric on the test set
                        {test_metric_name}.png  # Histogram distribution figure of the {test_metric_name} values of all the testing instances
                        ...
                    test.log                # Log file that contains the testing process of a specific test set
                    overall_results.md      # .md file that contains the model overall performance on the test set
                    instance_reports.md     # .md file that contains the detailed performance reports of each testing instance
                    topn_(max/min)_{xxx}.md # .md file that contains the top-n bad cases selected by the metric 'xxx'
                    idx2{xxx}               # suffix-free .txt files that contain individual metrics of each testing instance, a file corresponds to a metric 'xxx'
                ...                 # other test sets
            ...                 # other test models
        ...                 # other test configurations
        
        models/             # This sub-folder contains all the model files
            N_{xxx}_average.pth # Average model obtained by the metric 'xxx' on 'N' best models
            {xxx}_best.pth      # File soft link to the best model obtained by the metric 'xxx'
            {xxx}_best_2.pth    # File soft link to the second best model obtained by the metric 'xxx'
            ...                
            {xxx}_best_n.pth    # File soft link to the n-th best model obtained by the metric 'xxx'  
            epoch_{X}.pth       # Actual model file of the X-th epoch
            epoch_{Y}.pth       # Actual model file of the Y-th epoch
            ...          

        figures/            # This sub-folder contains all the snapshotting figures. The .txt files in this folder contain the data used to plot summary.png which can be used to plot your own figures.
            train/              # Snapshotting figures made during training
                consumed_memory/    # Folder containing the GPU memory consumption records
                     Rank0.txt                           # Numerical records of the memory consumption for the GPU rank.0 through epochs
                     ...                                 # Other GPUs if have
                     RankN.txt                           # Numerical records of the memory consumption for the GPU rank.N through epochs
                     summary.png                         # Curve graph of all the records above through epochs
                consumed_time/      # Folder containing the time consumption records
                     data_load_time.txt                  # Numerical records of the data loading time through epochs
                     model_forward_time.txt              # Numerical records of the model forward time through epochs
                     loss_backward_time_{optim_name}.txt # Numerical records of the loss backward time for the optimizer named {optim_name} through epochs
                     optim_time_{optim_name}.txt         # Numerical records of the parameter optimization time for the optimizer named {optim_name} through epochs
                     summary.png                         # Curve graph of all the records above through epochs
                criteria/           # Folder containing the training criterion value records
                     {train_criterion_name}              # Numerical records of the training criterion named {train_criterion_name} through epochs
                     ...                                 # Other training criteria if have
                     summary.png                         # Curve graph of all the records above through epochs
                optim_lr/           # Folder containing the learning rate records of each optimizer
                     {optim_name}.txt                    # Numerical recores of the learning rates of the optimizer named {optim_name} through epochs
                     ...                                 # Other optimizers if have
                     summary.png                         # Curve graph of all the records above through epochs
            valid/              # Snapshotting figures made during validation
                consumed_memory/    
                     Rank0.txt                           
                     ...
                     RankN.txt                           
                     summary.png                         
                consumed_time/      
                     data_load_time.txt                  
                     model_forward_time.txt              
                     summary.png                         
                criteria/           # Folder containing the validation criterion value records
                     {valid_criterion_name}              # Numerical records of the validation criterion named {valid_criterion_name} through epochs
                     ...                                 # Other validation criteria if have
                     summary.png                         # Curve graph of all the records above through epochs

        tensorboard/        # This sub-folder contains the writer events for tensorboard visualization 
        checkpoint.pth      # Checkpoint of the training process so far, used for resuming the training process
        train_data_cfg.yaml # Data loading configuration for the training-validation part of the experiment, used for resuming the training process
        test_data_cfg.yaml  # Data loading configuration for the testing part of the experiment, used for resuming the testing process
        exp_cfg.yaml        # Experiment environment configuration for the experiment, used for resuming the training process
        train_cfg.yaml      # Model and optimizer configuration for the experiment, used for resuming the training process
        train.log           # Log file that contains the training process of the given training sets and validation sets
```

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/recipes#table-of-contents)

