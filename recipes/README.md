# Recipes Folder of the SpeeChain toolkit
The recipes of the SpeeChain toolkit are grouped by the task. 
Each task has a sub-folder in */recipes/*. 
In the sub-folder of each task, each dataset has a second-level sub-folder.

👆[Back to the home page](https://github.com/ahclab/SpeeChain#the-speechain-toolkit)

## Available Recipes
### [Speech Recognition](https://github.com/ahclab/SpeeChain/tree/main/recipes/asr)
1. [LibriSpeech](https://github.com/ahclab/SpeeChain/tree/main/recipes/asr/librispeech)
2. [LJSpeech](https://github.com/ahclab/SpeeChain/tree/main/recipes/asr/ljspeech)

### [Speech Synthesis](https://github.com/ahclab/SpeeChain/tree/main/recipes/tts)
1. [LJSpeech]
1.1. [Core(txt2feat)](https://github.com/ahclab/SpeeChain/tree/main/recipes/tts/ljspeech/transformer)
1.2. [Vocoder(feat2wav)](https://github.com/ahclab/SpeeChain/tree/main/recipes/tts/ljspeech/vocoder)

2. [WSJ]
(pre-requisite: speaker recognition)
1.1. [Core(txt2feat)](https://github.com/ahclab/SpeeChain/tree/main/recipes/tts/wsj/transformer)
1.2. [Vocoder(feat2wav)](https://github.com/ahclab/SpeeChain/tree/main/recipes/tts/wsj/vocoder)

### [Speaker Recognition](https://github.com/ahclab/SpeeChain/tree/main/recipes/spkrec)
1. [WSJ](https://github.com/ahclab/SpeeChain/tree/main/recipes/spkrec/wsj)

## Experiment Results Structure
The structure of the result files in an experiment folder is shown as below:
```
exp/                # the 'exp' folder of each model
    exp_name/           # the name of a specific experiment
        test_cfg_name/      # the name of a testing configuration file
            test_model_name/    # the name of the model you want to test the performance
                test_set_name/      # the name of a test set
                    test.log            # the log file that contains the testing process of a specific test set
                    result              # the file that contains all the average metrics on all testing samples
                    ...                 # the files that contains individual metrics of each testing sample, a file corresponds to a metric
                ...                 # other test sets
            ...                 # other test models
        ...                 # other test configurations
        
        models/             # this sub-folder contains all the model files
            N_xxx_average.mdl   # the average model obtained by the metric 'xxx' on 'N' best models
            xxx_best.mdl        # the best model obtained by the metric 'xxx'
            xxx_best_2.mdl      # the second best model obtained by the metric 'xxx'
            ...                
            xxx_best_n.mdl      # the n-th best model obtained by the metric 'xxx'            

        figures/            # this sub-folder contains all the snapshotting figures made
            train/              # the snapshotting figures made during training
                ...
            valid/              # the snapshotting figures made during validation
                ...
        tensorboard/        # this sub-folder contains the writer events for tensorboard visualization 
        
        checkpoint.pth      # the checkpoint of the training process so far, used for resuming the training process
        data_cfg.yaml       # the data loading configuration for the experiment, used for resuming the training process
        exp_cfg.yaml        # the experiment environment configuration for the experiment, used for resuming the training process
        train_cfg.yaml      # the moder and optimizer configuration for the experiment, used for resuming the training process
        train.log           # the log file that contains the training process of the given training sets and validation sets
```