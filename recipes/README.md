# Recipes Folder of the SpeeChain toolkit
The recipes of the SpeeChain toolkit are grouped by the task. 
Each task has a sub-folder in `/speechain/recipes/`. 
In the sub-folder of each task, each dataset has a second-level sub-folder.

👆[Back to the handbook page](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#speechain-handbook)

## Table of Contents
1. [**Available Models & Datasets**](https://github.com/ahclab/SpeeChain/tree/main/recipes#available-models--datasets)
   1. [ASR (Automatic Speech Recognition)](https://github.com/ahclab/SpeeChain/tree/main/recipes#automatic-speech-recognition-asr)
   2. [TTS (Text-To-Speech Synthesis)](https://github.com/ahclab/SpeeChain/tree/main/recipes#text-to-speech-synthesis-tts)
   3. [SPKREC (Speaker Recognition)](https://github.com/ahclab/SpeeChain/tree/main/recipes#speaker-recognition-spkrec)
   4. [Offline TTS-to-ASR Chain](https://github.com/ahclab/SpeeChain/tree/main/recipes#offline-tts-to-asr-chain)
   4. [Offline ASR-to-TTS Chain](https://github.com/ahclab/SpeeChain/tree/main/recipes#offline-asr-to-tts-chain)
2. [**Experimental File System**](https://github.com/ahclab/SpeeChain/tree/main/recipes#experimental-file-system)


## Available Models & Datasets
We provide a specific README.md for each task in their folder. 
Please press the hyperlinks below and jump to the README.md of your target task for more details.
### [Automatic Speech Recognition (ASR)](https://github.com/ahclab/SpeeChain/tree/main/recipes/asr#automatic-speech-recognition-asr)
```
/asr
    /librispeech            # ASR Recipes for the LibriSpeech dataset
        /train-clean-100        # Labeled data: train-clean-100
            /data_cfg               # Data loading configuration files
            /exp_cfg                # Experimental environment configuration files
            /train_cfg              # Model construction and optimization configuration files
        /train-clean-460        # Labeled data: train-clean-460 (train-clean-100 + train-clean-360)
            /data_cfg
            /exp_cfg
            /train_cfg
        /train-960              # Labeled data: train_960 (train-clean-460 + train-other-500)
            /data_cfg
            /exp_cfg
            /train_cfg
```
<table>
	<tr>
	    <th>Dataset (Test Sets)</th>
	    <th>ASR Model</th>
	    <th>Setting</th>  
	    <th>WER w/o. LM</th>  
	    <th>WER w. Transformer LM</th>  
	</tr>
	<tr>
	    <td rowspan="3">LibriSpeech (test-clean / test-other)</td>
	    <td rowspan="3">Speech-Transformer</td>
	    <td>train-clean-100</td>
	    <td>12.10% / 29.10%</td>
	    <td></td>
	</tr>
	<tr>
	    <td>train-clean-460</td>
	    <td>5.73% / 16.63%</td>
        <td></td>
	</tr>
	<tr>
	    <td>train-960</td>
	    <td>4.45% / 10.46%</td>
        <td></td>
	</tr>
</table>

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/recipes#table-of-contents)


### [Text-To-Speech Synthesis (TTS)](https://github.com/ahclab/SpeeChain/tree/main/recipes/tts#text-to-speech-synthesis-tts)
```
/tts
    /libritts               # TTS Recipes for the LibriTTS dataset
        /train-clean-100        # Labeled data: train-clean-100
            /data_cfg               # Data loading configuration files that are shared by different models
            /exp_cfg                # Experimental environment configuration files
            /train_cfg              # Model construction and optimization configuration files
        /train-clean-460        # Labeled data: train-clean-460 (train-clean-100 + train-clean-360)
            /data_cfg
            /exp_cfg
            /train_cfg
        /train-960              # Labeled data: train_960 (train-clean-460 + train-other-500)
            /data_cfg
            /exp_cfg
            /train_cfg
    /ljspeech               # TTS Recipes for the LJSpeech dataset, LJSpeech doesn't have the official subset division.
        /data_cfg
        /exp_cfg
        /train_cfg
```
<table>
	<tr>
	    <th>Dataset (Test Sets)</th>
	    <th>TTS Model</th>
	    <th>SPK Model</th>
	    <th>Setting</th>  
	    <th>MCD</th>  
	</tr>
	<tr>
	    <td rowspan="3">LibriTTS (test-clean / test-other)</td>
	    <td rowspan="3">Transformer-TTS</td>
	    <td rowspan="3">One-hot Embedding</td>
	    <td>train-clean-100</td>
	    <td></td>
	</tr>
	<tr>
	    <td>train-clean-460</td>
	    <td></td>
	</tr>
	<tr>
	    <td>train-960</td>
	    <td></td>
	</tr>
    <tr>
	    <td>LJSpeech (test)</td>
	    <td>Transformer-TTS</td>
	    <td>N/A</td>
	    <td>train</td>
	    <td></td>
	</tr>
</table>

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/recipes#table-of-contents)

### [Speaker Recognition (SPKREC)](https://github.com/ahclab/SpeeChain/tree/main/recipes/spkrec#speaker-recognition-spkrec)

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/recipes#table-of-contents)

### [Offline TTS-to-ASR Chain](https://github.com/ahclab/SpeeChain/tree/main/recipes/offline_tts2asr#offline-tts-to-asr-chain)
```
/offline_tts2asr
    /libritts_librispeech   # TTS mode use LibriTTS and ASR model use LibriSpeech
        /train-clean-100-360    # Labeled data: train-clean-100; Unlabeled data: train-clean-360
            /data_cfg               # Data loading configuration files that are shared by different models
            /exp_cfg                # Experimental environment configuration files
            /train_cfg              # Model construction and optimization configuration files
        /train-100-860          # Labeled data: train-clean-100; Unlabeled data: train-clean-360 + train-other-500
            /data_cfg
            /exp_cfg
            /train_cfg
        /train-460-500          # Labeled data: train-clean-460; Unlabeled data: train-other-500
            /data_cfg
            /exp_cfg
            /train_cfg
```
<table>
	<tr>
	    <th>Dataset (Test Sets)</th>
	    <th>TTS Model</th>
	    <th>SPK Model</th>
	    <th>ASR Model</th>
	    <th>Setting</th>  
	    <th>WER w/o. LM</th>  
	</tr>
	<tr>
	    <td rowspan="3">LibriTTS-LibriSpeech (test-clean / test-other)</td>
	    <td rowspan="3">Transformer-TTS</td>
	    <td rowspan="3">One-hot Embedding</td>
	    <td rowspan="3">Speech-Transformer</td>
	    <td>train-clean-100-360</td>
	    <td></td>
	</tr>
	<tr>
	    <td>train-460-500</td>
	    <td></td>
	</tr>
	<tr>
	    <td>train-100-860</td>
	    <td></td>
	</tr>
</table>

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/recipes#table-of-contents)

### [Offline ASR-to-TTS Chain](https://github.com/ahclab/SpeeChain/tree/main/recipes/offline_asr2tts#offline-asr-to-tts-chain)
```
/offline_asr2tts
    /libritts   # ASR & TTS models are trained on LibriTTS
        /train-clean-100-360    # Labeled data: train-clean-100; Unlabeled data: train-clean-360
            /data_cfg               # Data loading configuration files that are shared by different models
            /exp_cfg                # Experimental environment configuration files
            /train_cfg              # Model construction and optimization configuration files
        /train-100-860          # Labeled data: train-clean-100; Unlabeled data: train-clean-360 + train-other-500
            /data_cfg
            /exp_cfg
            /train_cfg
        /train-460-500          # Labeled data: train-clean-460; Unlabeled data: train-other-500
            /data_cfg
            /exp_cfg
            /train_cfg
```
<table>
	<tr>
	    <th>Dataset (Test Sets)</th>
	    <th>ASR Model</th>
	    <th>TTS Model</th>
	    <th>SPK Model</th>
	    <th>Setting</th>  
	    <th>MCD</th>  
	</tr>
	<tr>
	    <td rowspan="3">LibriTTS (test-clean / test-other)</td>
	    <td rowspan="3">Speech-Transformer</td>
	    <td rowspan="3">Transformer-TTS</td>
	    <td rowspan="3">One-hot Embedding</td>
	    <td>train-clean-100-360</td>
	    <td></td>
	</tr>
	<tr>
	    <td>train-460-500</td>
	    <td></td>
	</tr>
	<tr>
	    <td>train-100-860s</td>
	    <td></td>
	</tr>
</table>

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/recipes#table-of-contents)

## Experimental File System
The file system of an experiment folder is shown as below:
```
/data_cfg
/train_cfg
/exp_cfg
/exp                # the 'exp' folder of each model
    /{exp_name}         # the name of a specific experiment
        /{test_cfg_name}    # the name of a testing configuration file
            /{test_model_name}  # the name of the model you want to test the performance
                /{test_set_name}    # the name of a test set
                    /figures                # the folder that contains all the distribution figures of each metric on the test set
                        {test_metric_name}.png  # the histogram distribution figure of the {test_metric_name} values of all the testing instances
                        ...
                    test.log                # the log file that contains the testing process of a specific test set
                    overall_results.md      # the .md file that contains the model overall performance on the test set
                    instance_reports.md     # the .md file that contains the detailed performance reports of each testing instance
                    topn_(max/min)_{xxx}.md # the .md file that contains the top-n bad cases selected by the metric 'xxx'
                    idx2{xxx}               # the suffix-free .txt files that contain individual metrics of each testing instance, a file corresponds to a metric 'xxx'
                ...                 # other test sets
            ...                 # other test models
        ...                 # other test configurations
        
        models/             # this sub-folder contains all the model files
            N_{xxx}_average.pth # the average model obtained by the metric 'xxx' on 'N' best models
            {xxx}_best.pth      # the file soft link to the best model obtained by the metric 'xxx'
            {xxx}_best_2.pth    # the file soft link to the second best model obtained by the metric 'xxx'
            ...                
            {xxx}_best_n.pth    # the file soft link to the n-th best model obtained by the metric 'xxx'  
            epoch_{X}.pth       # the actual model file of the X-th epoch
            epoch_{Y}.pth       # the actual model file of the Y-th epoch
            ...          

        figures/            # this sub-folder contains all the snapshotting figures. the .txt files in this folder contain the data used to plot summary.png which can be used to plot your own figures.
            train/              # the snapshotting figures made during training
                consumed_memory/    # the folder containing the GPU memory consumption records
                     Rank0.txt                           # the numerical records of the memory consumption for the GPU rank.0 through epochs
                     ...                                 # other GPUs if have
                     RankN.txt                           # the numerical records of the memory consumption for the GPU rank.N through epochs
                     summary.png                         # the curve graph of all the records above through epochs
                consumed_time/      # the folder containing the time consumption records
                     data_load_time.txt                  # the numerical records of the data loading time through epochs
                     model_forward_time.txt              # the numerical records of the model forward time through epochs
                     loss_backward_time_{optim_name}.txt # the numerical records of the loss backward time for the optimizer named {optim_name} through epochs
                     optim_time_{optim_name}.txt         # the numerical records of the parameter optimization time for the optimizer named {optim_name} through epochs
                     summary.png                         # the curve graph of all the records above through epochs
                criteria/           # the folder containing the training criterion value records
                     {train_criterion_name}              # the numerical records of the training criterion named {train_criterion_name} through epochs
                     ...                                 # other training criteria if have
                     summary.png                         # the curve graph of all the records above through epochs
                optim_lr/           # the folder containing the learning rate records of each optimizer
                     {optim_name}.txt                    # the numerical recores of the learning rates of the optimizer named {optim_name} through epochs
                     ...                                 # other optimizers if have
                     summary.png                         # the curve graph of all the records above through epochs
            valid/              # the snapshotting figures made during validation
                consumed_memory/    
                     Rank0.txt                           
                     ...
                     RankN.txt                           
                     summary.png                         
                consumed_time/      
                     data_load_time.txt                  
                     model_forward_time.txt              
                     summary.png                         
                criteria/           # the folder containing the validation criterion value records
                     {valid_criterion_name}              # the numerical records of the validation criterion named {valid_criterion_name} through epochs
                     ...                                 # other validation criteria if have
                     summary.png                         # the curve graph of all the records above through epochs

        tensorboard/        # this sub-folder contains the writer events for tensorboard visualization 
        checkpoint.pth      # the checkpoint of the training process so far, used for resuming the training process
        train_data_cfg.yaml # the data loading configuration for the training-validation part of the experiment, used for resuming the training process
        test_data_cfg.yaml  # the data loading configuration for the testing part of the experiment, used for resuming the testing process
        exp_cfg.yaml        # the experiment environment configuration for the experiment, used for resuming the training process
        train_cfg.yaml      # the moder and optimizer configuration for the experiment, used for resuming the training process
        train.log           # the log file that contains the training process of the given training sets and validation sets
```

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/recipes#table-of-contents)

