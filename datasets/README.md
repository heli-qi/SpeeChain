# Datasets
This folder contains all the available datasets in this toolkit. 
Each dataset corresponds to a sub-folder and has a uniform file system. 
You can easily dump your target dataset to your machine by following the instructions below. 
If you want to contribute a new dataset, we would appreciate it if you could follow our file systems and metadata formats.

👆[Back to the handbook page](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#speechain-handbook)

## Table of Contents
1. [**File System**](https://github.com/ahclab/SpeeChain/tree/main/datasets#file-system)
2. [**Metadata Format**](https://github.com/ahclab/SpeeChain/tree/main/datasets#metadata-format)
    1. [idx2wav](https://github.com/ahclab/SpeeChain/tree/main/datasets#idx2wav)
    2. [idx2wav_len](https://github.com/ahclab/SpeeChain/tree/main/datasets#idx2wav_len)
    3. [idx2feat](https://github.com/ahclab/SpeeChain/tree/main/datasets#idx2feat)
    4. [idx2feat_len](https://github.com/ahclab/SpeeChain/tree/main/datasets#idx2feat_len)
    5. [idx2text](https://github.com/ahclab/SpeeChain/tree/main/datasets#idx2text)
    6. [idx2spk](https://github.com/ahclab/SpeeChain/tree/main/datasets#idx2spk)
    7. [idx2spk_feat](https://github.com/ahclab/SpeeChain/tree/main/datasets#idx2spk_feat)
    8. [spk_list](https://github.com/ahclab/SpeeChain/tree/main/datasets#spk_list)
    9. [idx2gen](https://github.com/ahclab/SpeeChain/tree/main/datasets#idx2gen)
3. [**How to Dump a Dataset on your Machine**](https://github.com/ahclab/SpeeChain/tree/main/datasets#how-to-dump-a-dataset-on-your-machine)
4. [**How to Contribute a New Dataset**](https://github.com/ahclab/SpeeChain/tree/main/datasets#how-to-contribute-a-new-dataset)

## File System
```
/datasets
    /pyscripts                  # fixed .py scripts provided by the toolkit
    data_dumping.sh             # the shared .sh script across all the speech-text datasets. It contains the complete pipeline of data dumping.
    meta_generator.py           # the abstract .py script used by each dataset to decide their own meta generation logic.
    meta_post_processor.py      # the abstract .py script used by each dataset to decide their own meta post-processing logic.
    /{dataset_name}             # root folder of each dataset
        /data                       # main folder of each dataset (the folder name 'data' is shared across all the datasets)
            /wav                        # waveform folder (the folder name 'wav' is shared across all the datasets)
                /{subset_name}              # the name of a subset in this dataset 
                    /wav_{comp_file_ext}        # (optional) the folder that contains the compressed package files of all the waveform data
                    idx2wav_{comp_file_ext}     # (optional) the file that contains the pairs of index and the absolute address of waveforms in the compressed package files
                    idx2wav                     # the file that contains the pairs of index and the absolute address of waveform files
                    idx2wav_len                 # the file that contains the pairs of index and waveform length
                    idx2{txt_format}_text       # the file that contains the pairs of index and transcript text. (txt_format is the format of processed text which is used to distinguish different idx2text files)
                    ...                         # (optional) some other metadata files available in the dataset (such as idx2spk, idx2spk_feat, idx2gen, ...)
            /wav{sample_rate}           # (optional) downsampled waveform folder (sample_rate is the samping rate of waveforms after downsampling)
                ...                         # same structure as '/wav'
            /{feat_config}              # (optional) acoustic feature folder by a given configuration (feat_config is the name of the configuration file)
                /{subset_name}              # the name of a subset in this dataset 
                    /feat_{comp_file_ext}       # (optional) the folder that contains the compressed chunk files of all the acoustic feature data
                    idx2feat_{comp_file_ext}    # (optional) the file that contains the pairs of index and the absolute address of waveforms in the compressed package files
                    idx2feat                    # the file that contains the pairs of index and the absolute address of acoustic feature files
                    idx2feat_len                # the file that contains the pairs of index and feature length
                    idx2{txt_format}_text       # the file that contains the pairs of index and transcript text. (txt_format is the format of processed text which is used to distinguish different idx2text files)
                    ...                         # (optional) some metadata files available in the dataset (such as idx2spk, idx2gen, ...)
            /{token_type}               # token folder of a specific type (token_type is the name of the specified token type)
                /{src_subset}               # the source subset used for generating the token vocabulary (src_subset is the name of the used source subset)
                    /{token_config}             # the configuration for generating the vocabulary list.
                        /{txt_format}               # the format of text data used to generate the token vocabulary
                            idx2text                    # the file that contains the pairs of index and transcript text after tokenization
                            idx2text_len                # the file that contains the pairs of index and transcript text length after tokenization
                            vocab                       # token vocabulary
                            model                       # (optional) tokenizer model. Used for sentencepiece tokenizers.
        run.sh                      # the dataset-specific .sh script that controls how the data dumping for the corresponding dataset is going
        data_download.sh            # the dataset-specific .sh script that downloads the corresponding dataset from the internet
        meta_generator.py           # the dataset-specific .py script that generates the meta data files of the dataset (idx2wav, idx2spk, ...)
        meta_post_processor.py      # (optional) the dataset-specific .py script that performs post-processing for the meta data files (needed by some datasets like LibriSpeech)
```
The names in the braces({}) mean the undefined names depending on the settings of datasets and configuration.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/datasets#table-of-contents)


## Metadata Format
Metadata files are suffix-free ._txt_ files used to access data instances during training. 
The data formats of metadata files are uniform for all datasets. 
The format uniformity enables the automatic configuration for the data loading part of our toolkit. 
The format for each metadata file is shown below.

#### idx2wav
In `idx2wav`, each line corresponds to a pair of file index and the absolute address of raw waveforms. 
Index and address are separated by a blank. 

For example, 
```
103-1240-0000 ${SPEECHAIN_ROOT}/datasets/librispeech/data/wav/train-clean-100/103/1240/103-1240-0000.flac
103-1240-0001 ${SPEECHAIN_ROOT}/datasets/librispeech/data/wav/train-clean-100/103/1240/103-1240-0001.flac
103-1240-0002 ${SPEECHAIN_ROOT}/datasets/librispeech/data/wav/train-clean-100/103/1240/103-1240-0002.flac
103-1240-0003 ${SPEECHAIN_ROOT}/datasets/librispeech/data/wav/train-clean-100/103/1240/103-1240-0003.flac
103-1240-0004 ${SPEECHAIN_ROOT}/datasets/librispeech/data/wav/train-clean-100/103/1240/103-1240-0004.flac
```
Any audio files that can be processed by `soundfile.read()` (such as .flac, .wav, ...) are OK in _idx2wav_.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/datasets#table-of-contents)


#### idx2wav_len
In `idx2wav_len`, each line corresponds to a pair of file index and file length which are separated by a blank. 
The file length means the number of sampling points of the waveform.

For example, 
```
103-1240-0000 225360
103-1240-0001 255120
103-1240-0002 223120
103-1240-0003 235360
103-1240-0004 200240
```

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/datasets#table-of-contents)


#### idx2feat
In `idx2feat`, each line corresponds to a pair of file index and absolute address of the acoustic feature. 
The index and absolute address are separated by a blank. 

For example, 
```
103-1240-0000 ${SPEECHAIN_ROOT}/datasets/librispeech/data/{feat_config}/train-clean-100/103-1240-0000.npz
103-1240-0001 ${SPEECHAIN_ROOT}/datasets/librispeech/data/{feat_config}/train-clean-100/103-1240-0001.npz
103-1240-0002 ${SPEECHAIN_ROOT}/datasets/librispeech/data/{feat_config}/train-clean-100/103-1240-0002.npz
103-1240-0003 ${SPEECHAIN_ROOT}/datasets/librispeech/data/{feat_config}/train-clean-100/103-1240-0003.npz
103-1240-0004 ${SPEECHAIN_ROOT}/datasets/librispeech/data/{feat_config}/train-clean-100/103-1240-0004.npz
```
feat_config is the name of the used feature extraction configuration file.
In our toolkit, acoustic feature of a waveform is saved as a .npy file by the NumPy package.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/datasets#table-of-contents)


#### idx2feat_len
In `idx2feat_len`, each line corresponds to a pair of file index and file length which are separated by a blank. 
The file length means the number of time frames in the extracted acoustic features (e.g. log-mel spectrogram or MFCC).

For example, 
```
103-1240-0000 1408
103-1240-0001 1594
103-1240-0002 1394
103-1240-0003 1471
103-1240-0004 1251
```

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/datasets#table-of-contents)


#### idx2text
In `idx2text`, each line corresponds to a pair of file index and transcript text string which are separated by a blank. 
`idx2text` will be renamed as `idx2{txt_format}_text` to indicate the text processing format used to generate this file.
In our toolkit, there are many available text processing format to generate transcript text strings with different styles.

For example, 
```
103-1240-0000 chapter one missus rachel lynde is surprised missus rachel lynde lived just where the avonlea main road dipped down into a little hollow fringed with alders and ladies eardrops and traversed by a brook
103-1240-0001 that had its source away back in the woods of the old cuthbert place it was reputed to be an intricate headlong brook in its earlier course through those woods with dark secrets of pool and cascade but by the time it reached lynde's hollow it was a quiet well conducted little stream
103-1240-0002 for not even a brook could run past missus rachel lynde's door without due regard for decency and decorum it probably was conscious that missus rachel was sitting at her window keeping a sharp eye on everything that passed from brooks and children up
103-1240-0003 and that if she noticed anything odd or out of place she would never rest until she had ferreted out the whys and wherefores thereof there are plenty of people in avonlea and out of it who can attend closely to their neighbor's business by dint of neglecting their own
103-1240-0004 but missus rachel lynde was one of those capable creatures who can manage their own concerns and those of other folks into the bargain she was a notable housewife her work was always done and well done she ran the sewing circle
```
**Note**: you don't need to worry about the blanks inside each transcript text string. 
Those additional blanks will be ignored in the subsequent processing.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/datasets#table-of-contents)


#### idx2spk
In `idx2spk`, each line corresponds to a pair of file index and speaker ID. 

For example, 
```
103-1240-0000 103
103-1240-0001 103
103-1240-0002 103
103-1240-0003 103
103-1240-0004 103
```

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/datasets#table-of-contents)


#### idx2spk_feat
In `idx2spk_feat`, each line corresponds to a pair of file index and the absolute address of a speaker embedding. 

For example, 
```
1034_121119_000001_000001 ${SPEECHAIN_ROOT}/datasets/libritts/data/wav/train-clean-100/xvector/1034_121119_000001_000001.npy
1034_121119_000002_000001 ${SPEECHAIN_ROOT}/datasets/libritts/data/wav/train-clean-100/xvector/1034_121119_000002_000001.npy
1034_121119_000010_000004 ${SPEECHAIN_ROOT}/datasets/libritts/data/wav/train-clean-100/xvector/1034_121119_000010_000004.npy
1034_121119_000010_000006 ${SPEECHAIN_ROOT}/datasets/libritts/data/wav/train-clean-100/xvector/1034_121119_000010_000006.npy
1034_121119_000012_000000 ${SPEECHAIN_ROOT}/datasets/libritts/data/wav/train-clean-100/xvector/1034_121119_000012_000000.npy
1034_121119_000014_000000 ${SPEECHAIN_ROOT}/datasets/libritts/data/wav/train-clean-100/xvector/1034_121119_000014_000000.npy
1034_121119_000018_000000 ${SPEECHAIN_ROOT}/datasets/libritts/data/wav/train-clean-100/xvector/1034_121119_000018_000000.npy
```

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/datasets#table-of-contents)


#### spk_list
In _spk_list_, each line corresponds the string ID of a speaker in the corresponding subset of the dataset. 
This file is mainly used for training multi-speaker TTS models.

For example, 
```
116
1255
1272
1462
1585
1630
1650
1651
1673
1686
```

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/datasets#table-of-contents)


#### idx2gen
In _idx2gen_, each line corresponds to a pair of file index and gender. 

For example, 
```
103-1240-0000 F
103-1240-0001 F
103-1240-0002 F
103-1240-0003 F
103-1240-0004 F
```

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/datasets#table-of-contents)


## How to Dump a Dataset on your Machine
For dumping an existing dataset, 
1. Change the absolute paths at the top of `${SPEECHAIN_ROOT}/datasets/data_dumping.sh` to the corresponding places on your machine.
2. Move to the folder of your target dataset `${SPEECHAIN_ROOT}/datasets/{dataset_name}`
3. Change the absolute paths at the top of `${SPEECHAIN_ROOT}/datasets/{dataset_name}/run.sh` to the corresponding places on your machine.
4. Run `./run.sh -h` to familiarize yourself with the involved arguments.
5. Run `./run.sh` to dump your target dataset (add some arguments if needed).

In `${SPEECHAIN_ROOT}/datasets/data_dumping.sh`, there are 8 steps to dump your target dataset from the internet to your local machine:
1. **(Mandatory) Data downloading**:  
    In this step, the raw data of your target dataset is downloaded from the internet to `${SPEECHAIN_ROOT}/datasets/{dataset_name}/wav` by `${SPEECHAIN_ROOT}/datasets/{dataset_name}/data_download.sh`.
2. **(Mandatory) Metadata generation**:  
    In this step, the metadata files of your target dataset is generated in `${SPEECHAIN_ROOT}/datasets/dataset_name/wav` by `${SPEECHAIN_ROOT}/datasets/dataset_name/meta_generator.py`. 
    The metadata files will be saved to `${SPEECHAIN_ROOT}/datasets/{dataset_name}/wav/{subset_name}/`.
3. **(Optional) Waveform downsampling**:  
    In this step, the original waveform files are downsampled to your specified sampling rate. 
    This step is not mandatory if the sampling rate of the original files is OK for your experiments.
    The downsampled waveform files will be saved to `${SPEECHAIN_ROOT}/datasets/{dataset_name}/wav{sample_rate}` where _sample_rate_ is your target sampling rate.
4. **(Optional) Acoustic feature extraction**:  
    In this step, the acoustic features are extracted by your given configuration file in `${SPEECHAIN_ROOT}/speechain/config/feat/`.
    This step is not mandatory since our toolkit supports on-the-fly acoustic feature extraction during training.
    The extracted acoustic features will be saved to `${SPEECHAIN_ROOT}/datasets/{dataset_name}/{feat_config}` where _feat_config_ is the name of your given configuration file.
5. **(Mandatory) Data length generation**:  
    In this step, the file length will be extracted. 
    If the source data is waveform, the number of sampling points of each waveform will be recorded.
    If the source data is acoustic features, the number of time frames of each feature will be recorded.
    The extracted data length will be saved to `${SPEECHAIN_ROOT}/datasets/{dataset_name}/wav/idx2wav_len` for waveforms or `${SPEECHAIN_ROOT}/datasets/dataset_name/{feat_config}/idx2feat_len` for acoustic features.
6. **(Optional) Speaker Embedding Extraction**:  
    In this step, the speaker embeddings of the given waveforms will be extracted. 
    The extracted speaker embeddings will be saved to `${SPEECHAIN_ROOT}/datasets/{dataset_name}/wav/idx2{spk_emb_model}_spk_len` where `spk_emb_model` is either `xvector` or `ecapa`.  
    
    **Note:** This step can only process the waveform files.  
8. **(Optional) Data Package**:  
    In this step, all the individual data files (waveform files or acoustic feature files) will be packaged into several compressed binary chunk files.
    Those chunk packages will make the subsequent model training smoother and more efficient. 
    This step works better for large-scale datasets, so if your target dataset is not so large, you can simply skip this step.
    The packaged chunk files will be save to `${SPEECHAIN_ROOT}/datasets/{dataset_name}/wav/wav_{comp_chunk_ext}` and `${SPEECHAIN_ROOT}/datasets/{dataset_name}/wav/idx2wav_{comp_chunk_ext}`.
9. **(Optional) Metadata post-processing**:  
    In this step, all the metadata files generated in the above steps (not only the ones in the step 2) are post-processed by `${SPEECHAIN_ROOT}/datasets/dataset_name/meta_post_processor.py`.
    If there is no _meta_post_processor.py_ in `${SPEECHAIN_ROOT}/datasets/{dataset_name}/`, this step will be skipped.
10. **(Mandatory) Vocabulary generation**:  
     In this step, the token vocabulary and text length will be extracted based on `${SPEECHAIN_ROOT}/datasets/dataset_name/wav/{src_subset}/text` and `${SPEECHAIN_ROOT}/datasets/dataset_name/wav/{tgt_subset}/idx2text`.
     The extracted token vocabulary will be saved to `${SPEECHAIN_ROOT}/datasets/{dataset_name}/token_type/{src_subset}/{token_config}/{txt_format}/vocab`. 
     The extracted text lengths will be saved to `${SPEECHAIN_ROOT}/datasets/{dataset_name}/token_type/{src_subset}/{token_config}/{txt_format}/{tgt_subset}/idx2text_len`.
    
     **Note**: For subword tokenizers, there will be an additional file named `model` in `${SPEECHAIN_ROOT}/datasets/{dataset_name}/token_type/{src_subset}/{token_config}/{txt_format}` because the subword tokenization is done by third-party packages in this toolkit.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/datasets#table-of-contents)

## How to Contribute a New Dataset
If the dataset that you want to use for your experiments is not included here, 
you could make the dumping pipeline of your target dataset by the following instructions:
1. Move to `${SPEECHAIN_ROOT}/datasets/`.
2. Run `./data_dumping.sh -h` to familiarize yourself with the involved arguments.
3. Make a new sub-folder in `${SPEECHAIN_ROOT}/datasets/` with the folder name as the name of your target dataset.
4. Make a new ***data_download.sh*** in `${SPEECHAIN_ROOT}/datasets/{dataset_name}` to download your target dataset from the internet.
You could refer to the ones in the existing dataset sub-folders as a template.
5. Make a new ***meta_generator.py*** in `${SPEECHAIN_ROOT}/datasets/{dataset_name}` to extract the metadata files of your target dataset. 
You could refer to `${SPEECHAIN_ROOT}/datasets/meta_generator.py` for instructions.
6. If needed, make a new ***meta_post_processor.py*** in `${SPEECHAIN_ROOT}/datasets/{dataset_name}` to post-process all the extracted metadata files of your target dataset. 
You could refer to `${SPEECHAIN_ROOT}/datasets/meta_post_processor.py` for instructions.
7. Make a new ***run.sh*** in `${SPEECHAIN_ROOT}/datasets/{dataset_name}` to manipulate the dumping pipeline of your target dataset. 
You could refer to the ones in the existing dataset sub-folders as a template.
8. Run **_run.sh_** in the terminal by `./run.sh` with your specified arguments. 

**Note**: You should keep the same script names (i.e., `data_download.sh`, `meta_generator.py`, and `meta_post_processor.py`) for the compatibility with `data_dumping.sh`.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/datasets#table-of-contents)
