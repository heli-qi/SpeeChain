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
4. [**How to Extract Speaker Embedding by my own model**](https://github.com/ahclab/SpeeChain/tree/main/datasets#how-to-extract-speaker-embedding-by-my-own-model)
6. [**How to Contribute a New Dataset**](https://github.com/ahclab/SpeeChain/tree/main/datasets#how-to-contribute-a-new-dataset)

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
   1. Go to the folder of your target dataset `${SPEECHAIN_ROOT}/datasets/{dataset_name}` (e.g. if you want to dump LibriTTS, please go to `${SPEECHAIN_ROOT}/datasets/libritts`)
   2. Run `bash run.sh --help` to familiarize yourself with the involved arguments.
   3. Run `bash run.sh` to dump your target dataset (add some arguments if needed).

**Note:**  
   1. **If you already have the _decompressed_ dataset on your disk**, please attach the argument `--src_path {the-path-of-your-existing-dataset}` to the command `bash run.sh` in the no.3 step above.
   Please make sure that `src_path` is an absolute path starting with a slash '/' and the content of `src_path` should be exactly the same with the one downloaded from the internet (please see the help message of `--src_path` in each `run.sh`).
   2. **If you want to save the dumped data and metadata files outside the toolkit folder (`${SPEECHAIN_ROOT}`)**, please attach the argument `--tgt_path {the-path-you-want-to-save-files}` to the command `bash run.sh` in the no.3 step above.
   Please make sure that `tgt_path` is an absolute path starting with a slash '/'.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/datasets#table-of-contents)

## How to Extract Speaker Embedding by my own model
If you want to use the pretrained speaker embedding model on your machine, please
1. Don't give the argument `--spk_emb_model` when running the command `bash run.sh`
3. Write your own extraction script. You can use the metadata files `idx2wav` and `idx2wav_len` to read and organize the audio files. 
Please save all the speaker embedding vectors to a specific folder in the same directory of `idx2wav` and give a metadata file named `idx2spk_feat` for data reference.  
**Note:** 
   1. For the file format of `idx2spk_feat`, please click [here](https://github.com/ahclab/SpeeChain/tree/main/datasets#idx2spk_feat) for reference.
   2. Please keep the same data index with `idx2wav` in your `idx2spk_feat`.
   3. Each speaker embedding vector should be in the shape of `[1, spk_feat_dim]`.
   4. Speaker embedding vectors could be saved in two ways:
      1. save each vector to an individual `.npy` file
      2. save all vectors to a `.npz` file where the index of each vector is exactly the one in `idx2spk_feat`.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/datasets#table-of-contents)

## How to Contribute a New Dataset
If the dataset that you want to use for your experiments is not included here, 
you could make the dumping pipeline of your target dataset by the following instructions:
1. Go to `${SPEECHAIN_ROOT}/datasets/`.
2. Run `bash data_dumping.sh --help` to familiarize yourself with the involved arguments.
3. Make a new folder in `${SPEECHAIN_ROOT}/datasets/` with the name as your target dataset.
4. Make a new ***data_download.sh*** in `${SPEECHAIN_ROOT}/datasets/{dataset_name}` to download your target dataset from the internet.
Please download the dataset into `${SPEECHAIN_ROOT}/datasets/{dataset_name}/data/wav`.
5. Make a new ***meta_generator.py*** in `${SPEECHAIN_ROOT}/datasets/{dataset_name}` to extract the metadata files of your target dataset. 
Please refer to `${SPEECHAIN_ROOT}/datasets/meta_generator.py` for instructions of how to override the pipeline of metadata generation.
6. If needed, make a new ***meta_post_processor.py*** in `${SPEECHAIN_ROOT}/datasets/{dataset_name}` to post-process the extracted metadata files of all the subsets. 
(e.g. combine _train-clean-100_ and _train-clean-360_ of _LibriSpeech_ into _train-clean-460_)
Please refer to `${SPEECHAIN_ROOT}/datasets/meta_post_processor.py` for instructions of how to override the pipeline of metadata post-processing.
7. Make a new ***run.sh*** in `${SPEECHAIN_ROOT}/datasets/{dataset_name}` to manipulate the dumping pipeline of your target dataset. 
You could refer to the ones in the existing dataset folders as a template.

**Note**: Please keep the same script names (i.e., `data_download.sh`, `meta_generator.py`, and `meta_post_processor.py`) for the compatibility with `data_dumping.sh`.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/datasets#table-of-contents)
