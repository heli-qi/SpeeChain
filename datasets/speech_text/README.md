# Speech-Text Datasets
This folder contains all the available datasets that are made up of speech and text data. 
Each dataset corresponds to a sub-folder and has a uniform file system. 
You can easily dump your target dataset to your machine by following the instructions below. 
If you want to contribute a new dataset, we would appreciate it if you could follow our file systems and metadata formats.

👆[Back to the handbook page](https://github.com/ahclab/SpeeChain/tree/main/datasets#datasets-folder-of-the-speechain-toolkit)

## Table of Contents
1. [**File System**](https://github.com/ahclab/SpeeChain/tree/main/datasets/speech_text#file-system)
2. [**Metadata Format**](https://github.com/ahclab/SpeeChain/tree/main/datasets/speech_text#file-format)
    1. [idx2wav](https://github.com/ahclab/SpeeChain/tree/main/datasets/speech_text#idx2wav)
    2. [idx2wav_len](https://github.com/ahclab/SpeeChain/tree/main/datasets/speech_text#idx2wav_len)
    3. [idx2feat](https://github.com/ahclab/SpeeChain/tree/main/datasets/speech_text#idx2feat)
    4. [idx2feat_len](https://github.com/ahclab/SpeeChain/tree/main/datasets/speech_text#idx2feat_len)
    5. [idx2text](https://github.com/ahclab/SpeeChain/tree/main/datasets/speech_text#idx2sent)
    6. [idx2text_len](https://github.com/ahclab/SpeeChain/tree/main/datasets/speech_text#idx2sent_len)
    7. [text](https://github.com/ahclab/SpeeChain/tree/main/datasets/speech_text#text)
    8. [idx2spk](https://github.com/ahclab/SpeeChain/tree/main/datasets/speech_text#idx2spk)
    9. [spk_list](https://github.com/ahclab/SpeeChain/tree/main/datasets/speech_text#spk_list)
    10. [idx2gen](https://github.com/ahclab/SpeeChain/tree/main/datasets/speech_text#idx2gen)
3. [**How to Dump a Dataset on your Machine**](https://github.com/ahclab/SpeeChain/tree/main/datasets/speech_text#how-to-dump-a-dataset-on-your-machine)
4. [**How to Contribute a New Dataset**](https://github.com/ahclab/SpeeChain/tree/main/datasets/speech_text#how-to-contribute-a-new-dataset)

## File System
```
/datasets
    /pyscripts                  # fixed .py scripts provided by the toolkit
    /speech_text
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
                        {txt_format}_text           # the file that contains only transcript sentences. (txt_format is the format of processed text which is used to distinguish different idx2sent files)
                        ...                         # (optional) some other metadata files available in the dataset (such as idx2spk, idx2gen, ...)
                /wav{sample_rate}           # (optional) downsampled waveform folder (sample_rate is the samping rate of waveforms after downsampling)
                    ...                         # same structure as '/wav'
                /{feat_config}              # (optional) acoustic feature folder by a given configuration (feat_config is the name of the configuration file)
                    /{subset_name}              # the name of a subset in this dataset 
                        /feat_{comp_file_ext}       # (optional) the folder that contains the compressed chunk files of all the acoustic feature data
                        idx2feat_{comp_file_ext}    # (optional) the file that contains the pairs of index and the absolute address of waveforms in the compressed package files
                        idx2feat                    # the file that contains the pairs of index and the absolute address of acoustic feature files
                        idx2feat_len                # the file that contains the pairs of index and feature length
                        idx2{txt_format}_text       # the file that contains the pairs of index and transcript text. (txt_format is the format of processed text which is used to distinguish different idx2text files)
                        {txt_format}_text           # the file that contains only transcript sentences. (txt_format is the format of processed text which is used to distinguish different idx2text files)
                        ...                         # (optional) some metadata files available in the dataset (such as idx2spk, idx2gen, ...)
                /{token_type}               # token folder of a specific type (token_type is the name of the specified token type)
                    /{src_subset}               # the source subset used for generating the token vocabulary (src_subset is the name of the used source subset)
                        /{token_config}             # the configuration for generating the vocabulary list.
                            /{txt_format}               # the format of text data used to generate the token vocabulary
                                /{tgt_subset}               # the target subset whose text data is used for tokenization
                                    idx2text_len                # the file that contains the pairs of index and transcript text length after tokenization
                                vocab                       # token vocabulary
                                model                       # (optional) tokenizer model. Used for subword tokenizers.
            run.sh                      # the dataset-specific .sh script that controls how the data dumping for the corresponding dataset is going
            data_download.sh            # the dataset-specific .sh script that downloads the corresponding dataset from the internet
            meta_generator.py           # the dataset-specific .py script that generates the meta data files of the dataset (idx2wav, idx2spk, ...)
            meta_post_processor.py      # (optional) the dataset-specific .py script that performs post-processing for the meta data files (needed by some datasets like LibriSpeech)
```
The names in the braces({}) mean the undefined names depending on the settings of datasets and configuration.

👆[Back to the table of contents]()


## Metadata Format
The data formats of all metadata files are uniform for all datasets. 
The format uniformity enables the automatic configuration for the data loading part of our toolkit. 
The format for each metadata file is shown below.

#### idx2wav
In `idx2wav`, each line corresponds to a pair of file index and file absolute address. 
Index and address are separated by a blank. 

For example, 
```
103-1240-0000 /x/xx/xxx/datasets/speech_text/librispeech/data/wav/train-clean-100/103/1240/103-1240-0000.flac
103-1240-0001 /x/xx/xxx/datasets/speech_text/librispeech/data/wav/train-clean-100/103/1240/103-1240-0001.flac
103-1240-0002 /x/xx/xxx/datasets/speech_text/librispeech/data/wav/train-clean-100/103/1240/103-1240-0002.flac
103-1240-0003 /x/xx/xxx/datasets/speech_text/librispeech/data/wav/train-clean-100/103/1240/103-1240-0003.flac
103-1240-0004 /x/xx/xxx/datasets/speech_text/librispeech/data/wav/train-clean-100/103/1240/103-1240-0004.flac
```
Any audio files that can be processed by `soundfile.read()` (such as .flac, .wav, ...) are OK in _idx2wav_.

You can choose to package all the waveform files in a dataset to some compressed chunk files in `idx2wav_{comp_file_ext}`.
For example in `idx2wav_npz` where .npz is used as the extension of chunk files:
```
103-1240-0000 /x/xx/xxx/datasets/speech_text/librispeech/data/wav/train-clean-100/wav_npz/chunk_83.npz:103-1240-0000
103-1240-0001 /x/xx/xxx/datasets/speech_text/librispeech/data/wav/train-clean-100/wav_npz/chunk_13.npz:103-1240-0001
103-1240-0002 /x/xx/xxx/datasets/speech_text/librispeech/data/wav/train-clean-100/wav_npz/chunk_87.npz:103-1240-0002
103-1240-0003 /x/xx/xxx/datasets/speech_text/librispeech/data/wav/train-clean-100/wav_npz/chunk_62.npz:103-1240-0003
103-1240-0004 /x/xx/xxx/datasets/speech_text/librispeech/data/wav/train-clean-100/wav_npz/chunk_115.npz:103-1240-0004
```
The absolute addresses are separated by colons where the content before the colon indicates which chunk the waveform belongs to and the content after the colon indicates the data index in the chunk.

👆[Back to the table of contents]()


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

👆[Back to the table of contents]()


#### idx2feat
In `idx2feat`, each line corresponds to a pair of file index and file absolute address. 
The index and absolute address are separated by a blank. 

For example, 
```
103-1240-0000 /x/xx/xxx/datasets/speech_text/librispeech/data/{feat_config}/train-clean-100/103-1240-0000.npy
103-1240-0001 /x/xx/xxx/datasets/speech_text/librispeech/data/{feat_config}/train-clean-100/103-1240-0001.npy
103-1240-0002 /x/xx/xxx/datasets/speech_text/librispeech/data/{feat_config}/train-clean-100/103-1240-0002.npy
103-1240-0003 /x/xx/xxx/datasets/speech_text/librispeech/data/{feat_config}/train-clean-100/103-1240-0003.npy
103-1240-0004 /x/xx/xxx/datasets/speech_text/librispeech/data/{feat_config}/train-clean-100/103-1240-0004.npy
```
feat_config is the name of the used feature extraction configuration file.
In our toolkit, acoustic feature of a waveform is saved as a .npy file by the NumPy package.

You can choose to package all the acoustic feature files in a dataset to some compressed chunk files in `idx2feat_{comp_file_ext}`.
For example in `idx2feat_npz` where .npz is used as the extension of chunk files:
```
103-1240-0000 /x/xx/xxx/datasets/speech_text/librispeech/data/{feat_config}/train-clean-100/feat_npz/chunk_83.npz:103-1240-0000
103-1240-0001 /x/xx/xxx/datasets/speech_text/librispeech/data/{feat_config}/train-clean-100/feat_npz/chunk_13.npz:103-1240-0001
103-1240-0002 /x/xx/xxx/datasets/speech_text/librispeech/data/{feat_config}/train-clean-100/feat_npz/chunk_87.npz:103-1240-0002
103-1240-0003 /x/xx/xxx/datasets/speech_text/librispeech/data/{feat_config}/train-clean-100/feat_npz/chunk_62.npz:103-1240-0003
103-1240-0004 /x/xx/xxx/datasets/speech_text/librispeech/data/{feat_config}/train-clean-100/feat_npz/chunk_115.npz:103-1240-0004
```
The absolute addresses are separated by colons where the content before the colon indicates which chunk the waveform belongs to and the content after the colon indicates the data index in the chunk.

👆[Back to the table of contents]()


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

👆[Back to the table of contents]()


#### idx2text
In `idx2text`, each line corresponds to a pair of file index and transcript text string which are separated by a blank. 
`idx2text` will be renamed as `idx2{txt_format}_text` to indicate the text processing format used to generate this file.
In our toolkit, there are many available text processing format to generate transcript text strings with different styles.

For example, 
```
103-1240-0000 CHAPTER ONE MISSUS RACHEL LYNDE IS SURPRISED MISSUS RACHEL LYNDE LIVED JUST WHERE THE AVONLEA MAIN ROAD DIPPED DOWN INTO A LITTLE HOLLOW FRINGED WITH ALDERS AND LADIES EARDROPS AND TRAVERSED BY A BROOK
103-1240-0001 THAT HAD ITS SOURCE AWAY BACK IN THE WOODS OF THE OLD CUTHBERT PLACE IT WAS REPUTED TO BE AN INTRICATE HEADLONG BROOK IN ITS EARLIER COURSE THROUGH THOSE WOODS WITH DARK SECRETS OF POOL AND CASCADE BUT BY THE TIME IT REACHED LYNDE'S HOLLOW IT WAS A QUIET WELL CONDUCTED LITTLE STREAM
103-1240-0002 FOR NOT EVEN A BROOK COULD RUN PAST MISSUS RACHEL LYNDE'S DOOR WITHOUT DUE REGARD FOR DECENCY AND DECORUM IT PROBABLY WAS CONSCIOUS THAT MISSUS RACHEL WAS SITTING AT HER WINDOW KEEPING A SHARP EYE ON EVERYTHING THAT PASSED FROM BROOKS AND CHILDREN UP
103-1240-0003 AND THAT IF SHE NOTICED ANYTHING ODD OR OUT OF PLACE SHE WOULD NEVER REST UNTIL SHE HAD FERRETED OUT THE WHYS AND WHEREFORES THEREOF THERE ARE PLENTY OF PEOPLE IN AVONLEA AND OUT OF IT WHO CAN ATTEND CLOSELY TO THEIR NEIGHBOR'S BUSINESS BY DINT OF NEGLECTING THEIR OWN
103-1240-0004 BUT MISSUS RACHEL LYNDE WAS ONE OF THOSE CAPABLE CREATURES WHO CAN MANAGE THEIR OWN CONCERNS AND THOSE OF OTHER FOLKS INTO THE BARGAIN SHE WAS A NOTABLE HOUSEWIFE HER WORK WAS ALWAYS DONE AND WELL DONE SHE RAN THE SEWING CIRCLE
```
**Note**: you don't need to worry about the blanks inside each transcript text string. 
Those additional blanks will be ignored in the subsequent processing.

👆[Back to the table of contents]()


#### text
In `text`, each line corresponds to a transcript sentence string
This file is mainly used for token vocabulary generation.
`text` will be renamed as `{txt_format}_text` to indicate the text processing format used to generate this file.
In our toolkit, there are many available text processing format to generate transcript text strings with different styles.

For example, 
```
CHAPTER ONE MISSUS RACHEL LYNDE IS SURPRISED MISSUS RACHEL LYNDE LIVED JUST WHERE THE AVONLEA MAIN ROAD DIPPED DOWN INTO A LITTLE HOLLOW FRINGED WITH ALDERS AND LADIES EARDROPS AND TRAVERSED BY A BROOK
THAT HAD ITS SOURCE AWAY BACK IN THE WOODS OF THE OLD CUTHBERT PLACE IT WAS REPUTED TO BE AN INTRICATE HEADLONG BROOK IN ITS EARLIER COURSE THROUGH THOSE WOODS WITH DARK SECRETS OF POOL AND CASCADE BUT BY THE TIME IT REACHED LYNDE'S HOLLOW IT WAS A QUIET WELL CONDUCTED LITTLE STREAM
FOR NOT EVEN A BROOK COULD RUN PAST MISSUS RACHEL LYNDE'S DOOR WITHOUT DUE REGARD FOR DECENCY AND DECORUM IT PROBABLY WAS CONSCIOUS THAT MISSUS RACHEL WAS SITTING AT HER WINDOW KEEPING A SHARP EYE ON EVERYTHING THAT PASSED FROM BROOKS AND CHILDREN UP
AND THAT IF SHE NOTICED ANYTHING ODD OR OUT OF PLACE SHE WOULD NEVER REST UNTIL SHE HAD FERRETED OUT THE WHYS AND WHEREFORES THEREOF THERE ARE PLENTY OF PEOPLE IN AVONLEA AND OUT OF IT WHO CAN ATTEND CLOSELY TO THEIR NEIGHBOR'S BUSINESS BY DINT OF NEGLECTING THEIR OWN
BUT MISSUS RACHEL LYNDE WAS ONE OF THOSE CAPABLE CREATURES WHO CAN MANAGE THEIR OWN CONCERNS AND THOSE OF OTHER FOLKS INTO THE BARGAIN SHE WAS A NOTABLE HOUSEWIFE HER WORK WAS ALWAYS DONE AND WELL DONE SHE RAN THE SEWING CIRCLE
```

👆[Back to the table of contents]()


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

👆[Back to the table of contents]()


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

👆[Back to the table of contents]()


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

👆[Back to the table of contents]()


## How to Dump a Dataset on your Machine
For dumping an existing dataset, 
1. Change the absolute paths at the top of `/datasets/speech_text/data_dumping.sh` to the corresponding places on your machine.
2. Move to the folder of your target dataset `/datasets/speech_text/{dataset_name}`
3. Change the absolute paths at the top of `/datasets/speech_text/{dataset_name}/run.sh` to the corresponding places on your machine.
4. Run `./run.sh -h` to familiarize yourself with the involved arguments.
5. Run `./run.sh` to dump your target dataset (add some arguments if needed).

In `/datasets/speech_text/data_dumping.sh`, there are 8 steps to dump your target dataset from the internet to your local machine:
1. **(Mandatory) Data downloading**:  
    In this step, the raw data of your target dataset is downloaded from the internet to `/datasets/speech_text/{dataset_name}/wav` by `/datasets/speech_text/{dataset_name}/data_download.sh`.
2. **(Mandatory) Metadata generation**:  
    In this step, the metadata files of your target dataset is generated in `/datasets/speech_text/dataset_name/wav` by `/datasets/speech_text/dataset_name/meta_generator.py`. 
    The metadata files will be saved to `/datasets/speech_text/{dataset_name}/wav/{subset_name}/`.
3. **(Optional) Waveform downsampling**:  
    In this step, the original waveform files are downsampled to your specified sampling rate. 
    This step is not mandatory if the sampling rate of the original files is OK for your experiments.
    The downsampled waveform files will be saved to `/datasets/speech_text/{dataset_name}/wav{sample_rate}` where _sample_rate_ is your target sampling rate.
4. **(Optional) Acoustic feature extraction**:  
    In this step, the acoustic features are extracted by your given configuration file in `/speechain/config/feat/`.
    This step is not mandatory since our toolkit supports on-the-fly acoustic feature extraction during training.
    The extracted acoustic features will be saved to `/datasets/speech_text/{dataset_name}/{feat_config}` where _feat_config_ is the name of your given configuration file.
5. **(Mandatory) Data length generation**:  
    In this step, the file length will be extracted. 
    If the source data is waveform, the number of sampling points of each waveform will be recorded.
    If the source data is acoustic features, the number of time frames of each feature will be recorded.
    The extracted data length will be saved to `/datasets/speech_text/{dataset_name}/wav/idx2wav_len` for waveforms or `/datasets/speech_text/dataset_name/{feat_config}/idx2feat_len` for acoustic features.
6. **(Optional) Data Package**:  
    In this step, all the individual data files (waveform files or acoustic feature files) will be packaged into several compressed binary chunk files.
    Those chunk packages will make the subsequent model training smoother and more efficient. 
    This step works better for large-scale datasets, so if your target dataset is not so large, you can simply skip this step.
    The packaged chunk files will be save to `/datasets/speech_text/{dataset_name}/wav/wav_{comp_chunk_ext}` and `/datasets/speech_text/{dataset_name}/wav/idx2wav_{comp_chunk_ext}`.
7. **(Optional) Metadata post-processing**:  
    In this step, all the metadata files generated in the above steps (not only the ones in the step 2) are post-processed by `/datasets/speech_text/dataset_name/meta_post_processor.py`.
    If there is no _meta_post_processor.py_ in `/datasets/speech_text/{dataset_name}/`, this step will be skipped.
8. **(Mandatory) Vocabulary generation**:  
    In this step, the token vocabulary and text length will be extracted based on `/datasets/speech_text/dataset_name/wav/{src_subset}/text` and `/datasets/speech_text/dataset_name/wav/{tgt_subset}/idx2text`.
    The extracted token vocabulary will be saved to `/datasets/speech_text/{dataset_name}/token_type/{src_subset}/{token_config}/{txt_format}/vocab`. 
    The extracted text lengths will be saved to `/datasets/speech_text/{dataset_name}/token_type/{src_subset}/{token_config}/{txt_format}/{tgt_subset}/idx2text_len`.
    
    **Note**: For subword tokenizers, there will be an additional file named `model` in `/datasets/speech_text/{dataset_name}/token_type/{src_subset}/{token_config}/{txt_format}` because the subword tokenization is done by third-party packages in this toolkit.

👆[Back to the table of contents]()

## How to Contribute a New Dataset
If the speech-text dataset that you want to use for your experiments is not included here, 
you could make the dumping pipeline of your target dataset by the following instructions:
1. Change the absolute paths at the top of `/datasets/speech_text/data_dumping.sh` to the corresponding places on your machine.
2. Run `./data_dumping.sh -h` to familiarize yourself with the involved arguments.
3. Make a new sub-folder in `/datasets/speech_text/` with the folder name as the dataset name.
4. Make a new ***data_download.sh*** in `/datasets/speech_text/{dataset_name}` to download your target dataset from the internet.
You could refer to the ones in the existing dataset sub-folders as a template.
5. Make a new ***meta_generator.py*** in `/datasets/speech_text/{dataset_name}` to extract the metadata files of your target dataset. 
You could refer to `/datasets/speech_text/meta_generator.py` for instructions.
6. If needed, make a new ***meta_post_processor.py*** in `/datasets/speech_text/{dataset_name}` to post-process all the extracted metadata files of your target dataset. 
You could refer to `/datasets/speech_text/meta_post_processor.py` for instructions.
7. Make a new ***run.sh*** in `/datasets/speech_text/{dataset_name}` to manipulate the dumping pipeline of your target dataset. 
You could refer to the ones in the existing dataset sub-folders as a template.
8. Run **_run.sh_** in the terminal by `./run.sh` with your specified arguments. 

**Note**: You should keep the same script names (i.e., `data_download.sh`, `meta_generator.py`, and `meta_post_processor.py`) for the compatibility with `data_dumping.sh`.

👆[Back to the table of contents]()
