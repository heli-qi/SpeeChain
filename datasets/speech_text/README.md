# Speech-Text Datasets
This folder contains all the available datasets that are made up of speech and text data. 
Each dataset corresponds to a sub-folder and has a uniform file system. 
You can easily dump your target dataset to your machine by following the instructions. 
If you want to contribute a new dataset, we would appreciate it if you could follow our data formats and file systems.

👆[Back to the dataset home page](https://github.com/ahclab/SpeeChain/tree/main/datasets#datasets-folder-of-the-speechain-toolkit)

## Table of Contents
1. [**File System**](https://github.com/ahclab/SpeeChain/tree/main/datasets/speech_text#file-system)
2. [**Data Format**](https://github.com/ahclab/SpeeChain/tree/main/datasets/speech_text#data-format)
    1. [General data](https://github.com/ahclab/SpeeChain/tree/main/datasets/speech_text#general-data)
        1. [idx2wav & idx2feat](https://github.com/ahclab/SpeeChain/tree/main/datasets/speech_text#idx2wav)
        2. [idx2wav_len & idx2feat_len](https://github.com/ahclab/SpeeChain/tree/main/datasets/speech_text#idx2wav_len)
        3. [idx2sent](https://github.com/ahclab/SpeeChain/tree/main/datasets/speech_text#idx2sent)
        4. [idx2sent_len](https://github.com/ahclab/SpeeChain/tree/main/datasets/speech_text#idx2sent_len)
        5. [text](https://github.com/ahclab/SpeeChain/tree/main/datasets/speech_text#text)
    2. [Meta data](https://github.com/ahclab/SpeeChain/tree/main/datasets/speech_text#meta-data)
        1. [idx2spk](https://github.com/ahclab/SpeeChain/tree/main/datasets/speech_text#idx2spk)
        2. [idx2gen](https://github.com/ahclab/SpeeChain/tree/main/datasets/speech_text#idx2gen)
3. [**The Procedure of Dumping a Dataset on your Machine**](https://github.com/ahclab/SpeeChain/tree/main/datasets/speech_text#the-procedure-of-dumping-a-dataset-on-your-machine)
3. [**How to Contribute a New Dataset**](https://github.com/ahclab/SpeeChain/tree/main/datasets/speech_text#how-to-contribute-a-new-dataset)

## File System
```
/speech_text
    /dataset_name               # each dataset has a sub-folder
        /data                       # all the data of the dataset
            /wav                        # the unprocessed raw data of the dataset
                /subset_name                # (optional) the name of a subset in this dataset 
                    idx2wav                     # the file that contains the index and waveform file absolute path
                    idx2wav_len                 # the file that contains the index and waveform length
                    idx2sent                    # the file that contains the index and transcripts
                    text                        # the file that contains the index and transcripts
                    ...                         # some metadata files available in the dataset (such as idx2spk, idx2gen, ...)
                ...
            /feat_name                  # the sub-folder of a type of feature extracted from the raw data
                /subset_name                # (optional) the name of a subset in this dataset 
                    idx2feat                    # the file that contains the index and feature file absolute path
                    idx2feat_len                # the file that contains the index and feature length
                    idx2sent                    # the file that contains the index and transcripts
                    text                        # the file that contains the index and transcripts
                    ...                         # some metadata files available in the dataset (such as idx2spk, idx2gen, ...)
                ...
            /token_type                 # the sub-folder of a type of token
                /src_subset                 # (optional) the source subset used for generating the vocabulary list
                    /token_config               # (optional) the configuration for generating the vocabulary list. Used for subword tokenizers.
                        /tgt_subset                 # the target subset whose text data is used for tokenization
                            idx2sent_len                # the file that contains the index and sentence length after tokenization
                        ...                         # other target subsets
                        vocab                       # vocabulary list
                        model                       # (optional) tokenizer model. Used for subword tokenizers.
        data_download.sh            # the .sh script that downloads the dataset from the internet
        stat_info_generator.py      # the .py script that generates the statistic information of the dataset (feat.scp, text, metadata, ...)
        data_dumping.sh             # the .sh script that contains the complete pipeline of data dumping (i.e. data preparation)
        run.sh                      # the .sh script that controls how the data dumping is going
```
**dataset_name** is the first-level sub-folder that indicates which dataset the data in this folder comes from.
1. **data** is the second-level sub-folder that contains all kinds of the data obtained from the dataset. 
The folder name _data_ is shared by all datasets. 
    1. **wav** is the third-level sub-folder that contains the waveform data files of the dataset. 
    The name _wav_ is shared by all datasets.
        1. **subset_name** is the fourth-level sub-folder that indicates a subset of the dataset. 
        This folder level is only useful for the datasets that are divided into independent subsets (e.g. LibriSpeech).
            1. **idx2wav** contains the data sample index and the corresponding waveform file absolute path in each line.
            2. **idx2wav_len** contains the data sample index and the corresponding waveform file length in each line.
            3. **idx2sent** contains the data sample index and the corresponding transcript sentence in each line.
            4. **text** is the one-sentence-per-line file used for vocabulary list generation.
    2. **feat_name** is the third-level sub-folder that contains the acoustic feature extracted from the waveform data files. 
        1. **subset_name** is the fourth-level sub-folder that indicates a subset of the dataset. 
        This folder level is only useful for the datasets that are divided into independent subsets (e.g. LibriSpeech).
            1. **idx2feat** contains the data sample index and the corresponding feature file absolute path in each line.
            2. **idx2feat_len** contains the data sample index and the corresponding feature file length in each line.
            3. **idx2sent** contains the data sample index and the corresponding transcript sentence in each line.
            4. **text** is the one-sentence-per-line file used for vocabulary list generation.
    3. **token_type** is the third-level sub-folder that corresponds to a specific type of token.
        1. **dict** indicates the token dictionary extracted from the text data in the dataset. 
        The name _dict_ is shared by all the token sub-folders.
2. **data_download.sh** is the script used to download the dataset from the internet. 
Basically, this file is made up of _wget_ and _tar_ commands.
3. **stat_info_generator.py** is the script used to generate the statistical information of the dataset. 
The output files need to cover at least _idx2wav_, _idx2sent_, and _text_. 
If you want to generate some metadata files, please give the processing codes in this file.
4. **data_dumping.sh** is the script that contains the entire pipeline of data dumping. 
There are several steps in the dumping pipeline and each one can be done in a dataset-independent way if you follow our data format and file system. 
5. **run.sh** is the script that controls the starting and ending steps of the dumping process. 
It also controls the input arguments for dumping the dataset. 

For more details, please refer to */datasets/speech_text/librispeech* as an example.

## Data Format
The data format of all statistic files should be uniform for all datasets. 
The format uniformity enables the automatic configuration for the data loading part of our toolkit. 
The format for each file is shown below.

### General data
There are some statistic files that contain the general data shared by all speech-text datasets. 
These general data are necessary for running an ASR or TTS model.

#### idx2wav & idx2feat
In _idx2wav_, each waveform file corresponds to a row and each row should be in the form of _file name + blank + file path_. 

For example, 
```
103-1240-0000 /x/xx/xxx/datasets/speech/librispeech/data/wav/train_clean_100/103/1240/103-1240-0000.flac
103-1240-0001 /x/xx/xxx/datasets/speech/librispeech/data/wav/train_clean_100/103/1240/103-1240-0001.flac
103-1240-0002 /x/xx/xxx/datasets/speech/librispeech/data/wav/train_clean_100/103/1240/103-1240-0002.flac
103-1240-0003 /x/xx/xxx/datasets/speech/librispeech/data/wav/train_clean_100/103/1240/103-1240-0003.flac
103-1240-0004 /x/xx/xxx/datasets/speech/librispeech/data/wav/train_clean_100/103/1240/103-1240-0004.flac
```

If you would like to extract acoustic features from the waveforms in advance, 
please prepare _idx2feat_ to notify the toolkit the absolute path of your feature files. 
The structure of _idx2feat_ is the same as _idx2wav_.

#### idx2wav_len & idx2feat_len
In _idx2wav_len_, each waveform file corresponds to a row and each row should be in the form of _file name + blank + length_. 
The length means the number of sampling points of the corresponding waveform file.

For example, 
```
103-1240-0000 225360
103-1240-0001 255120
103-1240-0002 223120
103-1240-0003 235360
103-1240-0004 200240
```
If you would like to extract acoustic features from the waveforms in advance, 
please prepare _idx2feat_len_ to notify the toolkit the length of your feature files. 
The length here means the number of time frames in your acoustic features (e.g. log-mel spectrogram or MFCC).

#### idx2sent
In _idx2sent_, each waveform file corresponds to a row and each row should be in the form of _file name + blank + sentence_. 
For example, 
```
103-1240-0000 CHAPTER ONE MISSUS RACHEL LYNDE IS SURPRISED MISSUS RACHEL LYNDE LIVED JUST WHERE THE AVONLEA MAIN ROAD DIPPED DOWN INTO A LITTLE HOLLOW FRINGED WITH ALDERS AND LADIES EARDROPS AND TRAVERSED BY A BROOK
103-1240-0001 THAT HAD ITS SOURCE AWAY BACK IN THE WOODS OF THE OLD CUTHBERT PLACE IT WAS REPUTED TO BE AN INTRICATE HEADLONG BROOK IN ITS EARLIER COURSE THROUGH THOSE WOODS WITH DARK SECRETS OF POOL AND CASCADE BUT BY THE TIME IT REACHED LYNDE'S HOLLOW IT WAS A QUIET WELL CONDUCTED LITTLE STREAM
103-1240-0002 FOR NOT EVEN A BROOK COULD RUN PAST MISSUS RACHEL LYNDE'S DOOR WITHOUT DUE REGARD FOR DECENCY AND DECORUM IT PROBABLY WAS CONSCIOUS THAT MISSUS RACHEL WAS SITTING AT HER WINDOW KEEPING A SHARP EYE ON EVERYTHING THAT PASSED FROM BROOKS AND CHILDREN UP
103-1240-0003 AND THAT IF SHE NOTICED ANYTHING ODD OR OUT OF PLACE SHE WOULD NEVER REST UNTIL SHE HAD FERRETED OUT THE WHYS AND WHEREFORES THEREOF THERE ARE PLENTY OF PEOPLE IN AVONLEA AND OUT OF IT WHO CAN ATTEND CLOSELY TO THEIR NEIGHBOR'S BUSINESS BY DINT OF NEGLECTING THEIR OWN
103-1240-0004 BUT MISSUS RACHEL LYNDE WAS ONE OF THOSE CAPABLE CREATURES WHO CAN MANAGE THEIR OWN CONCERNS AND THOSE OF OTHER FOLKS INTO THE BARGAIN SHE WAS A NOTABLE HOUSEWIFE HER WORK WAS ALWAYS DONE AND WELL DONE SHE RAN THE SEWING CIRCLE
```
**Note** that you don't need to worry about the blanks inside each transcript. 
Those additional blanks will be automatically solved by our toolkit.

#### text
In _text_, each waveform file corresponds to a row and each row only has a sentence. 
This file is used for vocabulary list generation.

For example, 
```
CHAPTER ONE MISSUS RACHEL LYNDE IS SURPRISED MISSUS RACHEL LYNDE LIVED JUST WHERE THE AVONLEA MAIN ROAD DIPPED DOWN INTO A LITTLE HOLLOW FRINGED WITH ALDERS AND LADIES EARDROPS AND TRAVERSED BY A BROOK
THAT HAD ITS SOURCE AWAY BACK IN THE WOODS OF THE OLD CUTHBERT PLACE IT WAS REPUTED TO BE AN INTRICATE HEADLONG BROOK IN ITS EARLIER COURSE THROUGH THOSE WOODS WITH DARK SECRETS OF POOL AND CASCADE BUT BY THE TIME IT REACHED LYNDE'S HOLLOW IT WAS A QUIET WELL CONDUCTED LITTLE STREAM
FOR NOT EVEN A BROOK COULD RUN PAST MISSUS RACHEL LYNDE'S DOOR WITHOUT DUE REGARD FOR DECENCY AND DECORUM IT PROBABLY WAS CONSCIOUS THAT MISSUS RACHEL WAS SITTING AT HER WINDOW KEEPING A SHARP EYE ON EVERYTHING THAT PASSED FROM BROOKS AND CHILDREN UP
AND THAT IF SHE NOTICED ANYTHING ODD OR OUT OF PLACE SHE WOULD NEVER REST UNTIL SHE HAD FERRETED OUT THE WHYS AND WHEREFORES THEREOF THERE ARE PLENTY OF PEOPLE IN AVONLEA AND OUT OF IT WHO CAN ATTEND CLOSELY TO THEIR NEIGHBOR'S BUSINESS BY DINT OF NEGLECTING THEIR OWN
BUT MISSUS RACHEL LYNDE WAS ONE OF THOSE CAPABLE CREATURES WHO CAN MANAGE THEIR OWN CONCERNS AND THOSE OF OTHER FOLKS INTO THE BARGAIN SHE WAS A NOTABLE HOUSEWIFE HER WORK WAS ALWAYS DONE AND WELL DONE SHE RAN THE SEWING CIRCLE
```

### Meta data
There may be some meta data provided by your target dataset that contains additional information about the waveform files.
For metadata files, we recommend you name them in the form of _idx2xxx_. 
_xxx_ indicates a type of metadata, such as _idx2spk_ (speaker ID) and _idx2gen_ (gender).
In each metadata file, each audio file corresponds to a row and each row should be in the form of _file name + blank + meta data_. 

**Note** that the codes of collecting the metadata information and generating those files should be designed manually for each dataset in _stat_info_generator.py_. 

#### idx2spk
In _idx2spk_, each waveform file corresponds to a row and each row should be in the form of _file name + blank + speaker ID_. 

For example, 
```
103-1240-0000 103
103-1240-0001 103
103-1240-0002 103
103-1240-0003 103
103-1240-0004 103
```

#### idx2gen
In _idx2gen_, each waveform file corresponds to a row and each row should be in the form of _file name + blank + gender_. 

For example, 
```
103-1240-0000 F
103-1240-0001 F
103-1240-0002 F
103-1240-0003 F
103-1240-0004 F
```

## The Procedure of Dumping a Dataset on your Machine
There are 6 steps to dump your target dataset from the internet to your local machine:
1. **Data downloading**: 
    In this step, the original data of the target dataset is downloaded from the internet to _/datasets/speech_text/dataset_name/wav_ by **_data_download.sh_**. 

2. **Data preparation**: 
    In this step, the statistical information of the target dataset is extracted from _/datasets/speech_text/dataset_name/wav_ by **_stat_info_generator.py_**. 

    The statistical files will be saved to _/datasets/speech_text/dataset_name/wav/(subset_name/)_ with the names of _idx2wav_, _idx2sent_, _text_, and perhaps _idx2spk_ and _idx2gen_.

3. **Waveform downsampling (optional)**:
    In this step, the original waveform files are downsampled to your target frequency. 
    This step is not mandatory if the frequency of the original files is OK for your research.
    
    The downsampled waveform files will be save to _/datasets/speech_text/dataset_name/wavXXXXX_ where _XXXXX_ is your target frequency.

4. **Acoustic feature extraction (optional)**:
    In this step, the acoustic feature vectors are extracted by your given configuration.
    This step is not mandatory since our toolkit supports on-the-fly acoustic feature extraction during training.
    
    The extracted acoustic features will be save to _/datasets/speech_text/dataset_name/XXXX_ where _XXXX_ is the name of your given configuration.

5. **Data length generation**:
    In this step, the length of your specified source data will be extracted. 
    If the source data is waveform, the number of sampling points of each waveform file will be recorded.
    If the source data is acoustic feature vectors, the number of time frames of each feature file will be recorded.
    
    The extracted data length will be saved to _/datasets/speech_text/dataset_name/wav/idx2wav_len_ or _/datasets/speech_text/dataset_name/XXXX/idx2feat_len_.

6. **Vocabulary generation**:
    In this step, the vocabulary list and sentence length will be extracted based on _/datasets/speech_text/dataset_name/wav/(subset_name/)text_ and _/datasets/speech_text/dataset_name/wav/(subset_name/)idx2sent_.
    
    The extracted vocabulary list will be saved to _/datasets/speech_text/dataset_name/token_type/(subset_name/)(token_config/)vocab_. 
    The extracted sentence length will be saved to _/datasets/speech_text/dataset_name/token_type/(subset_name/)(token_config/)(tgt_subset/)idx2sent_len_.
    
    **Note**: _token_config_ is only used for subword tokenization because subword has some hyperparameters to be tuned.

## How to Contribute a New Dataset
If the speech dataset that you want to use for your experiments is not included here, 
we would appreciate it a lot if you could make the implementation of your target dataset. 
If you want to make a contribution, please follow the instructions below: 
1. Make a new sub-folder in _./datasets/speech_text/_ with the folder name as the dataset name.
2. Copy the scripts **_data_dumping.sh_** and **_run.sh_** from _./datasets/speech_text/_ to the dataset sub-folder that you have just created.
3. Change the default values of some arguments indicated at the top of the script according to your machine and the target dataset.
4. Make a new ***data_download.sh*** to download your target dataset from the internet.
5. Make a new ***stat_info_generator.py*** to extract the statistic information of your target dataset. 
You could refer to the ones in the existing dataset sub-folders as the template.
6. Set the input arguments of **_data_dumping.sh_** in **_run.sh_** and run **_run.sh_** in the terminal by _./run.sh_. 

    **PS:** If you meet some errors about the line break '_$\r_', please open **_data_dumping.sh_** and **_run.sh_** by the command _vim_, type `:set ff=unix`, and save the files by `:wq`. 
    These errors are probably because I created them in an external IDE instead of the Linux panel.

**Note** that you should keep the same names *data_download.sh* and *stat_info_generator.py* for the compatibility with _data_dumping.sh_ and _run.sh_.