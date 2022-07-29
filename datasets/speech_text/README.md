# Speech-Text Datasets
This folder contains all the available datasets that are made up of speech and text data. 
Each dataset corresponds to a sub-folder and has a uniform file system. 
You can easily dump your target dataset to your machine by following the instructions. 

Our file system is modified from the ESPNET style. 
If you want to contribute a new dataset, we would appreciate it if you could follow our data formats and file systems.

👆[Back to the dataset home page]()

## Table of Contents
1. [**File System**]()
2. [**Data Format**]()
    1. [feat.scp]()
    2. [feat_len.scp]()
    3. [text]()
    4. [metadata]()
3. [**How to Contribute a New Dataset**]()

## File System
```
/speech_text
    /dataset_name               # each dataset has a sub-folder
        /data                       # all the data of the dataset
            /raw                        # the unprocessed raw data of the dataset
                /subset_name                # the name of a subset in this dataset (if have)
                    feat.scp                    # the file that contains the index and audio file path (absolute)
                    feat_len.scp                # the file that contains the index and data length
                    text                        # the file taht contains the index and transcripts
                    ...                         # some metadata files avaiable for this dataset (such as speaker ID, genders, ...)
            /feat_name                  # the sub-folder of a type of feature extracted from the raw data
                ...                         # the same structure as '/raw'
            /token_type                 # the sub-folder of a type of token
                dict                        # the token dictionary
        data_download.sh            # the .sh script that downloads the dataset from the internet
        stat_info_generator.py      # the .py script that generates the statistic information of the dataset (feat.scp, text, metadata, ...)
        data_dumping.sh             # the .sh script that contains the complete pipeline of data dumping (i.e. data preparation)
        run.sh                      # the .sh script that controls how the data dumping is going
```
**dataset_name** is the first-level sub-folder that indicates where the data in this folder comes from.
1. **data** is the second-level sub-folder that contains all the data from the dataset. 
The folder name _data_ is shared by all datasets. 
    1. **raw** is the third-level sub-folder that contains the unprocessed raw data of the dataset. 
    The name _raw_ is shared by all datasets.
        1. **subset_name** is the fourth-level sub-folder that indicates a subset of the dataset. 
        Some datasets are divided into independent subsets (e.g. LibriSpeech).
            1. **feat.scp** contains the mapping between audio file names (used as the index) and their paths on your machine.
            2. **feat_len.scp** contains the mapping between audio file names (used as the index) and their data lengths.
            3. **text** contains the mapping between audio file names (used as the index) and their corresponding transcripts.
    2. **feat_name** is the third-level sub-folder that contains the acoustic feature extracted from the raw data. 
    It should have the same file structure as _raw_ for compatibility.
    3. **token_type** is the third-level sub-folder that corresponds to a specific type of token.
        1. **dict** indicates the token dictionary extracted from the text data in the dataset. 
        The name _dict_ is shared by all the token sub-folders.
2. **data_download.sh** is the script used to download the dataset from the internet. 
Basically, this file is made up of _wget_ and _tar_ commands.
3. **stat_info_generator.py** is the script used to generate the statistical information of the dataset. 
The output files need to cover at least _feat.scp_ and _text_. 
If you want to generate some metadata files, please give the processing logic in this file.
4. **data_dumping.sh** is the script that contains the entire pipeline of data dumping. 
There are several steps in the dumping pipeline and each one can be done in a dataset-independent way if you follow our data format and file system. 
5. **run.sh** is the script that controls the starting and ending steps of the dumping process. 
It also controls the input arguments for dumping the dataset. 

For more details, please refer to */datasets/speech_text/librispeech* as an example.

## Data Format
The data format of all statistic files should be uniform for all datasets. 
The format uniformity enables the automatic configuration for the data loading part of our toolkit. 
The format for each file is shown below.
### feat.scp
In _feat.scp_, each audio file corresponds to a row and each row should be in the form of _file_name + blank + file_path_. 

For example, 
```
103-1240-0000 ./datasets/speech/librispeech/data/raw/train_clean_100/103/1240/103-1240-0000.flac
103-1240-0001 ./datasets/speech/librispeech/data/raw/train_clean_100/103/1240/103-1240-0001.flac
103-1240-0002 ./datasets/speech/librispeech/data/raw/train_clean_100/103/1240/103-1240-0002.flac
103-1240-0003 ./datasets/speech/librispeech/data/raw/train_clean_100/103/1240/103-1240-0003.flac
103-1240-0004 ./datasets/speech/librispeech/data/raw/train_clean_100/103/1240/103-1240-0004.flac
```
**Note** that the paths are better to be the absolute path of the audio file. 
The relative paths here are just for demonstration.

### feat_len.scp
In _feat_len.scp_, each audio file corresponds to a row and each row should be in the form of _file_name + blank + length_. 

For example, 
```
103-1240-0000 225360
103-1240-0001 255120
103-1240-0002 223120
103-1240-0003 235360
103-1240-0004 200240
```
If the file contains a audio waveform, the length is the number of sampling points; 
if the file contains a acoustic feature (e.g. log-mel spectrogram or MFCC), the length is the number of time frames.

### text
In _text_, each audio file corresponds to a row and each row should be in the form of _file_name + blank + transcript_. 
For example, 
```
103-1240-0000 CHAPTER ONE MISSUS RACHEL LYNDE IS SURPRISED MISSUS RACHEL LYNDE LIVED JUST WHERE THE AVONLEA MAIN ROAD DIPPED DOWN INTO A LITTLE HOLLOW FRINGED WITH ALDERS AND LADIES EARDROPS AND TRAVERSED BY A BROOK
103-1240-0001 THAT HAD ITS SOURCE AWAY BACK IN THE WOODS OF THE OLD CUTHBERT PLACE IT WAS REPUTED TO BE AN INTRICATE HEADLONG BROOK IN ITS EARLIER COURSE THROUGH THOSE WOODS WITH DARK SECRETS OF POOL AND CASCADE BUT BY THE TIME IT REACHED LYNDE'S HOLLOW IT WAS A QUIET WELL CONDUCTED LITTLE STREAM
103-1240-0002 FOR NOT EVEN A BROOK COULD RUN PAST MISSUS RACHEL LYNDE'S DOOR WITHOUT DUE REGARD FOR DECENCY AND DECORUM IT PROBABLY WAS CONSCIOUS THAT MISSUS RACHEL WAS SITTING AT HER WINDOW KEEPING A SHARP EYE ON EVERYTHING THAT PASSED FROM BROOKS AND CHILDREN UP
103-1240-0003 AND THAT IF SHE NOTICED ANYTHING ODD OR OUT OF PLACE SHE WOULD NEVER REST UNTIL SHE HAD FERRETED OUT THE WHYS AND WHEREFORES THEREOF THERE ARE PLENTY OF PEOPLE IN AVONLEA AND OUT OF IT WHO CAN ATTEND CLOSELY TO THEIR NEIGHBOR'S BUSINESS BY DINT OF NEGLECTING THEIR OWN
103-1240-0004 BUT MISSUS RACHEL LYNDE WAS ONE OF THOSE CAPABLE CREATURES WHO CAN MANAGE THEIR OWN CONCERNS AND THOSE OF OTHER FOLKS INTO THE BARGAIN SHE WAS A NOTABLE HOUSEWIFE HER WORK WAS ALWAYS DONE AND WELL DONE SHE RAN THE SEWING CIRCLE
```
**Note** that you don't need to worry about the blanks inside each transcript. Those additional blanks will be solved by our framework codes.

### metadata
For metadata files, we recommend you name them in the form of _utt2xxx_. 
_xxx_ indicates a type of metadata, such as _utt2spk_ (speaker ID) and _utt2gen_ (gender).
In each metadata file, each audio file corresponds to a row and each row should be in the form of _file_name + blank + meta_. 

For example, in _utt2spk_
```
103-1240-0000 103
103-1240-0001 103
103-1240-0002 103
103-1240-0003 103
103-1240-0004 103
```

In _utt2gen_
```
103-1240-0000 F
103-1240-0001 F
103-1240-0002 F
103-1240-0003 F
103-1240-0004 F
```
**Note** that the codes of collecting the metadata information and generating those files should be designed manually for each dataset in _stat_info_generator.py_. 


## How to Contribute a New Dataset
If the speech dataset that you want to use for your experiments is not included here, 
we would appreciate it a lot if you could make the implementation of your target dataset. 
If you want to make a contribution, please follow the instructions below: 
1. Make a new sub-folder in _./datasets/speech_text/_ with the folder name as the dataset name.
2. Copy the scripts **_data_dumping.sh_** and **_run.sh_** from an existing dataset sub-folder to the dataset sub-folder that you have just created.
3. Change the default values of some arguments indicated at the top of the script according to your machine and the target dataset.
4. Make a new ***data_download.sh*** to download your target dataset from the internet.
5. Make a new ***stat_info_generator.py*** to extract the statistic information of your target dataset. 
You could refer to the ones in the existing dataset sub-folders as the template.
6. Set the input arguments of **_data_dumping.sh_** in **_run.sh_** and run **_run.sh_** in the terminal by _./run.sh_. 

    **PS:** If you meet some errors about the line break '_$\r_', please open **_data_dumping.sh_** and **_run.sh_** by the command _vim_, type `:set ff=unix`, and save the files by `:wq`. 
    These errors are probably because I created them in an external IDEA instead of the Linux panel.

**Note** that you should keep the same names *data_download.sh* and *stat_info_generator.py* for the compatibility with _data_dumping.sh_ and _run.sh_.