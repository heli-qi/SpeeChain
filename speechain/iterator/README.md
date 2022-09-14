# Data Loading Part
Data loading is done by two classes: *Dataset* and *Iterator*.

[*Dataset*](https://github.com/ahclab/SpeeChain/blob/main/speechain/dataset/abs.py) loads the raw data from the disk and transforms them into model-friendly vectors. 
Data preprocessing is also done in this class before feeding data to the model.

[*Iterator*](https://github.com/ahclab/SpeeChain/blob/main/speechain/iterator/abs.py) defines the strategy of forming batches. 
It holds a built-in *Dataset* and generates a `torch.utils.data.DataLoader` on its dataset in each epoch.
The iterators are divided into 3 groups: *train*, *valid*, and *test*. 
In each group, more than 1 iterator can be constructed so that there could be multiple data pairs in a single batch.

👆[Back to the home page](https://github.com/ahclab/SpeeChain#the-speechain-toolkit)

## Table of Contents
1. [**Configuration File Format**](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#configuration-file-format)
2. [**Abstract Interfaces Description**](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#abstract-interfaces-description)
    1. [Dataset](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#dataset)
    2. [Iterator](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#iterator)
3. [**How to Construct Multiple Dataloaders**](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#how-to-construct-multiple-dataloaders)
4. [**How to Mix Multiple Datasets in a Single Dataloader**](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#how-to-mix-multiple-datasets-in-a-single-dataloader)
5. [**How to Perform Data Selection in a Single Dataloader**](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#how-to-perform-data-selection-in-a-single-dataloader)
6. [**How to introduce Meta Information to your batches**](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#how-to-introduce-meta-information-to-your-batches)

## Configuration File Format
The configuration of *Dataset* and *Iterator* is given in *data_cfg*. 
The configuration format is shown below. If you would like to see some examples, please go to the following sections.
```
train:
    iterator1_name:
        type: file_name.class_name
        conf:
            dataset_type: datatype_name.file_name.class_name
            dataset_conf:
                src_data: data_file_path
                tgt_label: label_file_path
                meta_info:
                    meta_name1: meta_file_path
                    meta_name2: meta_file_path
                    ...
                ...
            ...
    ...
valid:
    iterator2_name:
        type: file_name.class_name
        conf:
            dataset_type: datatype_name.file_name.class_name
            dataset_conf:
                src_data: data_file_path
                tgt_label: label_file_path
                ...
            ...
    ...
test:
    iterator3_name:
        type: file_name.class_name
        conf:
            dataset_type: datatype_name.file_name.class_name
            dataset_conf:
                src_data: data_file_path
                tgt_label: label_file_path
                ...
            ...
    ...
```

1. The first-level keys must be one of ***train***, ***valid***, and ***test***. 
The combination of your first-level keys must be one of 
    1. *train, valid, test* (for training and testing)
    2. *train, valid* (for training only)
    3. *test* (for testing only).
2. **iterator_name** is the second-level key used for distinguishing the loaded data of your iterators. 
If you have more than 1 iterator, these second-level keys will be used as the names of loaded data to make them identifiable with each other. 
There is no restriction on the iterator names, so you can create them in the way you would like your iterators to be named.
3. **type** is the third-level key that indicates the type of the corresponding iterator. 
The value of this key acts as the query to pick up the target *Iterator* class in this toolkit. 
Your given query should be in the form of `file_name.class_name` to indicate the place and name of your target class. 
For example, `block.BlockIterator` means the class `BlockIterator` in `./speechain/iterator/block.py`.
4. **conf** is the third-level key that indicates your iterator configuration. The contents have the following 4 parts:
    1. **dataset_type** indicates the type of the built-in *Dataset* in this iterator. 
    This value will be used as the query to pick up the target *Dataset* class. 
    Your given query should be in the form of `datatype_name.file_name.class_name` to indicate the place of your target class. 
    For example, `speech.speech_text.SpeechTextDataset` means the class `SpeechTextDataset` in `./speechain/dataset/speech/speech_text.py`.
    2. **dataset_conf** contains all the configuration used to initialize the built-in *Dataset*. 
    The _src_data_, _tgt_label_, _meta_info_ arguments indicate the data samples used in this *Dataset*.
    3. **Iterator general configuration**. These configurations are used to initialize the general part shared by all types of iterators in this toolkit. 
    Please refer to the docstrings of [*Iterator*](https://github.com/ahclab/SpeeChain/blob/main/speechain/iterator/abs.py) for more details.
    4. **Iterator customized configuration**. This part is used to initialize the customized part of your chosen iterator. 
    Please refer to the docstrings of your target *Iterator* class for more details.

## Abstract Interfaces Description
### Dataset
1. **dataset_init()**: 
The initialization function of the customized part of your dataset implementation. 
2. **read_data_file()**:
This function reads the contents of your given data files into memory. 
These files contain the source data of the training samples.
3. **read_label_file()**:
This function reads the contents of your given label files into memory. 
These files contain the target labels of the training samples.
4. **read_meta_file()**:
This function reads the contents of your given data files into memory. 
These files contain the meta information of the training samples. 
This interface is not mandatory to be overridden unless you would like to introduce meta information in your batches.
5. **\__getitem\__()**: 
This function should return the corresponding data sample in the dataset by the given index.
6. **collate_fn()**:
This function does some preprocessing operations to the current batch of data samples, 
such as length mismatch unification, data precision adjustment, and so on.

For more details, please refer to [*Dataset*](https://github.com/ahclab/SpeeChain/blob/main/speechain/dataset/abs.py).

### Iterator
1. **iter_init()**: 
The initialization function of the customized part of your iterator. 
Your implementation of this interface should return the list of batches generated by your batching strategy.

For more details, please refer to [*Iterator*](https://github.com/ahclab/SpeeChain/blob/main/speechain/iterator/abs.py).

## How to Construct Multiple Dataloaders
Multiple Dataloaders can be easily constructed by giving the configuration of multiple iterators. 
Each iterator creates an independent Dataloader that contributes a data-label pair in the batch. 

An example of semi-supervised ASR training is shown below. There are two iterators in the _train_ group: _sup_ and _unsup_ (the iterator names are given by users based on their preferences). 
These two iterators are in the same type and have built-in datasets with the same type.
```
train:
    sup:
        type: block.BlockIterator
        conf:
            dataset_type: speech.speech_text.SpeechTextDataset
            dataset_conf:
                ...
            ...
    unsup:
        type: block.BlockIterator
        conf:
            dataset_type: speech.speech_text.SpeechTextDataset
            dataset_conf:
                ...
            ...
```
If there are multiple Dataloaders used to load data, each Dataloader will contribute a sub-Dict in the batch Dict _train_batch_ as shown below. 
The name of each sub-Dict is the one users give as the name of the corresponding iterator.
```
train_batch:
    sup:
        feat: torch.Tensor
        feat_len: torch.Tensor
        text: torch.Tensor
        text_len: torch.Tensor
    unsup:
        feat: torch.Tensor
        feat_len: torch.Tensor
        text: torch.Tensor
        text_len: torch.Tensor
```
If you have only one iterator like the configuration below, your _train_batch_ will not have any sub-Dict but only the data-label pair from that iterator. 
In this case, you don't need to give the name tag for the iterator.
```
train:
    type: block.BlockIterator
    conf:
        dataset_type: speech.speech_text.SpeechTextDataset
        dataset_conf:
            ...
        ...
```
```
train_batch:
    feat: torch.Tensor
    feat_len: torch.Tensor
    text: torch.Tensor
    text_len: torch.Tensor
```


## How to Mix Multiple Datasets in a Single Dataloader
If you want to initialize your iterator with multiple datasets and want your dataloader to pick up batches from the mixed dataset, 
you can simply give a list of file paths to the _src_data_ and _tgt_label_ arguments to initialize the built-in dataset of your iterator like the example below.
```
data_root: ./datasets/speech/librispeech/data/wav
train:
    type: block.BlockIterator
    conf:
        dataset_type: speech.speech_text.SpeechTextDataset
        dataset_conf:
            src_data:
                - !ref <data_root>/train_clean_100/feat.scp
                - !ref <data_root>/train_clean_360/feat.scp
                - !ref <data_root>/train_other_500/feat.scp
            tgt_label:
                - !ref <data_root>/train_clean_100/text
                - !ref <data_root>/train_clean_100/text
                - !ref <data_root>/train_other_500/text
        ...
```

## How to Perform Data Selection in a Single Dataloader
If you only need to load a part of the data samples from the built-in dataset, 
you can use the arguments _selection_mode_ and _selection_num_. 
_selection_mode_ specifies the selection method and _selection_num_ specifies the number of selected samples. 
_selection_num_ can be given as a positive float number or a negative integer number. 
The positive float number means the ratio of the dataset. In the example below, the first 50% of *LibriSpeech-train_clean_100* will be selected. 
```
data_root: ./datasets/speech/librispeech/data/wav
train:
    type: block.BlockIterator
    conf:
        dataset_type: speech.speech_text.SpeechTextDataset
        dataset_conf:
            src_data: !ref <data_root>/train_clean_100/feat.scp
            tgt_label: !ref <data_root>/train_clean_100/text

        selection_mode: order
        selection_num: 0.5
        ...
```
The negative integer number means the absolute number of the selected samples. 
In the example below, 1000 data samples of *LibriSpeech-train_clean_100* will be randomly selected. 
```
data_root: ./datasets/speech/librispeech/data/wav
train:
    type: block.BlockIterator
    conf:
        dataset_type: speech.speech_text.SpeechTextDataset
        dataset_conf:
            src_data: !ref <data_root>/train_clean_100/feat.scp
            tgt_label: !ref <data_root>/train_clean_100/text

        selection_mode: random
        selection_num: -1000
        ...
```
Moreover, data selection and datasets mixing can be used in a single iterator but they will be done sequentially. 
In the example below, _train_clean_100_, _train_clean_360_, and _train_other_500_ datasets of the _LibriSpeech_ corpus will be first mixed into a large dataset, and then the last 50% of the large dataset will be selected.
```
data_root: ./datasets/speech/librispeech/data/wav
train:
    type: block.BlockIterator
    conf:
        dataset_type: speech.speech_text.SpeechTextDataset
        dataset_conf:
            src_data:
                - !ref <data_root>/train_clean_100/feat.scp
                - !ref <data_root>/train_clean_360/feat.scp
                - !ref <data_root>/train_other_500/feat.scp
            tgt_label:
                - !ref <data_root>/train_clean_100/text
                - !ref <data_root>/train_clean_360/text
                - !ref <data_root>/train_other_500/text
        
        selection_mode: rev_order
        selection_num: 0.5
        ...
```

## How to introduce Meta Information to your batches
Our toolkit enables our users to include the information beyond the source data and target label of a training sample into their batches. 
This information is called meta information that describes various characteristic of the source data. 
For example, for an utterance, besides the transcript, we may also need to know its speaker information (speaker ID or speaker gender). 

The introduction of the meta information is very simple. First, please override the interface _read_meta_file()_ if you are making a new _speechain.abs.Dataset_ class. 
Then, giving the data loading configuration as follows:
```
data_root: ./datasets/speech/librispeech/data/wav
train:
    type: block.BlockIterator
    conf:
        dataset_type: speech.speech_text.SpeechTextDataset
        dataset_conf:
            src_data: !ref <data_root>/train_clean_100/feat.scp
            tgt_label: !ref <data_root>/train_clean_100/text
            meta_info:
                speaker: !ref <data_root>/train_clean_100/utt2spk
                gender: !ref <data_root>/train_clean_100/utt2gen
        
        selection_mode: rev_order
        selection_num: 0.5
        ...
```
Different from _src_data_ and _tgt_label_, _meta_info_ is given in the form of a _Dict_. 
So, multiple types of meta information can be included and each one corresponds to a key-value pair in this _Dict_. 
The tag names (e.g. gender, speaker) will act as the keys in the _meta_info_ Dict.