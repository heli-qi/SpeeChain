# Iterator

[*Iterator*](https://github.com/ahclab/SpeeChain/blob/main/speechain/iterator/abs.py) is the base class that takes charge of grouping data instances into batches for training or testing models.
Each iterator object has a built-in _speechain.dataset.Dataset_ object as a member variable. 
Actually, an _Iterator_ object cannot directly access the data instances in the built-in _Dataset_ object but maintains a batching view of the indices of the data instances used for model training or testing.

The iterators are divided into 3 groups: *train*, *valid*, and *test*. 
In each group, 2 or more iterator objects can be constructed so that there could be multiple data-label pairs in a single batch.

👆[Back to the handbook page](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#speechain-handbook)

## Table of Contents
1. [**Configuration File Format**](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#configuration-file-format)
2. [**Iterator Library**](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#iterator-library)
3. [**API Document**](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#api-document)
4. [**How to Construct Multiple Dataloaders for a Batch**](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#how-to-construct-multiple-dataloaders)

## Configuration File Format
The configuration of *Iterator* is given in *data_cfg*. 
The configuration format is shown below.
```
train:
    {iterator_name}:
        type: {file_name}.{class_name}
        conf:
            # Built-in Dataset Configuration
            dataset_type: {file_name}.{class_name}
            dataset_conf:
                ...
            # General Iterator Configuration
            batches_per_epoch:
            data_selection:
            is_descending:
            shuffle:
            data_len:
            group_info:
            # Customized Iterator Configuration
            ...
    ...
valid:
    {iterator_name}:
        type: {file_name}.{class_name}
        conf:
            dataset_type: {file_name}.{class_name}
            dataset_conf:
                ...
            ...
    ...
test:
    {test_set_name}:
        {iterator_name}:
            type: {file_name}.{class_name}
            conf:
                dataset_type: {file_name}.{class_name}
                dataset_conf:
                    ...
                ...
        ...
```
* The first-level keys must be one of ***train***, ***valid***, and ***test***. 
The combination of your first-level keys must be one of **train & valid & test** (for training and testing), **train & valid** (for training only), or **test** (for testing only).
   * The second-level keys are iterator names used for distinguishing the loaded data of each iterator. 
   There is no restriction on the iterator names, so you can name them in your own preference.  
   Under the name of each iterator, there are two third-level keys whose names are fixed:
     1. **type:**  
     The value of this key acts as the query string to pick up your target *Iterator* subclass in `SPEECHAIN_ROOT/speechain/iterator/`. 
     Your given query should be in the form of `{file_name}.{class_name}` where `file_name` specifies your target _.py_ file in `SPEECHAIN_ROOT/speechain/iterator/` and `class_name` indicates your target _Iterator_ subclass in `SPEECHAIN_ROOT/speechain/iterator/{file_name}.py`.   
     For example, `block.BlockIterator` means the subclass `BlockIterator` in `SPEECHAIN_ROOT/speechain/iterator/block.py`.
     2. **conf:**  
     The value of this key indicates the configuration of your iterator. 
     The configuration is made up of the following 4 fourth-level keys:
         1. **dataset_type:**  
         The value of this key acts as the query string to pick up your target built-in _Dataset_ subclass in `SPEECHAIN_ROOT/speechain/dataset/`.
         Your given query should be in the form of `{file_name}.{class_name}` where `file_name` specifies your target _.py_ file in `SPEECHAIN_ROOT/speechain/dataset/`, and `class_name` indicates your target _Dataset_ subclass in `SPEECHAIN_ROOT/speechain/dataset/{file_name}.py`.   
         For example, `speech_text.SpeechTextDataset` means the subclass `SpeechTextDataset` in `./speechain/dataset/speech_text.py`.
         2. **dataset_conf:**  
         The value of this key contains all the configuration used to initialize the built-in *Dataset* object. 
         Please refer to [Dataset API Document](https://github.com/ahclab/SpeeChain/tree/main/speechain/dataset#api-document) for more details.
         3. **General Iterator Configuration:**  
         These configurations are used to initialize the general part shared by all iterator subclasses. 
         There are 6 general arguments that can be set manually in _data_cfg:_ (please refer to [speechain.iterator.abs.Iterator.\_\_init__](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#__init__self-dataset_type-dataset_conf-batches_per_epoch-data_len-group_info-data_selection-is_descending-shuffle-seed-ngpu-num_workers-pin_memory-distributed-iter_conf) for more details)  
            1. batches_per_epoch
            2. is_descending
            3. shuffle
            4. data_len
            5. group_info
         4. **Customized Iterator Configuration:**  
         The arguments of the customized configuration are used by each *Iterator* subclass to generate the batching view. 
         Please refer to your target *Iterator* subclass for more details.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#table-of-contents)

## Iterator Library
```
/speechain
    /iterator
        /abs.py         # Abstract class of Iterator. Base of all Iterator implementations.
        /block.py       # Iterator implementation of the block strategy (variable utterances per batch). Mainly used for ASR and TTS training.
        /piece.py       # Iterator implementation of the piece strategy (fixed utterances per batch). Mainly used for ASR and TTS evaluation.
```

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#table-of-contents)

## API Document
1. [**speechain.iterator.abs.Iterator**](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#speechainiteratorabsiterator)  
   _Non-overridable backbone functions:_
   1. [\_\_init\_\_](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#__init__self-dataset_type-dataset_conf-batches_per_epoch-data_len-group_info-data_selection-is_descending-shuffle-seed-ngpu-num_workers-pin_memory-distributed-iter_conf)
   2. [\_\_len\_\_](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#__len__self)
   3. [get_batch_indices](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#get_batch_indicesself)
   4. [get_group_info](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#get_group_infoself)
   5. [build_loader](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#build_loaderself-epoch-start_step)  
   
   _Overridable interface functions:_  
   1. [batches_generate_fn](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#batches_generate_fnself-data_index-data_len-batch_size)
   
2. [**speechain.iterator.block.BlockIterator**](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#speechainiteratorblockblockiterator)
   1. [batches_generate_fn](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#batches_generate_fnself-data_index-data_len-batch_len)

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#table-of-contents)

### speechain.iterator.abs.Iterator
The initialization of the built-in _Dataset_ object is done automatically during the initialization of the iterator.
At the beginning of each epoch, the iterator generates a `torch.utils.data.DataLoader` object to fetch the batches of data instances from the disk.  
Each iterator subclass should override a static hook function `batches_generate_fn()` to generate the batching view of data instances in the built-in _Dataset_ object based on their own data batching strategy.
#### \_\_init\_\_(self, dataset_type, dataset_conf, batches_per_epoch, data_len, group_info, data_selection, is_descending, shuffle, seed, ngpu, num_workers, pin_memory, distributed, **iter_conf)
* **Description:**  
    The general initialization function shared by all the _Iterator_ classes. 
    _Dataset_ initialization is automatically done here by the given _dataset_type_ and _dataset_conf_.
* **Arguments:**
  * **_dataset_type:_** str  
    Query string to pick up the target Dataset subclass in `SPEECHAIN_ROOT/speechain/dataset/`
  * **_dataset_conf:_**  Dict  
    Dataset configuration for the automatic initialization of the built-in _Dataset_ object.
  * **_batches_per_epoch:_** int = None  
    The number of batches in each epoch. This number can be either smaller or larger than the real batch number. 
    If not given (None), all batches will be used in each epoch.
  * **_data_len:_** str or List[str] = None  
    The path of the data length file. Multiple data length files can be given in a list, but they must contain non-overlapping data instances.
  * _**group_info:**_ Dict[str, str or List[str]]  
    The dictionary of paths for the _idx2data_ files used for group-wise evaluation results visualization.
  * **_is_descending:_** bool = True  
    Whether the batches are sorted in the descending order by the length (True) or in the ascending order (False). 
    This argument is effective only when _data_len_ is given.
  * **_shuffle:_** bool = True  
    Whether the batches are shuffled at the beginning of each epoch.
  * **_seed:_** int = 0  
    Random seed for iterator initialization. This argument is automatically given by the experiment environment configuration.  
    The seed will be used to
    1. shuffle batches before giving to the Dataloader of each epoch.
    2. initialize all the workers of the Dataloader for the reproducibility.
  * **_ngpu:_** int = 1  
    The number of GPUs used to train or test models. This argument is automatically given by the experiment environment configuration.  
    The GPU number is used to ensure that each GPU process in the DDP mode has the batches with the same number of data instances.
  * **_num_workers:_** int = 1  
    Number of workers for the Dataloader. This argument is automatically given by the experiment environment configuration.  
  * **_pin_memory:_** bool = False  
    Whether pin_memory is activated in the Dataloader. This argument is automatically given by the experiment environment configuration.  
  * **_distributed:_** bool = False  
    Whether DDP is used to distribute the model. This argument is automatically given by the experiment environment configuration.  
  * **_iter_conf:_** Dict  
    Iterator configuration for customized batch generation

👆[Back to the API list](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#api-document)

#### \_\_len\_\_(self)
* **Description:**  
    Get the number of batches the iterator will load in each epoch. 
* **Return:**  
    If _batches_per_epoch_ is given, its value will be returned; otherwise, the total number of all the batches in the built-in _Dataset_ object will be returned.

👆[Back to the API list](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#api-document)

#### get_batch_indices(self)
* **Description:**  
    This function return the current batching view of the iterator object.
* **Return:** List[List[str]]  
    The batching view generated by the customized hook interface `batches_generate_fn()`. 
    Each element of the returned batching view list is a sub-list of data indices where each index corresponds to a data instance in the built-in _Dataset_ object.

👆[Back to the API list](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#api-document)

#### get_group_info(self)
* **Description:**  
    This function returns the group information of the data instances in the built-in _Dataset_ object.
    The returned metadata is mainly used for group-wise testing results visualization.
* **Return:** Dict  
    If metadata information is not initialized in the built-in _Dataset_ object, None will be returned.
    Otherwise, the meta_info member of the built-in _Dataset_ object will be returned which is a dictionary.

👆[Back to the API list](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#api-document)

#### build_loader(self, epoch, start_step)
* **Description:**  
    This function generate a `torch.util.data.DataLoader` object to load the batches of data instances for the current epoch.  
    If `batches_per_epoch` is not given, all the batches in `self.batches` will be used to generate the Dataloader;
    If `batches_per_epoch` is given, a batch clip containing `batches_per_epoch` batches will be used to generate the Dataloader.  
    `batches_per_epoch` can be either larger or smaller than the total number of batches. 
    For a smaller `batches_per_epoch`, a part of `self.batches` will be used as the batch clip; 
    For a larger `batches_per_epoch`, `self.batches` will be supplemented by a part of itself to form the batch clip.
* **Arguments:**  
  * **_epoch:_** int = 1  
    The number of the current epoch. Used as part of the random seed to shuffle the batches.
  * **_start_step:_** int = 0  
    The start point for the dataloader of the current epoch. 
    Mainly used for resuming a model testing job from a checkpoint.
* **Return:** torch.util.data.DataLoader  
    A DataLoader built on the batch clip of the current epoch. 
    If `batches_per_epoch` is not given, the batch clip is `self.batches`.

👆[Back to the API list](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#api-document)

#### batches_generate_fn(self, data_index, data_len, batch_size)
* **Description:**  
    This hook function generates the batching view based on your customized batch generation strategy.  
    Your overridden function should return the batches of instance indices as a List[List[str]] where each sub-list corresponds to a batch of data instances. 
    Each element in the sub-list is the index of a data instance.  
    In this original hook implementation, all the data instances in the built-in _Dataset_ object will be grouped into batches with exactly the same amount of instances. 
    `data_len` is not used in this hook function but used for sorting all the instances in the general initialization function of the iterator. 
    The sorted data instances make sure that the instances in a single batch have similar lengths.
* **Arguments:**
  * _**data_index:**_ List[str]  
    The list of indices of all the data instances available to generate the batching view.
  * _**data_len:**_ Dict[str, int]  
    The dictionary that indicates the data length of each available data instance in data_index.
  * _**batch_size:**_ int = None  
    How many data instances does a batch should have. 
    If not given, it will be the number of GPUs (ngpu) to ensure that the model validation or testing is done one data instance at each step on a single GPU process.  
    **Note:** `batch_size` is implicitly given by `**iter_conf` in `__init__()` to this static hook function, so your implementation don't need to keep this argument, and you can declare your own argument.
* **Return:** List[List[str]]  
  A list of batches generated by your batching strategy. This List[List[str]] is called the batching view of the iterator object.
  Each batch in the returned list is a sub-list whose elements are the indices of data instances in the corresponding batch.

👆[Back to the API list](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#api-document)

### speechain.iterator.block.BlockIterator
The strategy of this iterator is to generate batches with the same amount of data lengths. 
For sequence-to-sequence tasks, the data instances are usually different in data length. 
If there is a fixed number of data instances in each batch, the data volume of a single batch may constantly change during training. 
This may either cause a CUDA memory error (out of GPU memory) or large idle GPU memories.

It can be considered as the strategy that always gives 'rectangles' with similar 'areas' if we treat the number of data instances in a batch as the rectangle length and the maximal data length as the rectangle width.
#### batches_generate_fn(self, data_index, data_len, batch_len)
* **Description:**  
    All the data instances in the built-in _Dataset_ object will be grouped into batches with the same total lengths.
    The lengths used for grouping is given in data_len. 
    The customized argument batch_len specifies the total length that each batch should have.
* **Arguments:**  
  * _**data_index**_
  * _**data_len**_
  * _**batch_len:**_ int = None  
    The total data length of all the data instances in a batch.
    If the data is in the format of audio waveforms, batch_len is the amount of sampling points.
    If the data is in the format of acoustic features, batch_len is the amount of time frames.

👆[Back to the API list](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#api-document)

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#table-of-contents)


## How to Construct Multiple Dataloaders

[//]: # (Multiple Dataloaders can be easily constructed by giving the configuration of multiple iterators. )

[//]: # (Each iterator creates an independent Dataloader that contributes a data-label pair in the batch. )

[//]: # ()
[//]: # (An example of semi-supervised ASR training is shown below. There are two iterators in the _train_ group: _sup_ and _unsup_ &#40;the iterator names are given by users based on their preferences&#41;. )

[//]: # (These two iterators are in the same type and have built-in datasets with the same type.)

[//]: # (```)

[//]: # (train:)

[//]: # (    sup:)

[//]: # (        type: block.BlockIterator)

[//]: # (        conf:)

[//]: # (            dataset_type: speech_text.SpeechTextDataset)

[//]: # (            dataset_conf:)

[//]: # (                ...)

[//]: # (            ...)

[//]: # (    unsup:)

[//]: # (        type: block.BlockIterator)

[//]: # (        conf:)

[//]: # (            dataset_type: speech_text.SpeechTextDataset)

[//]: # (            dataset_conf:)

[//]: # (                ...)

[//]: # (            ...)

[//]: # (```)

[//]: # (If there are multiple Dataloaders used to load data, each Dataloader will contribute a sub-Dict in the batch Dict _train_batch_ as shown below. )

[//]: # (The name of each sub-Dict is the one users give as the name of the corresponding iterator.)

[//]: # (```)

[//]: # (train_batch:)

[//]: # (    sup:)

[//]: # (        feat: torch.Tensor)

[//]: # (        feat_len: torch.Tensor)

[//]: # (        text: torch.Tensor)

[//]: # (        text_len: torch.Tensor)

[//]: # (    unsup:)

[//]: # (        feat: torch.Tensor)

[//]: # (        feat_len: torch.Tensor)

[//]: # (        text: torch.Tensor)

[//]: # (        text_len: torch.Tensor)

[//]: # (```)

[//]: # (If you have only one iterator like the configuration below, your _train_batch_ will not have any sub-Dict but only the data-label pair from that iterator. )

[//]: # (In this case, you don't need to give the name tag for the iterator.)

[//]: # (```)

[//]: # (train:)

[//]: # (    type: block.BlockIterator)

[//]: # (    conf:)

[//]: # (        dataset_type: speech.speech_text.SpeechTextDataset)

[//]: # (        dataset_conf:)

[//]: # (            ...)

[//]: # (        ...)

[//]: # (```)

[//]: # (```)

[//]: # (train_batch:)

[//]: # (    feat: torch.Tensor)

[//]: # (    feat_len: torch.Tensor)

[//]: # (    text: torch.Tensor)

[//]: # (    text_len: torch.Tensor)

[//]: # (```)

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/speechain/iterator#table-of-contents)
