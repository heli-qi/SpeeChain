# Automatic Speech Recognition (ASR)

👆[Back to the recipe README.md](https://github.com/ahclab/SpeeChain/tree/main/recipes#recipes-folder-of-the-speechain-toolkit)

## Table of Contents
1. [Available Backbones](https://github.com/ahclab/SpeeChain/tree/main/recipes/asr#available-backbones)
2. [Pretrained Model for Reproducibility](https://github.com/ahclab/SpeeChain/tree/main/recipes/asr#pretrained-model-for-reproducibility)
3. [How to train an ASR model](https://github.com/ahclab/SpeeChain/tree/main/recipes/asr#how-to-train-an-asr-model)
4. [How to create your own ASR model](https://github.com/ahclab/SpeeChain/tree/main/recipes/asr#how-to-create-your-own-asr-model)

## Available Backbones
1. Speech-Transformer ([paper reference](https://ieeexplore.ieee.org/abstract/document/8462506))

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/recipes/asr#table-of-contents)


## Pretrained Model for Reproducibility
For your reproducibility of our ASR models in `${SPEECHAIN_ROOT}/recipes/asr/`, we provide the following pretrained models to ensure that you will get the similar performance:
1. BPE tokenizer models in `${SPEECHAIN_ROOT}/datasets/librispeech/data/sentencepiece`.
2. Transformer-based language models in `${SPEECHAIN_ROOT}/recipes/lm/librispeech`.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/recipes/asr#table-of-contents)


## How to train an ASR model
### Use a well-tuned configuration on an existing dataset
1. **Find** a _.yaml_ configuration file in `${SPEECHAIN_ROOT}/recipes/asr`. 
   Suppose that we want to train an ASR model by the configuration `${SPEECHAIN_ROOT}/recipes/asr/librispeech/train-960/exp_cfg/960-bpe5k_transformer-wide_ctc_perturb.yaml`.
2. **Train and evaluate** the ASR model on your target training set
   ```
   cd ${SPEECHAIN_ROOT}/recipes/asr/librispeech/train-960
   bash run.sh --exp_cfg 960-bpe5k_transformer-wide_ctc_perturb (--ngpu x --gpus x,x)
   ```
   **Note:** 
   1. Please take a look at the comments on the top of the configuration file to make sure that your computational equipments fit the configuration before training the model.
      If your equipments don't match the configuration, please adjust it by `--ngpu` and `--gpus` to make sure that you have the same amount of GPU memories.
   2. If you want to save the experimental results outside the toolkit folder `${SPEECHAIN_ROOT}`, 
      please specify where you want to save the results by attaching `--train_result_path {your-target-path}` to `bash run.sh`.  
      In this example, if you give `bash run.sh --test false --exp_cfg 960-bpe5k_transformer-wide_ctc_perturb --train_result_path /a/b/c`, 
      the results will be saved to `/a/b/c/960-bpe5k_transformer-wide_ctc_perturb`.

### Make a new configuration on a non-existing dataset
1. **Dump** your target dataset from the Internet to `${SPEECHAIN_ROOT}/datasets` by the [instructions](https://github.com/ahclab/SpeeChain/tree/main/datasets#how-to-contribute-a-new-dataset).
2. **Make** the corresponding folder of your dumped dataset `${SPEECHAIN_ROOT}/recipes/asr/{your-new-dataset}/{your-target-subset}`.
   ```
   mkdir ${SPEECHAIN_ROOT}/recipes/asr/{your-new-dataset}/{your-target-subset}
   cp ${SPEECHAIN_ROOT}/recipes/asr/librispeech/train-960/run.sh ${SPEECHAIN_ROOT}/recipes/asr/{your-new-dataset}/{your-target-subset}/run.sh
   ```
   **Note:** 
   1. Please don't forget to change the arguments `dataset` and `subset` (line no.16 & 17) in `${SPEECHAIN_ROOT}/recipes/asr/{your-new-dataset}/{your-target-subset}/run.sh`:
      ```
      dataset=librispeech -> 'your-new-dataset'
      subset='train-960' -> 'your-new-dataset'
      ```
3. **Select and copy** a tuned configuration file into your newly-created folder. 
   Suppose that we want to use the configuration `${SPEECHAIN_ROOT}/recipes/asr/librispeech/train-960/exp_cfg/960-bpe5k_transformer-wide_ctc_perturb.yaml`.
   ```
   cd ${SPEECHAIN_ROOT}/recipes/asr/{your-new-dataset}/{your-target-subset}
   mkdir ./data_cfg ./exp_cfg
   cp ${SPEECHAIN_ROOT}/recipes/asr/librispeech/train-960/exp_cfg/960-bpe5k_transformer-wide_ctc_perturb.yaml ./exp_cfg
   ```
   **Note:** 
   1. Please change the dataset argument at the beginning of your selected configuration:
      ```
      # dataset-related
      dataset: librispeech -> 'your-new-dataset'
      train_set: train-960 -> 'your-target-subset'
      valid_set: dev -> 'valid-set-of-new-dataset'
    
      # tokenizer-related
      txt_format: asr
      vocab_set: train-960 -> 'your-target-subset'
      token_type: sentencepiece
      token_num: bpe5k
      ```
4. **Train** the ASR model on your target training set
   ```
   cd ${SPEECHAIN_ROOT}/recipes/asr/{your-new-dataset}/{your-target-subset}
   bash run.sh --test false --exp_cfg 960-bpe5k_transformer-wide_ctc_perturb (--ngpu x --gpus x,x)
   ```
   **Note:** 
   1. The argument `--test false` is used to skip the testing stage.
   2. Please take a look at the comments on the top of the configuration file to make sure that your computational equipments fit the configuration before training the model.
      If your equipments don't match the configuration, please adjust it by `--ngpu` and `--gpus` to make sure that you have the same amount of GPU memories.
   3. If you want to save the experimental results outside the toolkit folder `${SPEECHAIN_ROOT}`, 
      please specify where you want to save the results by attaching `--train_result_path {your-target-path}` to `bash run.sh`.  
      In this example, if you give `bash run.sh --test false --exp_cfg 960-bpe5k_transformer-wide_ctc_perturb --train_result_path /a/b/c`, 
      the results will be saved to `/a/b/c/960-bpe5k_transformer-wide_ctc_perturb`.
5. **Tune the inference hyperparameters** on the corresponding validation set
   ```
   cp ${SPEECHAIN_ROOT}/recipes/asr/librispeech/train-960/data_cfg/test_dev-clean+other.yaml ./data_cfg
   mv ./data_cfg/test_dev-clean+other.yaml ./data_cfg/test_{your-valid-set-name}.yaml
   bash run.sh --train false --exp_cfg 960-bpe5k_transformer-wide_ctc_perturb --data_cfg test_{your-valid-set-name}
   ```
   **Note:** 
   1. Please change the dataset argument at the beginning of `./data_cfg/test_{your-valid-set-name}.yaml`:
      ```
      dataset: librispeech -> 'your-new-dataset'
      valid_dset: &valid_dset dev-clean -> 'valid-set-of-new-dataset'
      ```
   2. The argument `--train false` is used to skip the training stage.
   3. `--data_cfg` is used to change the data loading configuration from the original one for training in `exp_cfg` to the one for validation tuning.
   4. If your experimental results are saved outside the toolkit, please attach `--train_result_path {your-target-path}` to `bash run.sh`.
6. **Evaluate the trained ASR model** on the official test sets
   ```
   bash run.sh --train false --exp_cfg 960-bpe5k_transformer-wide_ctc_perturb --infer_cfg "{the-best-configuration-you-get-during-validation-tuning}"
   ```
   **Note:** 
   1. The argument `--train false` is used to skip the training stage.
   2. There are two ways to specify the optimal `infer_cfg` tuned on the validation set:
      1. Change `infer_cfg` in `${SPEECHAIN_ROOT}/recipes/asr/librispeech/train-clean-100/exp_cfg/960-bpe5k_transformer-wide_ctc_perturb.yaml`.
      2. Give a parsable string as the value for `--infer_cfg` in the terminal. For example, `beam_size:16,temperature:1.5` can be converted into a dictionary with two key-value items (`beam_size:16` and `temperature:1.5`).  
      For more details about how to give the parsable string that can be converted into a dictionary, please refer to [**here**](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#convertable-arguments-in-the-terminal) for instructions.
   3. If your experimental results are saved outside the toolkit, please attach `--train_result_path {your-target-path}` to `bash run.sh`.  

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/recipes/asr#table-of-contents)

# How to create your own ASR model


👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/recipes/asr#table-of-contents)
