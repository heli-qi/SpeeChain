# Automatic Speech Recognition (ASR)

👆[Back to the recipe README.md](https://github.com/ahclab/SpeeChain/tree/main/recipes#recipes-folder-of-the-speechain-toolkit)

## Table of Contents
1. [Model Structure](https://github.com/ahclab/SpeeChain/tree/main/recipes/asr#model-structure)
2. [Configuration Format](https://github.com/ahclab/SpeeChain/tree/main/recipes/asr#configuration-format)
3. [Available Backbones](https://github.com/ahclab/SpeeChain/tree/main/recipes/asr#available-backbones)
4. [API Document](https://github.com/ahclab/SpeeChain/tree/main/recipes/asr#api-document)
5. [How to train an ASR model](https://github.com/ahclab/SpeeChain/tree/main/recipes/asr#how-to-train-an-asr-model)

## Model Structure
![image](model_fig.png)

The neural network structure of an `ASR` _Model_ object is made up of two _Module_ members: an `ASREncoder` _Module_ object and an `ASRDecoder` _Module_ object.
1. `ASREncoder` is made up of:
   1. `frontend` converts the raw waveforms into acoustic features on-the-fly.
   2. `normalize` normalizes the extracted acoustic features to normal distribution for faster convergence.
   3. `specaug` randomly warps and masks the normalized acoustic features.
   4. `prenet` preprocesses the augmented acoustic features before passing them to the encoder.
   5. `encoder` extracts the encoder hidden representations of the preprocessed acoustic features and passes them to `ASRDecoder`.
2. `ASRDecoder` is made up of:
   1. `embedding` embeds each tokens in the input sentence into token embedding vectors.
   2. `decoder` extracts the decoder hidden representations based on the token embedding vectors and encoder hidden representations.
   3. `postnet` predicts the probability of the next tokens by the decoder hidden representations.

The concrete implementation classes of the blocks with dashed edges can be freely selected among the available options in the _Dict_ members `{block_name}_class_dict` of `ASREncoder` and `ASRDecoder`.

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/recipes/asr#table-of-contents)

## Configuration Format
```
model:
    model_type: xxx.XXX
    customize_conf:
        token_type: ...
        token_vocab: ...
        sample_rate: (optional) 
        audio_format: (optional)
    module_conf:
         frontend:
            type: ...
            conf:
                ...
         normalize:
            ...
         specaug:
            ...
         enc_prenet:
            type: ...
            conf:
                ...
         encoder:
            type: ...
            conf:
                ...
         dec_emb:
            type: ...
            conf:
                ...
         decoder:
            type: ...
            conf:
                ...
    criterion_conf:
        is_normalized: (optional)
        label_smoothing: (optional)
```

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/recipes/asr#table-of-contents)

## Available Backbones
1. Speech-Transformer ([paper reference](https://ieeexplore.ieee.org/abstract/document/8462506))
    ```
    model_type: ar_asr.ARASR
    module_conf:
        frontend:
            type: mel_fbank
            conf:
                ...
        enc_prenet:
            type: conv2d
            conf:
                ...
        encoder:
            type: transformer
            conf:
                ...
        dec_emb:
            type: emb
            conf:
                ...
        decoder:
            type: transformer
            conf:
                ...
    ```
2. Paraformer ([paper reference](https://arxiv.org/pdf/2206.08317), coming soon~)

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/recipes/asr#table-of-contents)

## API Document
1. [speechain.model.asr.ASR.module_init](https://github.com/ahclab/SpeeChain/tree/main/recipes/asr#speechainmodelasrasrmodule_init)
2. [speechain.model.asr.ASR.criterion_init](https://github.com/ahclab/SpeeChain/tree/main/recipes/asr#speechainmodelasrasrcriterion_init)
3. [speechain.model.asr.ASR.inference](https://github.com/ahclab/SpeeChain/tree/main/recipes/asr#speechainmodelasrasrinference)

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/recipes/asr#table-of-contents)

### speechain.model.asr.ASR.module_init()
* **Description:**  
    This initialization function contains three steps: 
    1. `Tokenizer` initialization. 
    2. `ASREncoder` initialization.
    3. `ASRDecoder` initialization.  
    
    The input arguments of this function are two-fold:
    1. the ones from `customize_conf` of `model` in `train_cfg`
    2. the ones from `module_conf` of `model` in `train_cfg`
* **Arguments:**  
  _Arguments from `customize_conf`:_  
  * _**token_type:**_ str  
    The type of the built-in tokenizer.  
    Currently, we support 'char' for `CharTokenizer` and 'subword' for `SubwordTokenizer`.
  * _**token_vocab:**_ str  
    The path of the vocabulary list `vocab` for initializing the built-in tokenizer.
  * _**sample_rate:**_ int = 16000  
    The sampling rate of the input speech.  
    Currently, it's used for acoustic feature extraction frontend initialization and tensorboard register of the input speech for model visualization.  
    In the future, this argument will also be used to on-the-fly downsample the input speech.
  * _**audio_format:**_ str == 'wav'   
    This argument is only used for input speech recording during model visualization.
  * _**ctc_weight:**_ float = 0.0  
    The weight on the CTC loss for training the ASR model. 
    If `ctc_weight` is 0, the CTC layer will be disabled and the ASR model is trained only by the cross-entropy loss.
  * _**return_att_type:**_ List[str] or str = `['encdec', 'enc', 'dec'] ` 
    The type of attentions you want to return for both attention guidance and attention visualization.
    It can be given as a string (one type) or a list of strings (multiple types).
    The type should be one of
      1. `encdec`: the encoder-decoder attention, shared by both Transformer and RNN
      2. `enc`: the encoder self-attention, only for Transformer
      3. `dec`: the decoder self-attention, only for Transformer
  * _**return_att_head_num:**_ int = -1  
    The number of returned attention heads. If -1, all the heads in an attention layer will be returned.
    RNN can be considered to one-head attention, so `return_att_head_num` > 1 is equivalent to 1 for RNN.
  * _**return_att_layer_num:**_ int = 1  
    The number of returned attention layers. If -1, all the attention layers will be returned.
    RNN can be considered to one-layer attention, so `return_att_layer_num` > 1 is equivalent to 1 for RNN.

  _Arguments from `module_conf`:_  
  * _**frontend:**_ Dict  
    The configuration of the acoustic feature extraction frontend in the `ASREncoder` member.
    This argument must be given since our toolkit doesn't support time-domain ASR.  
    For more details about how to give `frontend`, please refer to [speechain.module.encoder.asr.ASREncoder](https://github.com/ahclab/SpeeChain/blob/main/speechain/module/encoder/asr.py#L20).
  * _**enc_prenet:**_ Dict  
    The configuration of the prenet in the `ASREncoder` member.  
    For more details about how to give `enc_prent`, please refer to [speechain.module.encoder.asr.ASREncoder](https://github.com/ahclab/SpeeChain/blob/main/speechain/module/encoder/asr.py#L20).
  * _**encoder:**_ Dict  
    The configuration of the encoder main body in the `ASREncoder` member.  
    For more details about how to give `encoder`, please refer to [speechain.module.encoder.asr.ASREncoder](https://github.com/ahclab/SpeeChain/blob/main/speechain/module/encoder/asr.py#L20).
  * _**dec_emb:**_ Dict  
    The configuration of the embedding layer in the `ASRDecoder` member. 
    For more details about how to give `dec_emb`, please refer to [speechain.module.decoder.asr.ASRDecoder](https://github.com/ahclab/SpeeChain/blob/main/speechain/module/decoder/asr.py#L17).
  * _**decoder:**_ Dict  
    The configuration of the decoder main body in the `ASRDecoder` member.  
    For more details about how to give `decoder`, please refer to [speechain.module.decoder.asr.ASRDecoder](https://github.com/ahclab/SpeeChain/blob/main/speechain/module/decoder/asr.py#L17).
  * _**normalize:**_ Dict or bool = None  
    The configuration of the normalization layer in the `ASREncoder` member.  
    This argument can also be given as a bool value. True means the default configuration and False means no normalization.  
    For more details about how to give `normalize`, please refer to [speechain.module.norm.feat_norm.FeatureNormalization](https://github.com/ahclab/SpeeChain/blob/main/speechain/module/norm/feat_norm.py#L11).
  * _**specaug:**_ Dict or bool = None  
    The configuration of the SpecAugment layer in the `ASREncoder` member.  
    This argument can also be given as a bool value. True means the default configuration and False means no SpecAugment.  
    For more details about how to give `specaug`, please refer to [speechain.module.augment.specaug.SpecAugment](https://github.com/ahclab/SpeeChain/blob/main/speechain/module/augment/specaug.py#L8).

👆[Back to the API list](https://github.com/ahclab/SpeeChain/tree/main/recipes/asr#api-document)

### speechain.model.asr.ASR.criterion_init()
* **Description:**  
    This function initializes all the necessary _Criterion_ members:
    1. `speechain.criterion.cross_entropy.CrossEntropy` for CrossEntropy loss calculation.
    2. `speechain.criterion.ctc.CTCLoss` for CTC loss calculation.
    3. `speechain.criterion.accuracy.Accuracy` for teacher-forcing validation accuracy calculation.
    4. `speechain.criterion.error_rate.ErrorRate` for evaluation CER & WER calculation.
* **Arguments:**
  * _**ce_normalized:**_ bool = False  
    Controls whether the sentence normalization is performed for cross-entropy loss.  
    For more details, please refer to [speechain.criterion.cross_entropy.CrossEntropy](https://github.com/ahclab/SpeeChain/blob/main/speechain/criterion/cross_entropy.py#L14)
  * _**label_smoothing:**_ float = 0.0  
    Controls the scale of label smoothing. 0 means no smoothing.  
    For more details, please refer to [speechain.criterion.cross_entropy.CrossEntropy](https://github.com/ahclab/SpeeChain/blob/main/speechain/criterion/cross_entropy.py#L14)
  * _**ctc_zero_infinity:**_ bool = True  
    Whether to zero infinite losses and the associated gradients when calculating the CTC loss.
  * _**att_guid_sigma:**_ float = 0.0  
    The value of the sigma used to calculate the attention guidance loss.
    If this argument is set to 0.0, the attention guidance will be disabled.

👆[Back to the API list](https://github.com/ahclab/SpeeChain/tree/main/recipes/asr#api-document)

### speechain.model.asr.ASR.inference()
* **Description:**  
  The inference function for ASR models. There are two steps in this function:
    1. Decode the input speech into hypothesis transcript
    2. Evaluate the hypothesis transcript by the ground-truth  

  This function can be called for model evaluation, on-the-fly model visualization, and even pseudo transcript generation during training.
* **Arguments:**  
  _Arguments from `self.evaluate()`:_  
  * _**infer_conf:**_ Dict  
    The inference configuration given from the `infer_cfg` in your `exp_cfg`.  
    For more details, please refer to [speechain.infer_func.beam_search.beam_searching](https://github.com/ahclab/SpeeChain/blob/main/speechain/infer_func/beam_search.py#L106).
  * _**feat:**_ torch.Tensor  
    The speech data to be inferred.
  * _**feat_len:**_ torch.Tensor  
    The length of `feat`.
  * _**text:**_ torch.Tensor  
    The ground-truth transcript for the input speech
  * _**text_len:**_ torch.Tensor  
    The length of `text`.
  
  _Arguments given by other functions:_  
  * _**domain:**_ str = None   
    This argument indicates which domain the input speech belongs to.  
    It's used to indicate the `ASREncoder` member how to encode the input speech.
  * _**return_att:**_ bool = False  
    Whether the attention matrix of the input speech is returned.
  * _**decode_only:**_ bool = False  
    Whether skip the evaluation step and do the decoding step only.
  * _**teacher_forcing:**_ bool = False  
    Whether you use the teacher-forcing technique to generate the hypothesis transcript.
* **Return:** Dict  
    A Dict containing all the decoding and evaluation results.

👆[Back to the API list](https://github.com/ahclab/SpeeChain/tree/main/recipes/asr#api-document)

👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/recipes/asr#table-of-contents)

## How to train an ASR model
Suppose that we want to train an ASR model by the configuration `${SPEECHAIN_ROOT}/recipes/asr/librispeech/train-clean-100/exp_cfg/transformer-narrow_accum1_20gb.yaml`.
1. **Train the ASR model** on your target training set
   ```
   cd ${SPEECHAIN_ROOT}/recipes/asr/librispeech/train-clean-100
   bash run.sh --train true --exp_cfg transformer-narrow_accum1_20gb (--ngpu x --gpus x,x)
   ```
   **Note:** 
   1. Please take a look at the comments in the configuration file to make sure that your computational equipments fit the configuration before training the model.
      If your equipments don't match the configuration, please adjust it by `--ngpu` and `--gpus`.
   2. If you want to save the experimental results outside the toolkit folder `${SPEECHAIN_ROOT}`, 
      please specify where you want to save the results by attaching `--train_result_path {your-target-path}` to `bash run.sh`.  
      In this example, if you give `bash run.sh --train true --exp_cfg transformer-narrow_accum1_20gb --train_result_path /a/b/c`, 
      the results will be saved to `/a/b/c/transformer-narrow_accum1_20gb`.
2. **Tune the inference hyperparameters** on the corresponding validation set
   ```
   bash run.sh --test true --exp_cfg transformer-narrow_accum1_20gb --data_cfg validtune_dev-clean
   ```
   **Note:** 
   1. `--data_cfg` is used to change the data loading configuration from the original one for training in `exp_cfg` to the one for validation tuning.
   2. If your experimental results are saved outside the toolkit, please attach `--train_result_path {your-target-path}` to `bash run.sh`.
3. **Evaluate the trained ASR model** on the official test sets
   ```
   bash run.sh --test true --exp_cfg transformer-narrow_v1_accum1_ngpu2 --infer_cfg "{the-best-configuration-you-get-during-validation-tuning}"
   ```
   **Note:** There are two ways to specify the optimal `infer_cfg` tuned on the validation set:
   1. Change `infer_cfg` in `${SPEECHAIN_ROOT}/recipes/asr/librispeech/train-clean-100/exp_cfg/transformer-narrow_accum1_20gb.yaml`.
   2. Give a parsable string as the value for `--infer_cfg` in the terminal. For example, `{beam_size:16,temperature:1.5}` can be converted into a dictionary with two key-value items (`beam_size:16` and `temperature:1.5`).  
   For more details about how to give the parsable string that can be converted into a dictionary, please refer to [**here**](https://github.com/ahclab/SpeeChain/blob/main/handbook.md#convertable-arguments-in-the-terminal) for instructions.
   3. If your experimental results are saved outside the toolkit, please attach `--train_result_path {your-target-path}` to `bash run.sh`.  
   
👆[Back to the table of contents](https://github.com/ahclab/SpeeChain/tree/main/recipes/asr#table-of-contents)

