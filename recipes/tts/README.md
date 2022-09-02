# TTS
Recipes for TTS model. TTS consists of 2 modules:
1. Core module: Transform text sequence into speech feature sequence.
2. Vocoder/inverter module: Transform feature sequence into waveform

👆[Back to the home page](https://github.com/ahclab/SpeeChain#the-speechain-toolkit)

## Data preparation
Example:
1. Acoustic feature (mel-spectrogram): datasets/speech_text/ljspeech/run_taco_mel_f80.sh <br>
For the core module's output.
2. Linear spectrogram: datasets/speech_text/ljspeech/run_taco_raw_fft2048.sh <br>
For vocoder output (waveform will be reconstructed using griffin-lim)

## Single-speaker TTS
Example: [LJSpeech](https://github.com/ahclab/SpeeChain/tree/tts/recipes/tts/ljspeech)

### Training
#### 1. Core module

Script
```
python runner.py --config recipes/tts/ljspeech/transformer/exp_cfg/sup_8head_512dim_warmup20k_do0.15_0.2_berndo0.05.yaml
```

Pay attention into the following model hyperparameters (train_cfg inside --config file):
```
1. model_conf.customize_conf.speechfeat_group
Speech feature grouping factor to shorten the feature length by stacking multiple frames into 1 frame. E.g:
-- Before: (seq. length, feature dim) = (100,80)
-- After (speechfeat_group=1): (seq. length, feature dim) = (100/4,80*4) = (25,320)

2. module_conf.encoder.conf.embedding.conf.vocab_size
Vocabulary size for TTS input (must same as model_conf.customize_conf.token_dict)

3. module_conf.decoder.conf.prenet.conf.feat_dim
Original feature dimension * speechfeat_group

4. module_conf.decoder.conf.featnet.conf.lnr_dims
Original feature dimension * speechfeat_group

5. module_conf.decoder.conf.bernnet.conf.lnr_dims
Last layer's dimension = 1

6. criterion_conf.bern_loss.coef.coeff
Loss weight for speech end-flag prediction

7. criterion_conf.bern_loss.coef.coeff_bern_positive
Loss weight for speech end-flag prediction for the positive label. Usually between 5-8 for good performance.
```
In --config file, data_cfg is exactly the same as data_cfg for ASR (use as it is.)

#### 2. Vocoder
Script
```
python $runner --config recipes/tts/ljspeech/vocoder/exp_cfg/sup_TacotronV1Inverter_lrelu_group4.yaml
```
Data_cfg:
```
- src_data : Input (acoustic feature)
- tgt_label : Output target (linear spectrogram) 
Input/output sequence length must be the same.
```
Train_cfg: Pay attention into the following model hyperparameters:
```
1. model_conf.customize_conf.speechfeat_group
Speech feature grouping factor to shorten the feature length by stacking multiple frames into 1 frame. Will be applied to src_data and tgt_label

2. module_conf.vocoder.conf.in_size
Original input feature dimension * speechfeat_group

3. module_conf.vocoder.conf.out_size
Original output feature dimension * speechfeat_group

6. criterion_conf.feat_loss.conf.coeff
Loss weight.

7. criterion_conf.feat_freq_loss.topn
Loss for low frequency spectrogram. topn: output dimension range for loss calculation.
```

### Inference
#### 1. Core module
Script
```
python runner.py --config recipes/tts/ljspeech/transformer/exp_cfg/INFERENCE_sup_8head_512dim_warmup20k_do0.15_0.2_berndo0.05.py
```
(check the result_path inside --config. Must be the same as the folder path to the model) <br>
Output: acoustic feature in .npz file
#### 2. Vocoder
##### A. Input: Gold acoustic feature
```
python runner.py --config recipes/tts/ljspeech/transformer/exp_cfg/INFERENCE_sup_8head_512dim_warmup20k_do0.15_0.2_berndo0.05.py
```

##### B. Input: Acoustic feature predicted by the core module.
1. Run inference with core module <br>
1.1. Go to result_path/(test name)/(test model's name)/clean/ then copy `idx2hypo_feat_npz` file.
2. Open --config file for vocoder inference (gold) above
3. Go to data_cfg file and open data_cfg file
4. In data_cfg, go to test.clean.conf.dataset_conf.src_data, replace the path with `idx2hypo_feat_npz` from step 1.1.

## Multi-speaker TTS
Example: [WSJ](https://github.com/ahclab/SpeeChain/tree/tts/recipes/tts/wsj)
### Training
1. Train speaker recognition model and generate all data's speaker embedding beforehand (see: [speaker recognition](https://github.com/ahclab/SpeeChain/tree/tts/recipes/spkrec)) <br>
1.1. After generating the embedding, get the speaker embedding list `idx2pred_embedding`. <br>
1.2. In TTS data_cfg, put the following for each train, valid, and test sections:
```
dataset_conf:
    feat_type: feat
    src_data: <path to txt>
    tgt_label: <path to acoustic feat>
    meta_info:
        speaker_feat: <path to idx2pred_spkembedding>
```
2. Train core module. Same as the single-speaker version with exceptions:
- data_cfg: add meta_info.speaker_feat (filepath to list of speaker embedding filepath)
- train_cfg: set module_conf.decoder.type into `decoder.tts.TTSDecoderMultiSpeaker` <br>
Script
```
python runner.py --config recipes/tts/wsj/transformer/exp_cfg/sup_8head_512dim_warmup20k_do0.15_0.2_berndo0.05.yaml
```
3. Train vocoder model (same as the single speaker version but use a multispeaker data)
```
python runner.py --config recipes/tts/wsj/vocoder/exp_cfg/sup_TacotronV1Inverter_lrelu_group4.yaml
```

### Inference
#### Core
```
python runner.py --config recipes/tts/wsj/transformer/exp_cfg/INFERENCE_sup_8head_512dim_warmup20k_do0.15_0.2_berndo0.05.yaml
```

#### Vocoder
##### A. Input: Gold acoustic feature
```
python runner.py --config recipes/tts/wsj/transformer/exp_cfg/INFERENCE_sup_8head_512dim_warmup20k_do0.15_0.2_berndo0.05.yaml
```
##### B. Input: Acoustic feature predicted by the core module.

1. Run inference with core module <br>
1.1. Go to result_path/(test name)/(test model's name)/clean/ then copy `idx2hypo_feat_npz` file.
2. Open --config file for vocoder inference (gold) above
3. Go to data_cfg file and open data_cfg file
4. In data_cfg, go to test.clean.conf.dataset_conf.src_data, replace the path with `idx2hypo_feat_npz` from step 1.1.

## Training Tips
1. Core: Use big batchsize
2. Core: Use big number of batch/epoch
3. Core: Configure the hyperparams for speech-end flag well
4. Core (multi-speaker): Speaker recognition model accuracy > 99% (for speaker embedding generation) 