# Speaker Recognition
Recipes for speaker recognition model. Current implementation:
1. [Deep-Speaker](https://arxiv.org/abs/1705.02304)

For multi-speaker TTS, we will use the speaker recognition model output BEFORE the output/last layer.

👆[Back to the home page](https://github.com/ahclab/SpeeChain#the-speechain-toolkit)


## Training
### script
```
python runner.py --config recipes/spkrec/wsj/deepspeaker/exp_cfg/sup.yaml
```

#### train_cfg 
Pay attention into the following model hyperparameters:
```
1. model_conf.customize_conf.speaker_list
Filepath to speaker list (same format as vocab file for ASR/TTS).

2. module_conf.spkrec.conf.spkrec.in_size
Input (acoustic feature) dimension

3. module_conf.spkrec.conf.spkrec.out_emb_size
Speaker embedding dimension. Speaker embedding will be used in multi-speaker TTS.

3. module_conf.spkrec.conf.spkrec.num_speaker
Number of speaker in model_conf.customize_conf.speaker_list
```
#### data_cfg
Use the same data_cfg as ASR/TTS but also by adding meta_info.speaker under dataset_conf.
- meta_info.speaker : Filepath containing mapping: utterance ID to speaker ID


## Inference
### Script
```
python runner.py --config recipes/spkrec/wsj/deepspeaker/exp_cfg/sup_infer.yaml
```
### Speaker embedding generation for TTS (offline)
1. Open --config file above, go to data_cfg
2. In data_cfg, go to test section
3. Fill the src_data, tgt_label, meta_info, data_len with the corresponding files containing all utterance key for TTS training, dev, test
4. Run the inference script
5. Go to (result_path)/(test name)/(test model epoch)/clean then copy `idx2pred_embedding` filepath, then paste it to multi-speaker TTS data_cfg meta_info.speaker_feat section (see: TTS recipe example for WSJ dataset)