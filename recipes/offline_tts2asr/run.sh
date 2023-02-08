bash tts_synthesize.sh \
  --tts_model_path recipes/tts/libritts/train-clean-100/exp/16khz_ecapa_g2p_transformer-v3_accum1_20gb \
  --tts_syn_dataset libritts \
  --tts_syn_subset train-clean-360 \
  --spk_emb_subset train-clean-100 \
  --spk_emb_model ecapa \
  --batch_len 2000 \
  --ngpu 2 --gpus 0,1

${SPEECHAIN_PYTHON} ${SPEECHAIN_ROOT}/recipes/tts/feat_to_wav.py \
  --feat_path recipes/offline_tts2asr/tts_syn_speech/libritts/train-clean-360/default_inference/10_train_loss_average/seed=0_spk-emb=libritts-train-clean-100-ecapa_model=recipes%tts%libritts%train-clean-100%exp%16khz_ecapa_g2p_transformer-v3_accum1_20gb/idx2feat \
  --tts_model_cfg recipes/tts/libritts/train-clean-100/exp/16khz_ecapa_g2p_transformer-v3_accum1_20gb/train_cfg.yaml \
  --batch_size 10 \
  --ngpu 2