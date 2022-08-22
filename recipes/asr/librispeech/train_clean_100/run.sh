python=/home/is/heli-qi/anaconda3/envs/speechain/bin/python3.8
speechain=/ahc/work4/heli-qi/euterpe-heli-qi/speechain
cur_path=/ahc/work4/heli-qi/euterpe-heli-qi/recipes/asr/librispeech/train_clean_100

#CUDA_VISIBLE_DEVICES=1,2,3 ${python} ${speechain}/runner.py \
#  --config ${cur_path}/transformer/exp_cfg/char_smooth0.2.yaml \
#  --train \
#  --ngpu 3

CUDA_VISIBLE_DEVICES=0,1,2,3 ${python} ${speechain}/runner.py \
  --config ${cur_path}/transformer/exp_cfg/char_smooth0.2.yaml \
  --test \
  --ngpu 4