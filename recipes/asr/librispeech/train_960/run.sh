python=/home/is/heli-qi/anaconda3/envs/speechain/bin/python3.8
speechain=/ahc/work4/heli-qi/euterpe-heli-qi/speechain

#${python} ${speechain}/runner.py \
#  --config ./transformer/exp_cfg/bpe10k_middlespace_smooth0.2.yaml \
#  --train \
#  --ngpu 2 \
#  --dry_run \
#  --num_epochs 1

${python} ${speechain}/runner.py \
  --config ./transformer/exp_cfg/bpe5k_smooth0.2_normalized.yaml \
  --train \
  --ngpu 3

CUDA_VISIBLE_DEVICES=1,2 ${python} ${speechain}/runner.py \
  --config ./transformer/exp_cfg/bpe5k_smooth0.2_normalized.yaml \
  --test \
  --ngpu 2