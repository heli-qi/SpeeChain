### Author: Heli Qi
### Affiliation: NAIST
### Date: 2022.09

### --- Reference Values --- ###
python=/home/is/heli-qi/anaconda3/envs/speechain/bin/python3.8
infer_root=/ahc/work4/heli-qi/euterpe-heli-qi/config/infer/asr
speechain=/ahc/work4/heli-qi/euterpe-heli-qi/speechain
### ------------------------ ###


### --- (optional) 1. Data Loading Part Pre-heating --- ###
# For the first time you run a job using a new dataset, your job may suffer from long data loading time. It's probably
# because your target data haven't been accessed since your machine is turned on, hence it's difficult for your machine
# to read them into memory. If that happens, you could use the argument '--dry_run' to only perform the data loading for
# one epoch with '--num_epochs 1'. This will help your machine better access your target dataset.
#
# If the data loading speed of your job is still very slow even after the pre-heating, it's probably because your machine
# doesn't have enough memory for your target dataset. The lack of memory could be caused by either the limited equippment
# of your machine or occupation by the jobs of other memebers in your team.
### ----------------------------------------- ###
CUDA_VISIBLE_DEVICES=0,1,2,3 ${python} ${speechain}/runner.py \
  --config ./transformer/exp_cfg/transformer_specaug.yaml \
  --train \
  --dry_run \
  --num_epochs 1 \
  --ngpu 4


### --- 2. Model Training --- ###
# The GPUs can be specified by either 'CUDA_VISIBLE_DEVICES' or '--gpus'. They are identical to the backbone.
# If '--result_path' is not given, the experimental files will be automatically saved to /exp under the same directory
# of your given '--config'.
# In the example below, the experimental files will be saved to ./transformer/exp/transformer_specaug
### ------------------------- ###
${python} ${speechain}/runner.py \
  --config ./transformer/exp_cfg/transformer_specaug.yaml \
  --data_cfg ./data_cfg/train_frame10M_batch6k.yaml \
  --train \
  --gpus 0,1,2,3 \
  --ngpu 4


### --- (optional) 3. Model Training Resuming --- ###
# The training can be resumed from an existing checkpoint by giving the argument '--resume'.
# Note that if you give 'data_cfg' explicitly as the example below, the new data loading configuration will be adopted
# for resuming the model training. Otherwise, the data loading configuration of the last time will be used.
### --------------------------------------------- ###
#${python} ${speechain}/runner.py \
#  --config ./transformer/exp_cfg/transformer_specaug.yaml \
#  (--data_cfg <your new data loading configuration file path> \)
#  --train \
#  --resume \
#  --gpus 2,3 \
#  --ngpu 2


### --- 4. Model Testing --- ###
# To start the testing stage, you only need to replace the argument '--train' with '--test' like the way below.
### ------------------------- ###
${python} ${speechain}/runner.py \
  --config ./transformer/exp_cfg/transformer_specaug.yaml \
  --data_cfg ./data_cfg/test_clean+other_ngpu2.yaml \
  --test_cfg ${infer_root}/beam16_temperature1.3.yaml \
  --test \
  --ngpu 2


### --- (optional) 5. Model Testing Resuming --- ###
# The testing stage can also be resumed in our toolkit. The configuration is the same with training resuming.
# But one thing should be noted is that you must use the identical data loading configuration for resuming.
# It means that you should not give a new configuration by '--data_cfg'.
### -------------------------------------------- ###
#CUDA_VISIBLE_DEVICES=2,3 ${python} ${speechain}/runner.py \
#  --config ./transformer/exp_cfg/transformer_specaug.yaml \
#  ×(--data_cfg <your new data loading configuration file path> \)×
#  --data_cfg ./data_cfg/test_clean+other_ngpu2.yaml \
#  --test \
#  --resume \
#  --ngpu 2