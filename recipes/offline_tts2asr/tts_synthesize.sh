#  Author: Heli Qi
#  Affiliation: NAIST
#  Date: 2022.12


if [ -z "${SPEECHAIN_ROOT}" ] || [ -z "${SPEECHAIN_PYTHON}" ];then
  echo "Cannot find environmental variables SPEECHAIN_ROOT and SPEECHAIN_PYTHON.
  Please move to the root path of the toolkit and run envir_preparation.sh there!"
  exit 1
fi

function print_help_message {
  echo "usage:
  $0 \\ (The arguments in [] are optional while other arguments must be given by your run.sh.)

      # Group1: TTS Synthesis Environment
      [--batch_len BATCH_LEN] \\                             # The total length of all unlabeled sentences in a single batch. This argument is required if you want to conduct batch-level TTS synthesis. We recommend you to set 'batch_len' up to {25 * reduction_factor * total GBs of your GPUs} for AR-TTS and {100 * reduction_factor * total GBs of your GPUs} for NAR-TTS. (default: none)
      [--desc_sorting DESC_SORTING] \\                       # Whether to sort text data in the descending order. For FastSpeech2, we recommend you to set this argument to false because FastSpeech2 consumes more time and space when processing large batches of short text. (default: True)
      [--random_seed RANDOM_SEED] \\                         # The random seed used to control the randomness of reference speaker for TTS synthesis. (default: 0)
      [--ngpu NGPU] \\                                       # The number of GPUs you want to use. (default: 1)
      [--gpus GPUS] \\                                       # The GPUs you want to specify. (default: none)
      [--num_workers NUM_WORKERS] \\                         # The name of worker processes for data loading. (default: 1)
      [--resume RESUME] \\                                   # Whether to continue the unfinished evaluation. If true, the data loading strategy should remain the same as the last time. (default: false)
      [--saving_proc_num SAVING_PROC_NUM] \\                 # The number of daemon processes per GPU used to save synthetic speech data to the disk. (default: 1)

      # Group2: Speaker Embedding
      [--rand_spk_emb RAND_SPK_EMB] \\                       # Whether to randomly generate the speaker embedding feature for TTS synthesis. (default: false)
      [--spk_emb_mixup SPK_FEAT_MIXUP] \\                    # Whether to conduct speaker embedding mixup. (default: false)
      [--same_gender_mixup SAME_GENDER_MIXUP] \\             # Whether to conduct the mixup for the speakers with the same gender. (default: true)
      [--mixup_number MIXUP_NUMBER] \\                       # The number of speaker embedding features used for mixup. (default: 2)
      [--use_aver_feat USE_AVER_FEAT] \\                     # Whether to use the average speaker embedding of each speaker. (default: true)
      [--spk_emb_dataset SPK_EMB_DATASET] \\                 # The dataset where your target speaker embedding features are placed. If not given, this argument will be the same as 'tts_syn_dataset'. (default: none)
      [--spk_emb_subset SPK_EMB_SUBSET] \\                   # The subset where your target speaker embedding features are placed. If not given, this argument will be the same as 'tts_syn_subset'. (default: none)
      [--spk_emb_model SPK_EMB_MODEL] \\                     # The speaker embedding model you want to use for TTS synthesis. (default: none)

      # Group3: Token & Text
      [--txt_format TXT_FORMAT] \\                           # The text format of the text data. (default: no-punc)
      [--token_type TOKEN_TYPE] \\                           # The type of tokens in your target vocabulary. (default: g2p)
      [--token_num TOKEN_NUM] \\                             # The number of tokens in your target vocabulary. (default: stress)

      # Group4: Main Arguments for Synthesis
      [--tts_infer_cfg TTS_INFER_CFG] \\                     # The configuration for TTS inference. If not given, infer_cfg in {tts_model_path}/exp_cfg.yaml will be used. (default: none)
      [--syn_result_path SYN_RESULT_PATH] \\                 # The path where you want to place the synthetic pseudo utterances. (default: ${SPEECHAIN_ROOT}/recipes/offline_tts2asr/tts_syn_speech/)
      [--dump_data_path DUMP_DATA_PATH] \\                   # The path where your dumped data is placed. If your data is stored outside the toolkit, please specify them by this argument. (default: ${SPEECHAIN_ROOT}/datasets/)
      --tts_model_path TTS_MODEL_PATH \\                     # The path of the TTS model you want to use. There must be 'models/', 'exp_cfg.yaml', and 'train_cfg.yaml' in your specified folder.
      --tts_syn_dataset TTS_SYN_DATASET \\                   # The dataset whose text data you want to use for TTS synthesis.
      --tts_syn_subset TTS_SYN_SUBSET                       # The subset of your chosen dataset whose text data you want to use for TTS synthesis." >&2
  exit 0
}

ngpu=1
gpus=none
batch_len=
desc_sorting=true
random_seed=0
resume=false
num_workers=1
saving_proc_num=1

rand_spk_emb=false
spk_emb_mixup=false
same_gender_mixup=true
mixup_number=2
use_aver_feat=true
spk_emb_dataset=
spk_emb_subset=
spk_emb_model=

txt_format=no-punc
token_type=g2p
token_num=stress

tts_infer_cfg=
syn_result_path=${SPEECHAIN_ROOT}/recipes/offline_tts2asr/tts_syn_speech
dump_data_path=${SPEECHAIN_ROOT}/datasets/
tts_model_path=
tts_syn_dataset=
tts_syn_subset=


### get args from the command line ###
while getopts ":h-:" optchar; do
  case "${optchar}" in
    -)
      case "${OPTARG}" in
        tts_syn_dataset)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          tts_syn_dataset=${val}
          ;;
        tts_syn_subset)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          tts_syn_subset=${val}
          ;;
        txt_format)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          txt_format=${val}
          ;;
        token_type)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          token_type=${val}
          ;;
        token_num)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          token_num=${val}
          ;;
        rand_spk_emb)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          rand_spk_emb=${val}
          ;;
        spk_emb_mixup)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          spk_emb_mixup=${val}
          ;;
        same_gender_mixup)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          same_gender_mixup=${val}
          ;;
        mixup_number)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          mixup_number=${val}
          ;;
        use_aver_feat)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          use_aver_feat=${val}
          ;;
        spk_emb_dataset)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          spk_emb_dataset=${val}
          ;;
        spk_emb_subset)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          spk_emb_subset=${val}
          ;;
        spk_emb_model)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          spk_emb_model=${val}
          ;;
        syn_result_path)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          syn_result_path=${val}
          ;;
        dump_data_path)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          dump_data_path=${val}
          ;;
        tts_model_path)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          tts_model_path=${val}
          ;;
        tts_infer_cfg)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          tts_infer_cfg=${val}
          ;;
        resume)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          resume=${val}
          ;;
        ngpu)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          ngpu=${val}
          ;;
        gpus)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          gpus=${val}
          ;;
        batch_len)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          batch_len=${val}
          ;;
        desc_sorting)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          desc_sorting=${val}
          ;;
        random_seed)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          random_seed=${val}
          ;;
        num_workers)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          num_workers=${val}
          ;;
        saving_proc_num)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          saving_proc_num=${val}
          ;;
        help)
          print_help_message
          ;;
        *)
          echo "Unknown variable --$OPTARG"
          exit 1
          ;;
      esac
      ;;
    h)
      print_help_message
      ;;
    *)
      echo "Please refer to an argument by '--'."
      exit 1
      ;;
  esac
done


# --- 0. Argument Checking --- #
if [ -z ${tts_model_path} ];then
   echo "Please give the path of the TTS model you want to use by '--tts_model_path TTS_MODEL_PATH'!"
   exit 1
else
  if ! grep -q '/' <<< "${tts_model_path}";then
    echo "For tts_model_path, please give either its absolute address or in-toolkit relative path!"
    exit 1
  fi
fi

if ! grep -q '/' <<< "${syn_result_path}";then
  echo "There is no slash inside your given 'syn_result_path', please give either an absolute address or an in-toolkit relative path!"
  exit 1
fi

if ! grep -q '/' <<< "${dump_data_path}";then
  echo "There is no slash inside your given 'dump_data_path', please give either an absolute address or an in-toolkit relative path!"
  exit 1
fi

if [ -z ${tts_syn_dataset} ] || [ -z ${tts_syn_subset} ];then
   echo "Please give your target text data by '--tts_syn_dataset TTS_SYN_DATASET' and '--tts_syn_subset TTS_SYN_SUBSET'!"
   exit 1
fi

if ${spk_emb_mixup} && [ -z ${spk_emb_model} ];then
  echo "Please give a speaker embedding model by '--spk_emb_model SPK_EMB_MODEL' if you give '--spk_emb_mixup true'!"
  exit 1
fi

if [ ${token_type} != 'g2p' ];then
  echo "Currently, we only support 'g2p' as 'token_type' for speech synthesis."
  exit 1
fi

if [ -z ${spk_emb_dataset} ];then
   spk_emb_dataset=${tts_syn_dataset}
fi

if [ -z ${spk_emb_subset} ];then
   spk_emb_subset=${tts_syn_subset}
elif [ ${spk_emb_dataset} == ${tts_syn_dataset} ] && [ ${spk_emb_subset} == ${tts_syn_subset} ]; then
    echo "spk_emb_subset should not be equal to tts_syn_subset!"
    exit 1
fi


# --- 1. Argument Initialization --- #
args="--train False --test True --attach_model_folder_when_test False --test_result_path ${syn_result_path}/${tts_syn_dataset}/${tts_syn_subset}"

# automatically initialize the data loading configuration. The contents surrounded by a pair of brackets are optional.
#  test:{
#    seed=${random_seed}_desc-sort=${desc_sorting}_n-proc=${saving_proc_num}_ngpu=${ngpu}(_batch-len=${batch_len})(_spk-emb=${spk_emb_dataset}-${spk_emb_subset}-${spk_emb_model})_model=${tts_model_path}:{
#      (type:block.BlockIterator,) or (type:abs.Iterator,)
#      conf:{
#        (dataset_type:speech_text.SpeechTextDataset,) or (dataset_type:speech_text.RandomSpkFeatDataset,)
#        dataset_conf:{
#          main_data:{
#            text:${unspoken_idx2text}
#          },
#          (spk_feat:${refer_idx2spk_feat},)
#          (use_aver_feat:${use_aver_feat},)
#          (mixup_number:${mixup_number},)
#          (data_selection:[min,${filter_ratio},${unspoken_idx2text_len}],)
#        },
#        shuffle:false,
#        is_descending:${desc_sorting},
#        (data_len:${unspoken_idx2text_len},)
#        (batch_len:${batch_len})
#      }
#    }
#  }
# the following code does the same job as the configuration above
data_args="test:{seed=${random_seed}_desc-sort=${desc_sorting}_n-proc=${saving_proc_num}_ngpu=${ngpu}"
if [ -n "${batch_len}" ];then
  data_args="${data_args}_batch-len=${batch_len}"
fi
# if 'rand_spk_emb' is given, attach rand_spk_emb into the folder name
if ${rand_spk_emb};then
  data_args="${data_args}_spk-emb=random"
# if 'spk_emb_model' is given, attach the model name and reference subset into the folder name
elif [ -n "${spk_emb_model}" ];then
  data_args="${data_args}_spk-emb=${spk_emb_dataset}-${spk_emb_subset}-${spk_emb_model}"
  if ${use_aver_feat};then
    data_args="${data_args}-aver"
  fi
  #
  if ${spk_emb_mixup};then
    data_args="${data_args}_mixup=${mixup_number}"
    if ${same_gender_mixup};then
      data_args="${data_args}-same-gender"
    else
      data_args="${data_args}-diff-gender"
    fi
  fi
fi
data_args="${data_args}_model=${tts_model_path}:{type:"

# if 'batch_len' is given, block.BlockIterator will be used as the iterator; else, use abs.Iterator
if [ -n "${batch_len}" ];then
  data_args="${data_args}block.BlockIterator,"
else
  data_args="${data_args}abs.Iterator,"
fi
data_args="${data_args}conf:{"

# if rand_spk_emb is set or 'spk_emb_model' is not given, SpeechTextDataset will be used as the Dataset; else, use RandomSpkFeatDataset
if ${rand_spk_emb} || [ -z "${spk_emb_model}" ];then
  data_args="${data_args}dataset_type:speech_text.SpeechTextDataset"
else
  data_args="${data_args}dataset_type:speech_text.RandomSpkFeatDataset"
fi

data_args="${data_args},dataset_conf:{main_data:{text:"
# the idx2text file containing the split tokens has the higher priority
unspoken_idx2text="${dump_data_path}/${tts_syn_dataset}/data/${token_type}/${tts_syn_subset}/${token_num}/${txt_format}/idx2text"
unspoken_idx2text_len="${unspoken_idx2text}_len"
# if the idx2text file containing the split tokens doesn't exist, use the one containing the raw transcripts
if [ ! -f "${unspoken_idx2text}" ];then
  unspoken_idx2text="${dump_data_path}/${tts_syn_dataset}/data/wav/${tts_syn_subset}/idx2${txt_format}_text"
fi
data_args="${data_args}${unspoken_idx2text}}"

# if 'spk_emb_model' is set to true, attach 'spk_feat' in 'dataset_conf'
if [ -n "${spk_emb_model}" ];then
  data_args="${data_args},spk_feat:${dump_data_path}/${spk_emb_dataset}/data/wav/${spk_emb_subset}/idx2${spk_emb_model}_spk_feat,use_aver_feat:${use_aver_feat}"
fi

# if 'spk_emb_mixup' is set to true, attach 'mixup_number' in 'dataset_conf'
if ${spk_emb_mixup};then
  data_args="${data_args},mixup_number:${mixup_number},same_gender:${same_gender_mixup}"
  if ${same_gender_mixup};then
    data_args="${data_args},same_gender:${same_gender_mixup}"
  fi
fi

data_args="${data_args}},shuffle:false,is_descending:${desc_sorting},data_len:${unspoken_idx2text_len}"
# if 'batch_len' is given, attach 'batch_len'
if [ -n "${batch_len}" ];then
  data_args="${data_args},batch_len:${batch_len}"
fi
data_args="${data_args}}}}"
# explicitly specify the data loading configuration obtained from the codes above
args="${args} --data_cfg ${data_args}"

# explicitly specify the random seed if given
if [ -n "${random_seed}" ];then
  args="${args} --seed ${random_seed}"
fi
# explicitly specify the used GPUs and its number
args="${args} --ngpu ${ngpu} --gpus ${gpus}"
# explicitly specify the number of worker processes for data loading
args="${args} --test_num_workers ${num_workers}"
# explicitly specify the number of daemon processes for data saving
args="${args} --saving_proc_num ${saving_proc_num}"

# exp_cfg.yaml and train_cfg.yaml generated during training in ${tts_model_path} will be used
args="${args} --config ${tts_model_path}/exp_cfg.yaml --train_result_path ${tts_model_path} --attach_config_folder_to_path false --train_cfg ${tts_model_path}/train_cfg.yaml"
# explicitly specify the inference configuration if given
if [ -n "${tts_infer_cfg}" ];then
  # do sth when infer_cfg is the name of a configuration file
  if ! grep -q ':' <<< "${tts_infer_cfg}";then
    # attach .yaml suffix if needed
    if [[ "${tts_infer_cfg}" != *".yaml" ]];then
      tts_infer_cfg="${tts_infer_cfg}.yaml"
    fi
    # convert the relative path in ${infer_root}/${task} if no colon inside
    if ! grep -q ':' <<< "${tts_infer_cfg}";then
      tts_infer_cfg="${SPEECHAIN_ROOT}/config/infer/tts/${tts_infer_cfg}"
    fi
  fi
  args="${args} --infer_cfg ${tts_infer_cfg}"
else
  args="${args} --infer_cfg {}"
fi

if ${resume};then
  args="${args} --resume True"
else
  args="${args} --resume False"
fi


# --- 3. Execute the Job --- #
# ${args} should not be surrounded by double-quote
# shellcheck disable=SC2086
${SPEECHAIN_PYTHON} "${SPEECHAIN_ROOT}"/speechain/runner.py ${args}