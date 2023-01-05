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
      [--batch_len BATCH_LEN] \\                             # The total length of all unlabeled sentences in a single batch. This argument is required if you want to conduct batch-level TTS synthesis. (default: none)
      [--random_seed RANDOM_SEED] \\                         # The random seed used to control the randomness of reference speaker for TTS synthesis. (default: 0)
      [--ngpu NGPU] \\                                       # The number of GPUs you want to use. (default: 1)
      [--gpus GPUS] \\                                       # The GPUs you want to specify. (default: none)
      [--num_workers NUM_WORKERS] \\                         # The name of worker processes for data loading. (default: 1)
      [--resume RESUME] \\                                   # Whether to continue the unfinished evaluation. If true, the data loading strategy should remain the same as the last time. (default: false)
      [--long_filter LONG_FILTER] \\                         # Whether to filter out long sentences with the largest text_len. (default: true)
      [--filter_ratio FILTER_RATIO] \\                       # How many shorter utterances you want to retain. (default: 0.95)
      [--tts_infer_cfg TTS_INFER_CFG] \\                     # The configuration for TTS inference. If not given, infer_cfg in {tts_model_path}/exp_cfg.yaml will be used. (default: none)
      [--syn_result_path SYN_RESULT_PATH] \\                 # The path where you want to place the synthetic pseudo utterances. (default: recipes/offline_tts2asr/tts_syn_speech)
      [--spk_emb_dataset SPK_EMB_DATASET] \\                 # The dataset where your target speaker embedding features are placed. If not given, this argument will be the same as 'tts_syn_dataset'. (default: none)
      [--spk_emb_subset SPK_EMB_SUBSET] \\                   # The subset where your target speaker embedding features are placed. If not given, this argument will be the same as 'tts_syn_subset'. (default: none)
      [--spk_emb_model SPK_EMB_MODEL] \\                     # The speaker embedding model you want to use for TTS synthesis. (default: none)
      [--txt_format TXT_FORMAT] \\                           # The text format of the text data. (default: normal)
      [--token_type TOKEN_TYPE] \\                           # The type of tokens in your target vocabulary. (default: g2p)
      [--token_num TOKEN_NUM] \\                             # The number of tokens in your target vocabulary. (default: full_tokens)
      --tts_model_path TTS_MODEL_PATH \\                     # The path of the TTS model you want to use. There must be 'models/', 'exp_cfg.yaml', and 'train_cfg.yaml' in your specified folder.
      --tts_syn_dataset TTS_SYN_DATASET \\                   # The dataset whose text data you want to use for TTS synthesis.
      --tts_syn_subset TTS_SYN_SUBSET                       # The subset of your chosen dataset whose text data you want to use for TTS synthesis." >&2
  exit 1
}


tts_model_path=
tts_infer_cfg=
syn_result_path=recipes/offline_tts2asr/tts_syn_speech

long_filter=true
filter_ratio=0.95

ngpu=1
gpus=
batch_len=
random_seed=0
resume=false
num_workers=1

tts_syn_dataset=
tts_syn_subset=
spk_emb_dataset=
spk_emb_subset=
spk_emb_model=
txt_format=normal
token_type=g2p
token_num=full_tokens


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
        tts_model_path)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          tts_model_path=${val}
          ;;
        tts_infer_cfg)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          tts_infer_cfg=${val}
          ;;
        long_filter)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          long_filter=${val}
          ;;
        filter_ratio)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          filter_ratio=${val}
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
        random_seed)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          random_seed=${val}
          ;;
        num_workers)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          num_workers=${val}
          ;;
        help)
          print_help_message
          ;;
        ?)
          echo "Unknown variable $OPTARG"
          ;;
      esac
      ;;
    h)
      print_help_message
      ;;
    *)
      echo "Please refer to an argument by '--'."
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
  echo "For 'syn_result_path', please give either an absolute address or an in-toolkit relative path!"
  exit 1
fi

if [ -z ${tts_syn_dataset} ] || [ -z ${tts_syn_subset} ];then
   echo "Please give your target text data by '--dataset DATASET' and '--subset SUBSET'!"
   exit 1
fi

if [ -z ${spk_emb_dataset} ];then
   spk_emb_dataset=${tts_syn_dataset}
fi

if [ -z ${spk_emb_subset} ];then
   spk_emb_subset=${tts_syn_subset}
fi


# --- 1. Argument Initialization --- #
args="--train False --test True --test_result_path ${syn_result_path}/${tts_syn_dataset}/${tts_syn_subset}"

# automatically initialize the data loading configuration. The contents surrounded by a pair of brackets are optional.
#  test:{
#    seed=${random_seed}(_spk-emb=${spk_emb_dataset}-${spk_emb_subset}-${spk_emb_model})(_long-filter=${filter_ratio})_batch-len=${batch_len}_model=${tts_model_path}:{
#      (type:block.BlockIterator,) or (type:abs.Iterator,)
#      conf:{
#        (dataset_type:speech_text.SpeechTextDataset,) or (dataset_type:speech_text.RandomSpkFeatDataset,)
#        dataset_conf:{
#          main_data:{
#            text:${unspoken_idx2text}
#          },
#          (spk_feat:${refer_idx2spk_feat},)
#          (data_selection:[min,${filter_ratio},${unspoken_idx2text_len}])
#        },
#        shuffle:false,
#        (data_len:${unspoken_idx2text_len},)
#        (batch_len:${batch_len})
#      }
#    }
#  }
# the following code does the same job as the configuration above
data_args="test:{seed=${random_seed}_"
# if 'long_filter' is true, attach it into the folder name
if ${long_filter};then
  data_args="${data_args}long-filter=${filter_ratio}_"
fi
# if 'spk_emb_model' is given, attach it into the folder name
if [ -n "${spk_emb_model}" ];then
  data_args="${data_args}spk-emb=${spk_emb_dataset}-${spk_emb_subset}-${spk_emb_model}_model=${tts_model_path}:{type:"
fi
# if 'batch_len' is given, block.BlockIterator will be used as the iterator; else, use abs.Iterator
if [ -n "${batch_len}" ];then
  data_args="${data_args}block.BlockIterator,"
else
  data_args="${data_args}abs.Iterator,"
fi
data_args="${data_args}conf:{"
# if 'spk_emb_model' is given, RandomSpkFeatDataset will be used as the Dataset; else, use SpeechTextDataset
if [ -n "${spk_emb_model}" ];then
  data_args="${data_args}dataset_type:speech_text.RandomSpkFeatDataset"
else
  data_args="${data_args}dataset_type:speech_text.SpeechTextDataset"
fi
data_args="${data_args},dataset_conf:{main_data:{text:"
unspoken_idx2text="datasets/${tts_syn_dataset}/data/${token_type}/${tts_syn_subset}/${token_num}/${txt_format}/idx2text"
unspoken_idx2text_len="${unspoken_idx2text}_len"
# if the idx2text file containing the split tokens doesn't exist, use the one containing the raw transcripts
if [ ! -f "${SPEECHAIN_ROOT}/${unspoken_idx2text}" ];then
  unspoken_idx2text="datasets/${tts_syn_dataset}/data/wav/${tts_syn_subset}/idx2${txt_format}_text"
fi
data_args="${data_args}${unspoken_idx2text}}"
# if 'spk_emb_model' is set to true, attach 'spk_feat' in 'dataset_conf'
if [ -n "${spk_emb_model}" ];then
  data_args="${data_args},spk_feat:datasets/${spk_emb_dataset}/data/wav/${spk_emb_subset}/idx2${spk_emb_model}_spk_feat"
fi
# if 'long_filter' is set to true, attach 'data_selection' in 'dataset_conf'
if ${long_filter};then
  data_args="${data_args},data_selection:[min,${filter_ratio},${unspoken_idx2text_len}]"
fi
data_args="${data_args}},shuffle:false,data_len:${unspoken_idx2text_len}"
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
# explicitly specify the used GPUs
if [ -n "${gpus}" ];then
  args="${args} --gpus ${gpus}"
fi
# explicitly specify the number of used GPUs
args="${args} --ngpu ${ngpu}"
# explicitly specify the number of worker processes for data loading
args="${args} --num_workers ${num_workers}"

# exp_cfg.yaml and train_cfg.yaml generated during training in ${tts_model_path} will be used
args="${args} --config ${tts_model_path}/exp_cfg.yaml --train_cfg ${tts_model_path}/train_cfg.yaml"
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