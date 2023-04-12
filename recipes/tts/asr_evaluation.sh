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

      # Group1: ASR Decoding Environment
      [--batch_len BATCH_LEN] \\                            # The total length of all the synthetic utterances in a single batch to conduct batch-level ASR decoding. We recommend you to set 'batch_len' up to {1000 * beam number * total GBs of your GPUs}. (default: none)
      [--ngpu NGPU] \\                                      # The number of GPUs you want to use. If not given, ngpu in {asr_model_path}/exp_cfg.yaml will be used. (default: none)
      [--gpus GPUS] \\                                      # The GPUs you want to specify. If not given, gpus in {asr_model_path}/exp_cfg.yaml will be used. (default: none)
      [--num_workers NUM_WORKERS] \\                        # The name of worker processes for data loading. If not given, num_workers in {asr_model_path}/exp_cfg.yaml will be used. (default: none)
      [--resume RESUME] \\                                  # Whether to continue the unfinished evaluation. If true, the data loading strategy should remain the same as the last time. (default: false)

      # Group2: Long Synthetic Utterance Filtering
      [--long_filter LONG_FILTER] \\                        # Whether to filter out long utterances with the largest wav_len. (default: false)
      [--filter_ratio FILTER_RATIO] \\                      # How many shorter utterances you want to retain. (default: 0.95)

      # Group3: Main Arguments for ASR Evaluation
      [--vocoder VOCODER] \\                                # The vocoder you used to generate the waveforms. (default: gl)
      [--asr_infer_cfg ASR_INFER_CFG] \\                    # The configuration for ASR inference. If not given, infer_cfg in {asr_model_path}/exp_cfg.yaml will be used. (default: none)
      --asr_model_path ASR_MODEL_PATH \\                    # The path of the ASR model you want to use. There must be 'models/', 'exp_cfg.yaml', and 'train_cfg.yaml' in your specified folder.
      --tts_result_path TTS_RESULT_PATH \\                  # The path where you want to place the evaluation results. The metadata files named 'idx2{vocoder}_wav' and 'idx2{vocoder}_wav_len' in your given 'tts_result_path' will be used for ASR evaluation.
      --asr_refer_idx2text ASR_REFER_IDX2TEXT \\            # The idx2text file of the ground-truth text data." >&2
  exit 1
}

ngpu=
gpus=
batch_len=
resume=false
num_workers=

long_filter=false
filter_ratio=0.95

vocoder=gl
asr_infer_cfg=
asr_model_path=
tts_result_path=
asr_refer_idx2text=



### get args from the command line ###
while getopts ":h-:" optchar; do
  case "${optchar}" in
    -)
      case "${OPTARG}" in
        tts_result_path)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          tts_result_path=${val}
          ;;
        asr_refer_idx2text)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          asr_refer_idx2text=${val}
          ;;
        asr_model_path)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          asr_model_path=${val}
          ;;
        asr_infer_cfg)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          asr_infer_cfg=${val}
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
        vocoder)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          vocoder=${val}
          ;;
        batch_len)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          batch_len=${val}
          ;;
        num_workers)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          num_workers=${val}
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
if [ -z ${asr_model_path} ];then
   echo "Please give the path of the ASR model you want to use by '--asr_model_path'!"
   exit 1
fi

if ! grep -q '/' <<< "${asr_model_path}";then
  echo "For asr_cfg_cfg, please give either its absolute address or in-toolkit relative path!"
  exit 1
fi

if [ -z ${tts_result_path} ];then
   echo "Please give the path of your target TTS model by '--tts_result_path'!"
   exit 1
fi

if ! grep -q '/' <<< "${tts_result_path}";then
  echo "For tts_result_path, please give either its absolute address or in-toolkit relative path!"
  exit 1
fi

if [ -z ${asr_refer_idx2text} ];then
   echo "Please give the path of the idx2text file for the reference text by '--asr_refer_idx2text'!"
   exit 1
fi


# --- 1. Argument Initialization --- #
args="--train False --test True --train_result_path ${asr_model_path} --attach_config_folder_to_path false --attach_model_folder_when_test false --test_result_path ${tts_result_path}"

# automatically initialize the data loading configuration. The contents surrounded by a pair of brackets are optional.
#  test:{
#    ${vocoder}_${asr_model_path}:{
#      (type:block.BlockIterator,) or (type:abs.Iterator,)
#      conf:{
#        dataset_type:speech_text.SpeechTextDataset,
#        dataset_conf:{
#          main_data:{
#            feat:${tts_result_path}/idx2${vocoder}_wav,
#            text:${asr_refer_idx2text}
#          },
#          (data_selection:[min,${filter_ratio},${tts_result_path}/idx2${vocoder}_wav_len])
#        },
#        shuffle:false,
#        data_len:${tts_result_path}/idx2${vocoder}_wav_len,
#        (batch_len:${batch_len},)
#        (group_info:{speaker:${tts_result_path}/idx2ref_spk})
#      }
#    }
#  }
# the following code does the same job as the configuration above
data_args="test:{vocoder=${vocoder}"
# if 'long_filter' is true, attach filter_ratio into the folder name
if ${long_filter};then
  data_args="${data_args}_long-filter=${filter_ratio}"
fi
data_args="${data_args}_model=${asr_model_path}:{"

# if 'batch_len' is given, block.BlockIterator will be used as the iterator; else, use abs.Iterator
if [ -n "${batch_len}" ];then
  data_args="${data_args}type:block.BlockIterator,"
else
  data_args="${data_args}type:abs.Iterator,"
fi
data_args="${data_args}conf:{dataset_type:speech_text.SpeechTextDataset,dataset_conf:{main_data:{feat:${tts_result_path}/idx2${vocoder}_wav,text:${asr_refer_idx2text}}"
# if 'long_filter' is set to true, attach 'data_selection' in 'dataset_conf'
if ${long_filter};then
  data_args="${data_args},data_selection:[min,${filter_ratio},${tts_result_path}/idx2${vocoder}_wav_len]"
fi
data_args="${data_args}},shuffle:false,data_len:${tts_result_path}/idx2${vocoder}_wav_len"
# if 'batch_len' is given, attach 'batch_len'
if [ -n "${batch_len}" ];then
  data_args="${data_args},batch_len:${batch_len}"
fi

# include idx2ref_spk into group_info to evaluate the speaker-wise performance
data_args="${data_args},group_info:{speaker:${tts_result_path}/idx2ref_spk}}}}"
#
args="${args} --data_cfg ${data_args}"

#
if [ -n "${gpus}" ];then
  args="${args} --gpus ${gpus}"
fi
#
if [ -n "${ngpu}" ];then
  args="${args} --ngpu ${ngpu}"
fi
#
if [ -n "${num_workers}" ];then
  args="${args} --num_workers ${num_workers}"
fi

#
if [ -n "${asr_model_path}" ];then
  args="${args} --config ${asr_model_path}/exp_cfg.yaml --train_cfg ${asr_model_path}/train_cfg.yaml"
fi
#
if [ -n "${asr_infer_cfg}" ];then
  # do sth when infer_cfg is the name of a configuration file
  if ! grep -q ':' <<< "${asr_infer_cfg}";then
    # attach .yaml suffix if needed
    if [[ "${asr_infer_cfg}" != *".yaml" ]];then
      asr_infer_cfg="${asr_infer_cfg}.yaml"
    fi
    # convert the relative path in ${infer_root}/${task} if no colon inside
    if ! grep -q ':' <<< "${asr_infer_cfg}";then
      asr_infer_cfg="${SPEECHAIN_ROOT}/config/infer/asr/${asr_infer_cfg}"
    fi
  fi
  args="${args} --infer_cfg ${asr_infer_cfg}"
fi

#
if ${resume};then
  args="${args} --resume True"
else
  args="${args} --resume False"
fi


# --- 3. Execute the Job --- #
# ${args} should not be surrounded by double-quote
# shellcheck disable=SC2086
${SPEECHAIN_PYTHON} "${SPEECHAIN_ROOT}"/speechain/runner.py ${args}