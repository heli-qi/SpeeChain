#  Author: Heli Qi
#  Affiliation: NAIST
#  Date: 2022.11

if [ -z "${SPEECHAIN_ROOT}" ];then
  echo "Cannot find environmental variable SPEECHAIN_ROOT.
  Please move to the root path of the toolkit and run envir_preparation.sh there!"
  exit 1
fi


function print_help_message {
  echo "usage:
  $0 \\ (The arguments in [] are optional while other arguments must be given by your run.sh.)
      [--start_step START_STEP] \\                          # Which step you would like to start from. (default: 1)
      [--stop_step STOP_STEP] \\                            # Which step you would like to end at. (default: 10000)
      [--src_path SRC_PATH] \\                              # The path of the existing LibriTTS dataset on your disk. If this argument is not given, LibriTTS will be downloaded to ${SPEECHAIN_ROOT}/datasets/libritts. If you have already downloaded the dataset, please give its absolute path (starting by a slash '/') by this argument. (default: none)
      [--tgt_path TGT_PATH] \\                              # The metadata files will be generated to {tgt_path}/libritts. If tgt_path is not given, metadata files will be saved to ${SPEECHAIN_ROOT}/datasets/libritts. If you want to save metadata files elsewhere, please give its absolute path (starting by a slash '/') by this argument. (default: none)
      [--feat_type FEAT_TYPE] \\                            # The type of the feature you would like to dump. (default: wav)
      [--feat_config FEAT_CONFIG] \\                        # The name of acoustic feature extraction configuration file. (default: none)
      [--sample_rate SAMPLE_RATE] \\                        # The sampling rate you want the waveforms to have. (default: none)
      [--spk_emb_model SPK_EMB_MODEL] \\                    # The speaker recognition model you want to use to extract the speaker embeddings. If given, this argument must be either 'xvector' or 'ecapa'. (default: none)
      [--comp_chunk_ext COMP_CHUNK_EXT] \\                  # The file extension of the compressed chunk files. (default: none)
      [--token_type TOKEN_TYPE] \\                          # The type of the token you want your tokenizer to have. (default: g2p)
      [--txt_format TXT_FORMAT] \\                          # The text processing format for the transcripts in the dataset. (default: normal)
      [--ncpu NCPU] \\                                      # The number of processes used for all the multiprocessing jobs. (default: 8)
      [--ngpu NGPU] \\                                      # The number of GPUs used to extract speaker embeddings. If not given, extraction will be done by CPUs. (default: 1)
      [--vocab_size VOCAB_SIZE] \\                          # The size of the tokenizer vocabulary. (default: none)
      [--model_type MODEL_TYPE] \\                          # The model_type argument for building sentencepiece tokenzier model. (default: bpe)
      [--character_coverage CHARACTER_COVERAGE] \\          # The character_coverage argument for building sp tokenizer model. (default: 1.0)
      [--split_by_whitespace SPLIT_BY_WHITESPACE] \\        # The split_by_whitespace argument for building sp tokenizer model. (default: true)
      [--separator SEPARATOR] \\                            # The separator used to separate the 'subsets' arguments from a string into an array of string. (default: ',')
      [--package_removal PACKAGE_REMOVAL] \\                # Whether to remove the downloaded data package after unzipping. (default: false)
      [--dump_part DUMP_PART]                              # Which part of LibriTTS you would like to dump. '100' means 'train-clean-100'; '460' means 'train-clean-100' + 'train-clean-360'; '960' means 'train-clean-100' + 'train-clean-360' + 'train-other-500'. 'dev-clean', 'dev-other', 'test-clean', 'test-other' will be dumped for all options. (default: '960')" >&2
  exit 1
}


# --- Arguments --- #
# the dataset root for speech-text datasets, don't need to be changed
datatype_root=${SPEECHAIN_ROOT}/datasets

# general arguments, their values are shared across different datasets
# execution-related arguments
start_step=1
stop_step=10000
src_path=
tgt_path=
ncpu=8
ngpu=1
# acoustic feature-related arguments
feat_type=wav
# empty feat_config means no feature extraction configuration
feat_config=
# empty sample_rate means the sampling rate of the original LibriTTS (24kHz) will be used
sample_rate=
# empty spk_emb_model means no speaker embedding will be extracted
spk_emb_model=
# tokenizer for LibriTTS is default to be g2p
token_type=g2p
# empty vocab_size will be automatically initialized if token_type is 'word' or 'sentencepiece':
# 1000 for dump_part '100'; 5000 for dump_part '460'; 10000 for dump_part '960'
vocab_size=
# sentencepiece-specific arguments, won't be used if token_type is not 'sentencepiece'
model_type=bpe
character_coverage=1.0
split_by_whitespace=true
# arguments used by data_download.sh
separator=','
package_removal=false
# text format for LibriTTS is default to be normal
txt_format=normal


# LibriTTS-specific arguments
# which portion of LibriTTS corpus you want to process
# 100: train-clean-100; 460: train-clean-100 + train-clean-360; 960: train-clean-100 + train-clean-360 + train-other-500
# dev-clean, dev-other, test-clean, test-other will be downloaded regardless of your input proc_portion
proc_portion=960


### get args from the command line ###
while getopts ":h-:" optchar; do
  case "${optchar}" in
    -)
      case "${OPTARG}" in
        start_step)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          start_step=${val}
          ;;
        stop_step)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          stop_step=${val}
          ;;
        src_path)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          src_path=${val}
          ;;
        tgt_path)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          tgt_path=${val}
          ;;
        feat_type)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          feat_type=${val}
          ;;
        feat_config)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          feat_config=${val}
          ;;
        sample_rate)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          sample_rate=${val}
          ;;
        spk_emb_model)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          spk_emb_model=${val}
          ;;
        token_type)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          token_type=${val}
          ;;
        vocab_size)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          vocab_size=${val}
          ;;
        model_type)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          model_type=${val}
          ;;
        character_coverage)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          character_coverage=${val}
          ;;
        split_by_whitespace)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          split_by_whitespace=${val}
          ;;
        ncpu)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          ncpu=${val}
          ;;
        ngpu)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          ngpu=${val}
          ;;
        proc_portion)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          proc_portion=${val}
          ;;
        separator)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          separator=${val}
          ;;
        package_removal)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          package_removal=${val}
          ;;
        txt_format)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          txt_format=${val}
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

# --- Argument Checking --- #
if [ ${separator} == ' ' ]; then
  echo "Your input separator cannot be a blank!"
  exit 1
fi


# --- Argument Initialization --- #
# only generate the vocabulary for the training sets (vocab_subsets)
case "${proc_portion}" in
  100)
    subsets="train-clean-100"
    subsets_args="train-clean-100"
    vocab_src_subsets="${subsets} dev-clean dev-other dev test-clean test-other"

    if [ ${token_type} == 'word' ] || [ ${token_type} == 'sentencepiece' ]; then
      if [ -z ${vocab_size} ];then
        vocab_size=1000
      fi
    fi
    ;;
  460)
    subsets="train-clean-100 train-clean-360"
    subsets_args="train-clean-100${separator}train-clean-360"
    vocab_src_subsets="${subsets} train-clean-460 dev-clean dev-other dev test-clean test-other"

    if [ ${token_type} == 'word' ] || [ ${token_type} == 'sentencepiece' ]; then
      if [ -z ${vocab_size} ];then
        vocab_size=5000
      fi
    fi
    ;;
  960)
    subsets="train-clean-100 train-clean-360 train-other-500"
    subsets_args="train-clean-100${separator}train-clean-360${separator}train-other-500"
    vocab_src_subsets="${subsets} train-clean-460 train-960 dev-clean dev-other dev test-clean test-other"

    if [ ${token_type} == 'word' ] || [ ${token_type} == 'sentencepiece' ]; then
      if [ -z ${vocab_size} ];then
        vocab_size=5000
      fi
    fi
    ;;
  ?)
    echo "Your input proc_portion '${proc_portion}' is invalid. It must be one of [100, 460, 960]!"
    exit 1
esac

# attach dev and test sets to 'subsets' arguments for any input proc_portion
subsets="${subsets} dev-clean dev-other test-clean test-other"
subsets_args="${subsets_args}${separator}dev-clean${separator}dev-other${separator}test-clean${separator}test-other"
# enter extra arguments for vocab_generator.py
vocab_generate_args=
# number of tokens in the vocabulary
if [ -n "${vocab_size}" ]; then
  vocab_generate_args="--vocab_size ${vocab_size}"
fi
# sentencepiece-specific arguments
if [ ${token_type} == 'sentencepiece' ]; then
  vocab_generate_args="${vocab_generate_args} --model_type ${model_type} --character_coverage ${character_coverage} --split_by_whitespace ${split_by_whitespace}"
fi


# --- Data Dumping Execution --- #
"${datatype_root}"/data_dumping.sh \
  --start_step "${start_step}" \
  --stop_step "${stop_step}" \
  --src_path "${src_path}" \
  --tgt_path "${tgt_path}" \
  --feat_type "${feat_type}" \
  --feat_config "${feat_config}" \
  --sample_rate "${sample_rate}" \
  --spk_emb_model "${spk_emb_model}" \
  --token_type "${token_type}" \
  --txt_format "${txt_format}" \
  --dataset_name "libritts" \
  --download_args "--subset ${subsets_args} --separator ${separator} --package_removal ${package_removal}" \
  --meta_generate_args "--subset ${subsets_args} --separator ${separator} --ncpu ${ncpu}" \
  --subsets "${subsets}" \
  --vocab_src_subsets "${vocab_src_subsets}" \
  --vocab_generate_args "${vocab_generate_args}" \
  --ncpu "${ncpu}" \
  --ngpu "${ngpu}"
