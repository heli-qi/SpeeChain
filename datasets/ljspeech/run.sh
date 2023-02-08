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
      [--src_path SRC_PATH] \\                              # If you already have the decompressed folder 'LJSpeech-1.1' downloaded from https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2, you can give its absolute path (starting by a slash '/', i.e. /xxx/xxx/LJSpeech-1.1) to this argument so that the data downloading step will be skipped. (default: none)
      [--tgt_path TGT_PATH] \\                              # The metadata files will be generated to {tgt_path}/ljspeech. If tgt_path is not given, metadata files will be saved to ${SPEECHAIN_ROOT}/datasets/ljspeech. If you want to save metadata files elsewhere, please give its absolute path (starting by a slash '/') by this argument. (default: none)
      [--feat_type FEAT_TYPE] \\                            # The type of the feature you would like to dump. (default: wav)
      [--feat_config FEAT_CONFIG] \\                        # The name of acoustic feature extraction configuration file. (default: none)
      [--sample_rate SAMPLE_RATE] \\                        # The sampling rate you want the waveforms to have. If not given, the original sampling rate of LJSpeech (22.05kHz) will be used for the folder 'ljspeech/data/wav' (default: none)
      [--spk_emb_model SPK_EMB_MODEL] \\                    # The speaker recognition model you want to use to extract the speaker embeddings. If given, this argument must be either 'xvector' or 'ecapa'. (default: none)
      [--comp_chunk_ext COMP_CHUNK_EXT] \\                  # The file extension of the compressed chunk files. (default: none)
      [--token_type TOKEN_TYPE] \\                          # The type of the token you want your tokenizer to have. (default: g2p)
      [--txt_format TXT_FORMAT] \\                          # The text processing format for the transcripts in the dataset. (default: tts)
      [--ncpu NCPU] \\                                      # The number of processes used for all the multiprocessing jobs. (default: 8)
      [--ngpu NGPU] \\                                      # The number of GPUs used to extract speaker embeddings. If not given, extraction will be done by CPUs. (default: 1)
      [--vocab_size VOCAB_SIZE] \\                          # The size of the tokenizer vocabulary. (default: 1000 for dump_part '100'; 5000 for dump_part '460'; 10000 for dump_part '960')
      [--model_type MODEL_TYPE] \\                          # The model_type argument for building sentencepiece tokenzier model. (default: bpe)
      [--character_coverage CHARACTER_COVERAGE] \\          # The character_coverage argument for building sp tokenizer model. (default: 1.0)
      [--split_by_whitespace SPLIT_BY_WHITESPACE] \\        # The split_by_whitespace argument for building sp tokenizer model. (default: true)
      [--separator SEPARATOR] \\                            # The separator used to separate the 'subsets' arguments from a string into an array of string. (default: ',')
      [--valid_section VALID_SECTION] \\                    # Which section of LJSpeech you want to use as the validation set. Each digit represent a section in LJSpeech (i.e., LJ0XX). Two consecutive digits are separated by a comma。 (default: '3')
      [--test_section TEST_SECTION]                        # Which section of LJSpeech you want to use as the test set. Each digit represent a section in LJSpeech (i.e., LJ0XX). Two consecutive digits are separated by a comma。 (default: '1,2')" >&2
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
# empty sample_rate means the sampling rate of the original LJSpeech (22.05kHz) will be used
sample_rate=
# empty spk_emb_model means no speaker embedding will be extracted
spk_emb_model=
# tokenization-related arguments
token_type=g2p
# empty vocab_size will be automatically initialized if token_type is 'word' or 'sentencepiece':
# 1000 for dump_part '100'; 5000 for dump_part '460'; 10000 for dump_part '960'
vocab_size=
# sentencepiece-specific arguments, won't be used if token_type is 'char' or 'word'
model_type=bpe
character_coverage=1.0
split_by_whitespace=true
# arguments used by stat_info_generator.py
separator=','
txt_format=tts


# LJSpeech-specific arguments
# which section of LJSpeech you want to use as the validation set
# Each digit represent a section in LJSpeech (i.e., LJ0XX). Two consecutive digits are separated by a comma
valid_section="3"
# which section of LJSpeech you want to use as the test set
test_section="1,2"


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
        valid_section)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          valid_section=${val}
          ;;
        test_section)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          test_section=${val}
          ;;
        separator)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          separator=${val}
          ;;
        txt_format)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          txt_format=${val}
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

# --- Argument Checking --- #
if [ ${separator} == ' ' ]; then
  echo "Your input separator cannot be a blank!"
  exit 1
fi


# --- Argument Initialization --- #
# There is no official subsets of LJSpeech, so the names are simply set to 'train', 'valid', and 'test' here.
subsets="train valid test"
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
  --dataset_name "ljspeech" \
  --meta_generate_args "--valid_section ${valid_section} --test_section ${test_section} --separator ${separator}" \
  --subsets "${subsets}" \
  --vocab_src_subsets "train valid test" \
  --vocab_generate_args "${vocab_generate_args}" \
  --ncpu "${ncpu}" \
  --ngpu "${ngpu}"
