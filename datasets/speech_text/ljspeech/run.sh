#  Author: Heli Qi
#  Affiliation: NAIST
#  Date: 2022.11

# --- Arguments --- #
# toolkit root, need to be changed to your own place
root=/ahc/work4/heli-qi/euterpe-heli-qi
# the dataset root for speech-text datasets, don't need to be changed
datatype_root=${root}/datasets/speech_text

# general arguments, their values are shared across different datasets
# execution-related arguments
start_step=1
stop_step=10000
ncpu=8
# acoustic feature-related arguments
feat_type=wav
feat_config=
sample_rate=
# tokenization-related arguments
token_type=char
vocab_size=
# subword-specific arguments, won't be used if token_type is 'char' or 'word'
subword_type=bpe
subword_package=sp
character_coverage=1.0
split_by_whitespace=true
# arguments used by data_download.sh
package_removal=false
# arguments used by stat_info_generator.py
separator=','
txt_format=normal


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
        token_type)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          token_type=${val}
          ;;
        vocab_size)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          vocab_size=${val}
          ;;
        subword_type)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          subword_type=${val}
          ;;
        subword_package)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          subword_package=${val}
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
        package_removal)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          package_removal=${val}
          ;;
        txt_format)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          txt_format=${val}
          ;;
        ?)
          echo "Unknown variable $OPTARG"
          ;;
      esac
      ;;
    h)
      echo "usage: $0 \\ (The arguments in [] are optional while other arguments must be given by your run.sh.)
      [--start_step START_STEP] \\                          # Which step you would like to start from. (default: 1)
      [--stop_step STOP_STEP] \\                            # Which step you would like to end at. (default: 10000)
      [--feat_type FEAT_TYPE] \\                            # The type of the feature you would like to dump. (default: wav)
      [--feat_config FEAT_CONFIG] \\                        # The name of acoustic feature extraction configuration file. (default: none)
      [--sample_rate SAMPLE_RATE] \\                        # The sampling rate you want the waveforms to have. (default: none)
      [--comp_chunk_ext COMP_CHUNK_EXT] \\                  # The file extension of the compressed chunk files. (default: none)
      [--token_type TOKEN_TYPE] \\                          # The type of the token you want your tokenizer to have. (default: subword)
      [--txt_format TXT_FORMAT] \\                          # The text processing format for the transcripts in the dataset. (default: normal)
      [--ncpu NCPU] \\                                      # The number of processes used for all the multiprocessing jobs. (default: 8)
      [--vocab_size VOCAB_SIZE] \\                          # The size of the tokenizer vocabulary. (default: 1000 for dump_part '100'; 5000 for dump_part '460'; 10000 for dump_part '960')
      [--subword_type SUBWORD_TYPE] \\                      # The type of the subword tokenizer model. (default: bpe)
      [--subword_package SUBWORD_PACKAGE] \\                # The package you would like to use to build the tokenizer model. (default: sp)
      [--character_coverage CHARACTER_COVERAGE] \\          # The character_coverage argument for building sp tokenizer model. (default: 1.0)
      [--split_by_whitespace SPLIT_BY_WHITESPACE] \\        # The split_by_whitespace argument for building sp tokenizer model. (default: true)
      [--separator SEPARATOR] \\                            # The separator used to separate the 'subsets' arguments from a string into an array of string. (default: ',')
      [--package_removal PACKAGE_REMOVAL] \\                # Whether to remove the downloaded data package after unzipping. (default: false)
      [--valid_section VALID_SECTION] \\                    # Which section of LJSpeech you want to use as the validation set. Each digit represent a section in LJSpeech (i.e., LJ0XX). Two consecutive digits are separated by a comma。 (default: '3')
      [--test_section TEST_SECTION]                        # Which section of LJSpeech you want to use as the test set. Each digit represent a section in LJSpeech (i.e., LJ0XX). Two consecutive digits are separated by a comma。 (default: '1,2')" >&2
      exit 1
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
# There is no official subsets of LJSpeech, so the names are simply set to 'train', 'valid', and 'test' here.
subsets="train valid test"
# enter extra arguments for vocab_generator.py
vocab_generate_args=
# number of tokens in the vocabulary
if [ -n "${vocab_size}" ]; then
  vocab_generate_args="--vocab_size ${vocab_size}"
fi
# subword-specific arguments
if [ ${token_type} == 'subword' ]; then
  vocab_generate_args="${vocab_generate_args} --subword_package ${subword_package} --subword_type ${subword_type} --character_coverage ${character_coverage} --split_by_whitespace ${split_by_whitespace}"
fi


# --- Data Dumping Execution --- #
${datatype_root}/data_dumping.sh \
  --start_step "${start_step}" \
  --stop_step "${stop_step}" \
  --feat_type "${feat_type}" \
  --feat_config "${feat_config}" \
  --sample_rate "${sample_rate}" \
  --token_type "${token_type}" \
  --txt_format "${txt_format}" \
  --dataset_name "ljspeech" \
  --download_args " --package_removal ${package_removal}" \
  --meta_generate_args "--valid_section ${valid_section} --test_section ${test_section} --separator ${separator}" \
  --subsets "${subsets}" \
  --vocab_src_subsets "train" \
  --vocab_tgt_subsets "train valid test" \
  --vocab_generate_args "${vocab_generate_args}" \
  --ncpu "${ncpu}"
