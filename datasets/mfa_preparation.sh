#  Author: Heli Qi
#  Affiliation: NAIST
#  Date: 2023.02


if [ -z "${SPEECHAIN_ROOT}" ] || [ -z "${SPEECHAIN_PYTHON}" ];then
  echo "Cannot find environmental variables SPEECHAIN_ROOT and SPEECHAIN_PYTHON.
  Please move to the root path of the toolkit and run envir_preparation.sh there!"
  exit 1
fi

function print_help_message {
  echo "usage:
  $0 \\ (The arguments in [] are optional while other arguments must be given by your run.sh.)
      [--src_path SRC_PATH] \\                              # The path of the dumped dataset. If src_path is not given, it will be initialized to ${SPEECHAIN_ROOT}/datasets/. If you have already dumped the dataset, please give its full path (starting by a slash '/') by this argument. (default: none)
      [--tgt_path TGT_PATH] \\                              # The mfa-related files will be saved to {tgt_path}/{dataset_name}/mfa. If tgt_path is not given, those files will be saved to {src_path}/{dataset_name}/mfa. If you want to save the files elsewhere, please give its full path (starting by a slash '/') by this argument. (default: none)
      [--pretrained PRETRAINED] \\                          # Whether to use the pretrained acoustic model for alignment. (default: true)
      [--ncpu NCPU] \\                                      # The number of processes used for the alignment jobs. (default: 8)
      --dataset_name DATASET_NAME                          # The name of the dataset you want to get the MFA files." >&2
  exit 1
}

src_path=${SPEECHAIN_ROOT}/datasets
tgt_path=
pretrained=true
ncpu=8
dataset_name=

### get args from the command line ###
while getopts ":h-:" optchar; do
  case "${optchar}" in
    -)
      case "${OPTARG}" in
        src_path)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          src_path=${val}
          ;;
        tgt_path)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          tgt_path=${val}
          ;;
        pretrained)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          pretrained=${val}
          ;;
        ncpu)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          ncpu=${val}
          ;;
        dataset_name)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          dataset_name=${val}
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

# --- Arguments Checking --- #
if [ -z ${tgt_path} ];then
   tgt_path=${src_path}
fi


# --- 1. Generate .TextGrid Files --- #
mkdir -p ${tgt_path}/${dataset_name}/data/mfa
# skip this step if the folder TextGrid exists
if [ ! -d ${tgt_path}/${dataset_name}/data/mfa/TextGrid ];then
  # produce the .TextGrid files locally by the mfa commands
  if [ ${dataset_name} == 'ljspeech' ];then
    corpus_path=${src_path}/${dataset_name}/data/wav16000
    lexicon_path="${SPEECHAIN_ROOT}"/datasets/mfa_lexicons/librispeech-lexicon.txt
    pretrained_model=english_us_arpa
  elif [ ${dataset_name} == 'libritts' ];then
    corpus_path=${src_path}/${dataset_name}/data/wav16000
    lexicon_path="${SPEECHAIN_ROOT}"/datasets/mfa_lexicons/librispeech-lexicon.txt
    pretrained_model=english_us_arpa
  else
    echo "Currently dataset_name could only be 'ljspeech' or 'libritts', but got ${dataset_name}!"
    exit 1
  fi

  if [ ! -d ${corpus_path} ];then
    echo "MFA is better to be performed on 16khz-downsampled dataset, but ${corpus_path} doens't exist!"
    if [ ${dataset_name} == 'ljspeech' ];then
      echo "Please go to ${SPEECHAIN_ROOT}/datasets/ljspeech and run the command below to dump wav16000/"
    elif [ ${dataset_name} == 'libritts' ];then
      echo "Please go to ${SPEECHAIN_ROOT}/datasets/libritts and run the command below to dump wav16000/"
    fi
    echo "bash run.sh --start_step 3 --stop_step 8 --sample_rate 16000 --txt_format asr"
    exit 1
  fi

  # Prepare .lab files
  ${SPEECHAIN_PYTHON} "${SPEECHAIN_ROOT}"/datasets/pyscripts/lab_file_generator.py \
    --dataset_path ${corpus_path} \
    --ncpu ${ncpu}

  # Create the aligner environment if there is no environment named aligner
  aligner_envir=$(conda env list | grep aligner)
  if [ -z "${aligner_envir}" ]; then
    conda create -n aligner -c conda-forge montreal-forced-aligner
    aligner_envir=$(conda env list | grep aligner)
  fi

  # Get the environment root path by conda
  read -ra aligner_envir <<< "${aligner_envir}"
  envir_path="${aligner_envir[$((${#aligner_envir[*]}-1))]}"

  # Get the mfa command path in the environment root
  mfa_command_path="${envir_path}"/bin/mfa

  if ${pretrained};then
    # Download the pretrained acoustic model for American English
    ${mfa_command_path} model download acoustic ${pretrained_model}

    # Generate .TextGrid files
    ${mfa_command_path} align \
      ${corpus_path} "${lexicon_path}" ${pretrained_model} ${tgt_path}/${dataset_name}/data/mfa/TextGrid \
      -j ${ncpu} --clean True
  else
    # train an acoustic model on the target corpus and use it to get the alignments
    ${mfa_command_path} train \
      ${corpus_path} "${lexicon_path}" \
      ${tgt_path}/${dataset_name}/data/mfa/acoustic_model.zip ${tgt_path}/${dataset_name}/data/mfa/TextGrid \
      -j ${ncpu} --clean True
  fi

else
  echo ".TextGrid files have already existed in ${tgt_path}/${dataset_name}/TextGrid!"
fi


# skip this step if the folder TextGrid doesn't exist
if [ -d ${tgt_path}/${dataset_name}/data/mfa/TextGrid ];then
  # --- Generate idx2duration.json by .TextGrid Files --- #
  ${SPEECHAIN_PYTHON} "${SPEECHAIN_ROOT}"/datasets/pyscripts/duration_calculator.py \
      --textgrid_path ${tgt_path}/${dataset_name}/data/mfa/TextGrid \
      --proc_dataset ${dataset_name}
else
  echo ".TextGrid files don't exist in ${tgt_path}/${dataset_name}/data/mfa/TextGrid. Please use mfa commands in the documents to prepare them and then run this script again."
  exit 1
fi
