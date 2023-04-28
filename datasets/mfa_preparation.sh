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
      [--pretrained PRETRAINED] \\                          # Whether to use the pretrained acoustic model for alignment. If pretrained is set to true but pretrained_dataset and pretrained_subset are not give, the pretrained English model provided by MFA toolkit will be used which is pretrained on LibriSpeech corpus. (default: true)
      [--pretrained_dataset PRETRAINED_DATASET] \\          # The dataset name which is used to train the pretrained model you want to use for alignment. (default: none)
      [--pretrained_subset PRETRAINED_SUBSET] \\            # The subset name of your given pretrained_dataset which your target pretrained model is trained on.  (default: none)
      [--retain_stress RETAIN_STRESS] \\                    # Whether to retain the stress indicators at the end of each vowel phonemes.  (default: true)
      [--ncpu NCPU] \\                                      # The number of processes used for the alignment jobs. (default: 8)
      --dataset_name DATASET_NAME \\                        # The name of the dataset you want to get the MFA files.
      [--subset_name SUBSET_NAME]                          # The name of the subset in your specified dataset you want to use. If not given, the entire dataset will be procesed. (default: none)" >&2
  exit 1
}

src_path=${SPEECHAIN_ROOT}/datasets
tgt_path=
pretrained=true
pretrained_dataset=
pretrained_subset=
retain_stress=true
ncpu=8
dataset_name=
subset_name=

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
        pretrained_dataset)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          pretrained_dataset=${val}
          ;;
        pretrained_subset)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          pretrained_subset=${val}
          ;;
        retain_stress)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          retain_stress=${val}
          ;;
        ncpu)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          ncpu=${val}
          ;;
        dataset_name)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          dataset_name=${val}
          ;;
        subset_name)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          subset_name=${val}
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

if [ -z ${dataset_name} ];then
   echo "dataset_name cannot be empty! Please give a dataset that you want to process."
   exit 1
fi

if [ -n "${pretrained_dataset}" ] && [ -n "${pretrained_subset}" ];then
  # path is used for mfa command while name is used for the file path
  pretrained_model_path=${tgt_path}/${pretrained_dataset}/data/mfa/models/${pretrained_subset}.zip
  pretrained_model_name=${pretrained_dataset}_${pretrained_subset}
elif [ -n "${pretrained_dataset}" ] || [ -n "${pretrained_subset}" ]; then
  echo "pretrained_dataset and pretrained_subset must be given at the same time!"
  exit 1
else
  pretrained_model_path=english_us_arpa
  pretrained_model_name=english_us_arpa
fi


# --- 1. Generate .TextGrid Files --- #
mkdir -p ${tgt_path}/${dataset_name}/data/mfa/${pretrained_model_name}
# skip this step if the folder TextGrid exists
if [ ! -d ${tgt_path}/${dataset_name}/data/mfa/${pretrained_model_name}/TextGrid/${subset_name} ];then
  # produce the .TextGrid files locally by the mfa commands
  if [ ${dataset_name} == 'ljspeech' ];then
    corpus_path=${src_path}/${dataset_name}/data/wav16000/${subset_name}
    lexicon_path="${SPEECHAIN_ROOT}"/datasets/mfa_lexicons/librispeech-lexicon.txt

  elif [ ${dataset_name} == 'libritts' ];then
    corpus_path=${src_path}/${dataset_name}/data/wav16000/${subset_name}
    lexicon_path="${SPEECHAIN_ROOT}"/datasets/mfa_lexicons/librispeech-lexicon.txt

  elif [ ${dataset_name} == 'librispeech' ];then
    corpus_path=${src_path}/${dataset_name}/data/wav/${subset_name}
    lexicon_path="${SPEECHAIN_ROOT}"/datasets/mfa_lexicons/librispeech-lexicon.txt

  else
    echo "Currently dataset_name could only be one of ['ljspeech', 'libritts'. 'librispeech'], but got ${dataset_name}!"
    exit 1
  fi

  if [ ! -d ${corpus_path} ];then
    echo "MFA is better to be performed on 16khz-downsampled dataset, but ${corpus_path} doens't exist!"
    if [ ${dataset_name} == 'ljspeech' ];then
      echo "Please go to ${SPEECHAIN_ROOT}/datasets/ljspeech and run the command below to dump wav16000/"
      echo "bash run.sh --stop_step 8 --sample_rate 16000 --txt_format asr"
    elif [ ${dataset_name} == 'libritts' ];then
      echo "Please go to ${SPEECHAIN_ROOT}/datasets/libritts and run the command below to dump wav16000/"
      echo "bash run.sh --stop_step 8 --sample_rate 16000 --txt_format asr"
    elif [ ${dataset_name} == 'librispeech' ];then
      echo "Please go to ${SPEECHAIN_ROOT}/datasets/librispeech and run the command below to dump wav/"
      echo "bash run.sh --stop_step 8 --txt_format asr"
    fi
    exit 1
  fi

  # Prepare .lab files
  ${SPEECHAIN_PYTHON} "${SPEECHAIN_ROOT}"/datasets/pyscripts/lab_file_generator.py \
    --corpus_path ${corpus_path} \
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
  mkdir -p ${tgt_path}/${dataset_name}/data/mfa/${pretrained_model_name}/TextGrid/${subset_name}
  if ${pretrained};then
    # Download the MFA-builtin pretrained acoustic model
    if [ ${pretrained_model_path} == 'english_us_arpa' ];then
      ${mfa_command_path} model download acoustic ${pretrained_model_path}
    fi

    # Generate .TextGrid files
    ${mfa_command_path} align \
      ${corpus_path} "${lexicon_path}" ${pretrained_model_path} \
      ${tgt_path}/${dataset_name}/data/mfa/${pretrained_model_name}/TextGrid/${subset_name} \
      -j ${ncpu} --clean True
  else
    mkdir -p ${tgt_path}/${dataset_name}/data/mfa/models
    # train an acoustic model on the target corpus and use it to get the alignments
    ${mfa_command_path} train \
      ${corpus_path} "${lexicon_path}" \
      ${tgt_path}/${dataset_name}/data/mfa/models/${subset_name}.zip \
      ${tgt_path}/${dataset_name}/data/mfa/${pretrained_model_name}/TextGrid/${subset_name} \
      -j ${ncpu} --clean True
  fi

else
  echo ".TextGrid files have already existed in ${tgt_path}/${dataset_name}/data/mfa/${pretrained_model_name}/TextGrid/${subset_name}!"
fi


# skip this step if the folder TextGrid doesn't exist
if [ -d ${tgt_path}/${dataset_name}/data/mfa/${pretrained_model_name}/TextGrid/${subset_name} ];then
  if [ -n "${subset_name}" ];then
    args="--subset_name ${subset_name}"
  else
    args=
  fi

  # --- Generate idx2duration.json by .TextGrid Files --- #
  ${SPEECHAIN_PYTHON} "${SPEECHAIN_ROOT}"/datasets/pyscripts/duration_calculator.py \
      --data_path ${tgt_path}/${dataset_name}/data \
      --save_path ${tgt_path}/${dataset_name}/data/mfa/${pretrained_model_name} \
      --pretrained_model_name ${pretrained_model_name} \
      --retain_stress ${retain_stress} \
      --dataset_name ${dataset_name} \
      ${args}
else
  echo ".TextGrid files don't exist in ${tgt_path}/${dataset_name}/data/mfa/TextGrid. Please use mfa commands in the documents to prepare them and then run this script again."
  exit 1
fi
