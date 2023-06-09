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
      [--lab_cover_flag LAB_COVER_FLAG] \\                  # Whether to cover the old .lab files. (default: false)
      [--pretrained PRETRAINED] \\                          # Whether to use the pretrained acoustic model for alignment. If pretrained is set to true but pretrained_dataset and pretrained_subset are not give, the default MFA pretrained English model (english_us_arpa) will be used which is pretrained on LibriSpeech corpus. (default: true)
      [--pretrained_dataset PRETRAINED_DATASET] \\          # The dataset name which is used to train the pretrained model you want to use for alignment. (default: none)
      [--pretrained_subset PRETRAINED_SUBSET] \\            # The subset name of your given pretrained_dataset which your target pretrained model is trained on.  (default: none)
      [--lexicon LEXICON] \\                                # The lexicon used by the acoustic model to extract phonemes from words. If not given, the default MFA lexicon (english_us_arpa) will be used. (default: librispeech)
      [--retain_punc RETAIN_PUNC] \\                        # Whether to retain the punctuation marks in raw sentences.  (default: false)
      [--retain_stress RETAIN_STRESS] \\                    # Whether to retain the stress indicators at the end of each vowel phonemes.  (default: true)
      [--ncpu NCPU] \\                                      # The number of processes used for the alignment jobs. (default: 8)
      [--single_speaker SINGLE_SPEAKER] \\                  # Whether to assign '--single_speaker' to 'mfa align'. If you want to process single-speaker dataset like LJSpeech, please turn it on. (default: false)
      --dataset_name DATASET_NAME \\                        # The name of the dataset you want to get the MFA files.
      [--subset_name SUBSET_NAME]                          # The name of the subset in your specified dataset you want to use. If not given, the entire dataset will be procesed. (default: none)" >&2
  exit 1
}

src_path=${SPEECHAIN_ROOT}/datasets
tgt_path=
lab_cover_flag=false
pretrained=true
pretrained_dataset=
pretrained_subset=
retain_punc=false
retain_stress=true
lexicon=librispeech
ncpu=8
single_speaker=false
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
        lab_cover_flag)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          lab_cover_flag=${val}
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
        retain_punc)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          retain_punc=${val}
          ;;
        retain_stress)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          retain_stress=${val}
          ;;
        lexicon)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          lexicon=${val}
          ;;
        ncpu)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          ncpu=${val}
          ;;
        single_speaker)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          single_speaker=${val}
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
  pretrained_model_name=${pretrained_dataset}_${pretrained_subset}
  pretrained_model_path=${src_path}/mfa_models/${pretrained_model_name}.zip
  if [ ! -f "${pretrained_model_path}" ];then
    if [ ${pretrained_model_name} == 'librispeech_train-clean-100' ];then
      gdown -O ${src_path}/mfa_models/ https://drive.google.com/uc?id=11x_gX4H_jRIR4QAadR32GYM7V9JMRNlB
    fi
  fi

elif [ -n "${pretrained_dataset}" ] || [ -n "${pretrained_subset}" ]; then
  echo "pretrained_dataset and pretrained_subset must be given at the same time!"
  exit 1
else
  pretrained_model_name=english_us_arpa
  pretrained_model_path=english_us_arpa
fi

if [ -z ${lexicon} ] || [ "${lexicon}" == 'english_us_arpa' ];then
  lexicon_path=english_us_arpa
elif [ "${lexicon}" == 'librispeech' ];then
  lexicon_path=${src_path}/mfa_lexicons/librispeech-lexicon.txt
  if [ ! -f "${lexicon_path}" ]; then
    mkdir -p ${src_path}/mfa_lexicons
    gdown -O ${src_path}/mfa_lexicons/ https://drive.google.com/uc?id=1v5EYiR7QJOYgG8yj4uVZ-zInZPYbRCCE
  fi
else
  echo "Unknown lexicon ${lexicon}. Please give 'librispeech' or 'english_us_arpa'."
  exit 1
fi


# --- 1. Generate .TextGrid Files --- #
save_folder_name="acoustic=${pretrained_model_name}_lexicon=${lexicon}"
mkdir -p ${tgt_path}/${dataset_name}/data/mfa/${save_folder_name}
textgrid_path=${tgt_path}/${dataset_name}/data/mfa/${save_folder_name}/TextGrid/${subset_name}

# skip this step if the folder TextGrid exists
if [ ! -d ${textgrid_path} ] || [ -z "$(ls -A ${textgrid_path})" ];then
  # produce the .TextGrid files locally by the mfa commands
  corpus_path=${src_path}/${dataset_name}/data/wav16000/${subset_name}

  if [ ! -d ${corpus_path} ];then
    echo "In SpeeChain, MFA duration calculation is performed on 16khz-downsampled dataset, but ${corpus_path} doens't exist!
          We do this by the following two reasons:
          1. The acoustic models of MFA accept 16khz audio data.
          2. Avoid to dump .lab files to the original dataset which may be shared by many users in your server."
    exit 1
  fi

  # Prepare .lab files
  ${SPEECHAIN_PYTHON} "${SPEECHAIN_ROOT}"/datasets/pyscripts/lab_file_generator.py \
    --corpus_path ${corpus_path} \
    --lab_cover_flag ${lab_cover_flag} \
    --ncpu ${ncpu}

  # Create the aligner environment if there is no environment named speechain_mfa
  aligner_envir=$(conda env list | grep speechain_mfa)
  if [ -z "${aligner_envir}" ]; then
    conda create -n speechain_mfa -c conda-forge montreal-forced-aligner gdown
    aligner_envir=$(conda env list | grep speechain_mfa)
  fi

  if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != 'speechain_mfa' ];then
    echo "Please activate the virtual environment 'speechain_mfa' by 'conda activate speechain_mfa' and then run the command you have just entered again."
    exit 1
  fi

  # Get the environment root path by conda
  read -ra aligner_envir <<< "${aligner_envir}"
  envir_path="${aligner_envir[$((${#aligner_envir[*]}-1))]}"

  # Get the mfa command path in the environment root
  mfa_command_path="${envir_path}"/bin/mfa
  if ${pretrained};then
    # Download the MFA default pretrained acoustic model
    if [ ${pretrained_model_path} == 'english_us_arpa' ];then
      ${mfa_command_path} model download acoustic ${pretrained_model_path}
    fi

    # Download the MFA default lexicon dictionary
    if [ ${lexicon_path} == 'english_us_arpa' ];then
      ${mfa_command_path} model download dictionary ${lexicon_path}
    fi

    align_args=
    if ${single_speaker};then
      align_args='--single_speaker True'
    fi

    # Generate .TextGrid files
    mkdir -p ${textgrid_path}
    ${mfa_command_path} align \
      ${corpus_path} "${lexicon_path}" ${pretrained_model_path} ${textgrid_path} \
      -j ${ncpu} --clean True ${align_args}
  else
    # train an acoustic model on the target corpus and use it to get the alignments
    ${mfa_command_path} train \
      ${corpus_path} "${lexicon_path}" \
      ${src_path}/mfa_models/${pretrained_model_name}.zip \
      ${textgrid_path} \
      -j ${ncpu} --clean True
  fi

else
  echo ".TextGrid files have already existed in ${textgrid_path}!"
fi

# skip this step if the folder TextGrid doesn't exist
if [ -d ${textgrid_path} ] && [ -n "$(ls -A ${textgrid_path})" ];then
  if [ -n "${subset_name}" ];then
    args="--subset_name ${subset_name}"
  else
    args=
  fi

  # --- Generate idx2duration.json by .TextGrid Files --- #
  ${SPEECHAIN_PYTHON} "${SPEECHAIN_ROOT}"/datasets/pyscripts/duration_calculator.py \
      --data_path ${tgt_path}/${dataset_name}/data \
      --save_path ${tgt_path}/${dataset_name}/data/mfa/${save_folder_name} \
      --save_folder_name ${save_folder_name} \
      --retain_punc ${retain_punc} \
      --retain_stress ${retain_stress} \
      --dataset_name ${dataset_name} \
      ${args}
else
  echo ".TextGrid files don't exist in ${tgt_path}/${dataset_name}/data/mfa/TextGrid. Please use mfa commands in the documents to prepare them and then run this script again."
  exit 1
fi
