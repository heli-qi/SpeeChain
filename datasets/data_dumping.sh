#  Author: Heli Qi
#  Affiliation: NAIST
#  Date: 2022.07


if [ -z "${SPEECHAIN_ROOT}" ] || [ -z "${SPEECHAIN_PYTHON}" ];then
  echo "Cannot find environmental variables SPEECHAIN_ROOT and SPEECHAIN_PYTHON.
  Please move to the root path of the toolkit and run envir_preparation.sh there!"
  exit 1
fi

function print_help_message {
  echo "usage:
  $0 \\ (The arguments in [] are optional while other arguments must be given by your run.sh.)
      [--start_step START_STEP] \\                          # Which step you would like to start from. (default: 1)
      [--stop_step STOP_STEP] \\                            # Which step you would like to end at. (default: 10000)
      [--feat_type FEAT_TYPE] \\                            # The type of the feature you would like to dump. If given, its value should match the name of the folder in ${SPEECHAIN_ROOT}/config/feat/ (default: wav)
      [--feat_config FEAT_CONFIG] \\                        # The name of acoustic feature extraction configuration file under ${SPEECHAIN_ROOT}/config/feat/{feat_type}/. (default: none)
      [--sample_rate SAMPLE_RATE] \\                        # The sampling rate you want the waveforms to have. (default: none)
      [--spk_emb_model SPK_EMB_MODEL] \\                    # The model you want to use to extractor speaker embeddings from the raw waveforms. (default: none)
      [--comp_chunk_ext COMP_CHUNK_EXT] \\                  # The file extension of the compressed chunk files. (default: none)
      [--token_type TOKEN_TYPE] \\                          # The type of the token you want your tokenizer to have. (default: char)
      [--txt_format TXT_FORMAT] \\                          # The text processing format for the transcripts in the dataset. (default: normal)
      [--ncpu NCPU] \\                                      # The number of processes used for all the multiprocessing jobs. (default: 8)
      [--ngpu NGPU] \\                                      # The number of GPUs used only for speaker embedding extraction. (default: 0)
      --dataset_name DATASET_NAME \\                        # The name of the dataset folder you want to dump.
      --subsets SUBSETS \\                                  # A blank-separated string which defines the subsets of the target dataset you want to dump
      --download_args DOWNLOAD_ARGS \\                      # The arguments you want to give to the data_download.sh.
      --meta_generate_args META_GENERATE_ARGS \\            # The arguments you want to give to the meta_generator.sh.
      --meta_post_process_args META_POST_PROCESS_ARGS \\    # The arguments you want to give to the meta_post_processor.sh.
      --vocab_src_subsets VOCAB_SRC_SUBSETS \\              # The source subsets for generating the tokenizer vocabulary.
      --vocab_generate_args VOCAB_GENERATE_ARGS            # The arguments you want to give to vocab_generator.py" >&2
  exit 1
}


# --- Absolute Path References --- #
# the following path don't need to be changed #
# the absolute path of the toolkit main folder
pyscript_root=${SPEECHAIN_ROOT}/datasets/pyscripts
# the absolute path of the toolkit config folder
config_root=${SPEECHAIN_ROOT}/config/feat
# the absolute path of the current dataset folder
data_root=${SPEECHAIN_ROOT}/datasets



# --- Arguments --- #
# general arguments, their values are shared across different datasets
start_step=1
stop_step=10000
dataset_name=
feat_type=wav
feat_config=
sample_rate=
spk_emb_model=
comp_chunk_ext=
token_type=char
txt_format=normal
ncpu=8
ngpu=

# specific arguments, their values need to be designed for each dataset
# the additional arguments for data downloading in step1
download_args=
# the additional arguments for statistics generation in step2
meta_generate_args=
# the name of the subsets used for step3, step4, and step5
subsets=
# the additional arguments that you want to give to stat_post_processor.py in step6
meta_post_process_args=
# the name of the source and target subsets used for vocabulary generation in step7
vocab_src_subsets=
# the additional arguments that you want to give to vocab_generator.py for customized token vocabulary generation
vocab_generate_args=

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
        spk_emb_model)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          spk_emb_model=${val}
          ;;
        comp_chunk_ext)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          comp_chunk_ext=${val}
          ;;
        token_type)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          token_type=${val}
          ;;
        txt_format)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          txt_format=${val}
          ;;
        ncpu)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          ncpu=${val}
          ;;
        ngpu)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          ngpu=${val}
          ;;
        dataset_name)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          dataset_name=${val}
          ;;
        download_args)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          download_args=${val}
          ;;
        meta_generate_args)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          meta_generate_args=${val}
          ;;
        subsets)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          subsets=${val}
          ;;
        meta_post_process_args)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          meta_post_process_args=${val}
          ;;
        vocab_src_subsets)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          vocab_src_subsets=${val}
          ;;
        vocab_generate_args)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          vocab_generate_args=${val}
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

# --- Arguments Checking --- #
if [ ${start_step} \> ${stop_step} ]; then
   echo "start_step should not be greater than stop_step!"
   exit 1
fi

if [ -z ${dataset_name} ];then
   echo "Please enter the name of your target dataset by '--dataset_name'!"
   exit 1
fi

if [ ${stop_step} -ge 3 ] && [ -z "${subsets}" ];then
   echo "Please enter the subsets you want to process by '--subsets'!"
   exit 1
fi

if [ ${stop_step} -ge 8 ] && [ -z "${token_type}" ];then
   echo "Please enter the token type you want to use by '--token_type'!"
   exit 1
fi


# --- Step1: Data Downloading from the Internet --- #
if [ ${start_step} -le 1 ] && [ ${stop_step} -ge 1 ]; then
  echo "Downloading the dataset from the Internet to ${data_root}/${dataset_name}/"
  "${data_root}"/${dataset_name}/data_download.sh \
    --download_path "${data_root}"/${dataset_name} \
    ${download_args}
fi


# --- Step2: Meta Data Generation --- #
if [ ${start_step} -le 2 ] && [ ${stop_step} -ge 2 ]; then
  echo "Generate the statistical information of the dataset in ${data_root}/${dataset_name}/data/wav"
  ${SPEECHAIN_PYTHON} "${data_root}"/${dataset_name}/meta_generator.py \
    --src_path "${data_root}"/${dataset_name}/data/wav \
    --txt_format ${txt_format} \
    ${meta_generate_args}
fi


# --- Step3: Downsampling the audio files (optional) --- #
if [ -n "${sample_rate}" ] && [ ${start_step} -le 3 ] && [ ${stop_step} -ge 3 ]; then
  mkdir -p "${data_root}"/${dataset_name}/data/wav${sample_rate}
  for set in ${subsets}; do
    echo "Downsampling the audio files in ${data_root}/${dataset_name}/data/wav/${set}/idx2wav to ${data_root}/${dataset_name}/data/wav${sample_rate}/${set}."
    ${SPEECHAIN_PYTHON} "${pyscript_root}"/wave_downsampler.py \
      --sample_rate ${sample_rate} \
      --src_file "${data_root}"/${dataset_name}/data/wav/${set}/idx2wav \
      --tgt_path "${data_root}"/${dataset_name}/data/wav${sample_rate}/${set} \
      --ncpu ${ncpu}
  done
fi


# --- Step4: Acoustic Feature Extraction (optional) --- #
if [ ${feat_type} != 'wav' ] && [ ${start_step} -le 4 ] && [ ${stop_step} -ge 4 ]; then
  if [ -z "${feat_config}" ];then
   echo "Please enter a config file name if your entered '--feat_type' is not 'wav'!"
   exit 1
  fi

  for set in ${subsets}; do
    echo "Generating acoustic features from ${data_root}/${dataset_name}/data/wav${sample_rate}/${set} to ${data_root}/${dataset_name}/data/${feat_type}/${feat_config}/${set}"
    mkdir -p "${data_root}"/${dataset_name}/data/${feat_type}/${feat_config}/${set}

    ${SPEECHAIN_PYTHON} "${pyscript_root}"/feat_extractor.py \
      --idx2wav "${data_root}"/${dataset_name}/data/wav${sample_rate}/${set}/idx2wav \
      --feat_type ${feat_type} \
      --feat_config "${config_root}"/${feat_type}/${feat_config}.yaml \
      --feat_path "${data_root}"/${dataset_name}/data/${feat_type}/${feat_config}/${set} \
      --ncpu ${ncpu}
  done
fi


# --- Step5: Acoustic Feature Length Generation --- #
if [ ${start_step} -le 5 ] && [ ${stop_step} -ge 5 ]; then
  if [ ${feat_type} != 'wav' ]; then
    folder_name=${feat_config}
  else
    folder_name=${feat_type}${sample_rate}
  fi

  for set in ${subsets}; do
    echo "Generating data lengths from ${data_root}/${dataset_name}/data/${folder_name}/${set}/idx2${feat_type}
    to ${data_root}/${dataset_name}/data/${folder_name}/${set}/idx2${feat_type}_len"

    ${SPEECHAIN_PYTHON} "${pyscript_root}"/data_len_generator.py \
      --src_file "${data_root}"/${dataset_name}/data/${folder_name}/${set}/idx2${feat_type} \
      --ncpu ${ncpu}
  done
fi


# --- Step6: Speaker Embedding Extraction --- #
if [ -n "${spk_emb_model}" ] && [ ${feat_type} == 'wav' ] && [ ${start_step} -le 6 ] && [ ${stop_step} -ge 6 ]; then
  for set in ${subsets}; do
    echo "Generating speaker embedding from ${data_root}/${dataset_name}/data/${feat_type}${sample_rate}/${set}/idx2${feat_type}
    to ${data_root}/${dataset_name}/data/${feat_type}${sample_rate}/${set}/idx2${spk_emb_model}_spk_feat"

    args=
    if [ -n "${ngpu}" ];then
      args="--ngpu ${ngpu}"
    fi

    ${SPEECHAIN_PYTHON} "${pyscript_root}"/spk_feat_extractor.py \
      --src_file "${data_root}"/${dataset_name}/data/${feat_type}${sample_rate}/${set}/idx2${feat_type} \
      --spk_emb_model "${spk_emb_model}" \
      --ncpu ${ncpu} \
      ${args}
  done
fi


# --- Step7: Data Packaging --- #
if [ -n "${comp_chunk_ext}" ] && [ ${start_step} -le 7 ] && [ ${stop_step} -ge 7 ]; then
#  if [ ${feat_type} != 'wav' ]; then
#    folder_name=${feat_config}
#  else
#    folder_name=${feat_type}${sample_rate}
#  fi
#
#  for set in ${subsets}; do
#    echo "Packaging ${feat_type} data in ${data_root}/${dataset_name}/data/${folder_name}/${set}/......"
#    ${SPEECHAIN_PYTHON} "${pyscript_root}"/data_packager.py \
#      --src_path "${data_root}"/${dataset_name}/data/${folder_name}/${set} \
#      --comp_chunk_ext ${comp_chunk_ext} \
#      --feat_type ${feat_type} \
#      --ncpu ${ncpu}
#  done

  echo "Data Packaging is not available yet~~~~"
  exit 1
fi


# --- Step8: Meta Data Post-processing after all the Speech-related Steps --- #
if [ ${start_step} -le 8 ] && [ ${stop_step} -ge 8 ]; then
  if [ ${feat_type} != 'wav' ]; then
    folder_name=${feat_config}
  else
    folder_name=${feat_type}${sample_rate}
  fi

  if [ -f "${data_root}/${dataset_name}/meta_post_processor.py" ]; then
    echo "Post-processing the statistical information in ${data_root}/${dataset_name}/data/${folder_name}."
    ${SPEECHAIN_PYTHON} "${data_root}"/${dataset_name}/meta_post_processor.py \
      --src_path "${data_root}"/${dataset_name}/data/${folder_name} \
      ${meta_post_process_args}
  fi
fi


# --- Step9: Vocabulary List and Sentence Length Generation --- #
if [ ${start_step} -le 9 ] && [ ${stop_step} -ge 9 ]; then
  for set in ${vocab_src_subsets}; do
    echo "Generating ${token_type} vocabulary by ${data_root}/${dataset_name}/data/wav/${set}/idx2${txt_format}_text......"
    ${SPEECHAIN_PYTHON} "${pyscript_root}"/vocab_generator.py \
      --text_path "${data_root}"/${dataset_name}/data/wav/${set} \
      --save_path "${data_root}"/${dataset_name}/data/${token_type}/${set} \
      --token_type ${token_type} \
      --txt_format ${txt_format} \
      ${vocab_generate_args}
  done
fi
