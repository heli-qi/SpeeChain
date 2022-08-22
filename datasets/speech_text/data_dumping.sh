#  Author: Heli Qi
#  Affiliation: NAIST
#  Date: 2022.07

# the absolute path of the speechain folder on your machine
speechain=/ahc/work4/heli-qi/euterpe-heli-qi/speechain
# the absolute path of the config folder on your machine
config=/ahc/work4/heli-qi/euterpe-heli-qi/config
# the absolute path of the python complier on your machine
python=/home/is/heli-qi/anaconda3/envs/speechain/bin/python3.8

# Note that the value of the following two arguments must end with a space. Otherwise, the last element will not be used.
# the name of all the subsets, used for step3, step4, and step5
subsets=" "
# the name of the subsets used for vocabulary generation in step6
vocab_subsets=" "

# the additional arguments that you want to give to stat_info_generator.py for custimized statistic information generation
stat_info_args=""
# the additional arguments that you want to give to vocab_generator.py for customized token vocabulary generation
vocab_args=""

# You need to override the default values of the arguments above according to your machine and target dataset
# The default values for the arguments below are shared across different datasets
start_step=1
stop_step=10000
feat_config=wav
sample_rate=
token_type=
ncpu=16

### get args from the command line ###
while getopts ":h-:" optchar; do
  case "${optchar}" in
    -)
      case "${OPTARG}" in
        start_step)
          val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
          start_step=${val}
          ;;
        stop_step)
          val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
          stop_step=${val}
          ;;
        feat_config)
          val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
          feat_config=${val}
          ;;
        sample_rate)
          val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
          sample_rate=${val}
          ;;
        token_type)
          val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
          token_type=${val}
          ;;
        ncpu)
          val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
          ncpu=${val}
          ;;
        subsets)
          val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
          subsets=${val}
          ;;
        vocab_subsets)
          val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
          vocab_subsets=${val}
          ;;
        stat_info_args)
          val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
          stat_info_args=${val}
          ;;
        vocab_args)
          val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
          vocab_args=${val}
          ;;
        ?)
          echo "Unknown variable $OPTARG"
          ;;
      esac
      ;;
    h)
      echo "under construction......"
      ;;
  esac
done

# arguments checking
if [ -z "${feat_config}" ];then
   echo "Please input the config file of extracting acoustic features."
   exit 1
fi


# --- Step1: Data Acquisition --- #
if [ ${start_step} -le 1 ] && [ ${stop_step} -ge 1 ]; then
  ./data_download.sh
fi


# --- Step2: Data Preparation --- #
if [ ${start_step} -le 2 ] && [ ${stop_step} -ge 2 ]; then
  echo "Generate the meta information of the dataset in ./data/wav"
  ${python} stat_info_generator.py \
    --src_path data/wav \
    ${stat_info_args}
fi


# --- Step3: Downsampling the audio files (optional) --- #
if [ -n "$sample_rate" ] && [ ${start_step} -le 3 ] && [ ${stop_step} -ge 3 ]; then
  mkdir -p data/wav${sample_rate}
  for set in ${subsets}; do
    echo "Downsampling the audio files in ./data/wav/${set}/idx2wav to ./data/wav${sample_rate}/${set}."
    ${python} ${speechain}/utilbox/wave_downsampler.py \
      --sample_rate ${sample_rate} \
      --src_file data/wav/${set}/idx2wav \
      --tgt_path data/wav${sample_rate}/${set} \
      --ncpu ${ncpu}
  done
fi


# --- Step4: Acoustic Feature Extraction (optional) --- #
if [ ${feat_config} != 'wav' ] && [ ${start_step} -le 4 ] && [ ${stop_step} -ge 4 ]; then
  for set in ${subsets}; do
    echo "Generating acoutic features from ./data/wav${sample_rate}/${set} to ./data/${feat_config}/${set}"
    mkdir -p data/${feat_config}/${set}

    ${python} ${speechain}/utils/feat_generator.py \
      --wav_scp data/wav${sample_rate}/${set}/idx2wav \
      --cfg ${config}/feat/"${feat_config}.json" \
      --output_path data/${feat_config}/${set} \
      --ncpu ${ncpu} \
      --type npz
  done
fi


# --- Step5: Acoustic Feature Length Generation --- #
if [ ${start_step} -le 5 ] && [ ${stop_step} -ge 5 ]; then
  if [ ${feat_config} != 'wav' ]; then
    feat_type=${feat_config}
    file_name=idx2feat
  else
    feat_type=${feat_config}${sample_rate}
    file_name=idx2wav
  fi

  for set in ${subsets}; do
    echo "Generating data lengths from ./data/${feat_type}/${set}/${file_name} to ./data/${feat_type}/${set}/${file_name}_len"
    ${python} ${speechain}/utilbox/data_len_generator.py \
      --data_path data/${feat_type}/${set}/${file_name} \
      --feat_type ${feat_config} \
      --ncpu ${ncpu}
  done
fi

# --- Step6: Vocabulary List and Sentence Length Generation --- #
if [ ${start_step} -le 6 ] && [ ${stop_step} -ge 6 ]; then
  for set in ${vocab_subsets}; do
    echo "Generating ${token_type} vocabulary by ./data/wav/${set}/text......"
    mkdir -p data/${token_type}/${set}

    ${python} ${speechain}/utilbox/vocab_generator.py \
      --text_path data/wav/${set}/text \
      --output_path data/${token_type}/${set} \
      --token_type ${token_type} \
      --tgt_subsets "${subsets}" \
      ${vocab_args}
  done
fi
