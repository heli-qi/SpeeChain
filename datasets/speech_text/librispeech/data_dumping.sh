#  Author: Heli Qi
#  Affiliation: NAIST
#  Date: 2022.07

# the absolute path of the speechain folder on your machine
speechain=/ahc/work4/heli-qi/euterpe-heli-qi/speechain
# the absolute path of the config folder on your machine
config=/ahc/work4/heli-qi/euterpe-heli-qi/config
# the absolute path of the python complier on your machine
python=/home/is/heli-qi/anaconda3/envs/speechain/bin/python3.8

# the name of involved subsets, used for step3 and step4
# Note that the value of this argument should end with a blank. Otherwise, the last subset will not be used.
subsets="train_clean_100 train_clean_360 train_clean_460 train_other_500 train_960 dev_clean dev_other test_clean test_other "

# the additional arguments that you want to give to stat_info_generator.py for statistic information generation
# for example, in the case of LJSpeech, stat_info_args can be "--valid_size 500 --test_size 500" to divide the dataset in another way instead of the default values 400
stat_info_args=""

# You need to override the default values of the arguments above according to your machine and target dataset
# The default values for the arguments below are shared across different datasets
start_step=1
stop_step=10000
feat_config=
sample_rate=
token_type=char
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
        stat_info_args)
          val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
          stat_info_args=${val}
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


### Step1: Data Acquisition ###
if [ ${start_step} -le 1 ] && [ ${stop_step} -ge 1 ]; then
  ./data_download.sh
fi


### Step2: Data Preparation ###
if [ ${start_step} -le 2 ] && [ ${stop_step} -ge 2 ]; then
  echo "Generate the meta information of the dataset in ./data/raw"
  ${python} stat_info_generator.py --src_path data/raw ${stat_info_args}
fi


### Step3: Downsampling the audio files ###
if [ -n "$sample_rate" ] && [ ${start_step} -le 3 ] && [ ${stop_step} -ge 3 ]; then
  mkdir -p data/raw${sample_rate}
  for set in ${subsets}; do
    echo "Downsampling the audio files in ./data/raw/${set}/feat.scp to ./data/raw${sample_rate}/${set}."
    ${python} ${speechain}/utilbox/wave_downsampler.py \
      --sample_rate ${sample_rate} \
      --src_file data/raw/${set}/feat.scp \
      --tgt_path data/raw${sample_rate}/${set} \
      --ncpu ${ncpu}
  done
fi


### Step4: Acoustic Feature Extraction ###
if [ ${feat_config} != 'raw' ] && [ ${start_step} -le 4 ] && [ ${stop_step} -ge 4 ]; then
  for set in ${subsets}; do
    echo "Generating acoutic features from ./data/raw${sample_rate}/${set} to ./data/${feat_config}/${set}"
    mkdir -p data/${feat_config}/${set}

    ${python} ${speechain}/utils/feat_generator.py \
      --wav_scp data/raw${sample_rate}/${set}/wav.scp \
      --ncpu ${ncpu} \
      --cfg ${config}/feat/"${feat_config}.json" \
      --output_path data/${feat_config}/${set} \
      --type npz
  done
fi


### Step5: Acoustic Feature Length Generation ###
if [ ${start_step} -le 5 ] && [ ${stop_step} -ge 5 ]; then
  if [ ${feat_config} != 'raw' ]; then
    data_type=${feat_config}
  else
    data_type=${feat_config}${sample_rate}
  fi

  for set in ${subsets}; do
    echo "Generating acoutic feature lengths from ./data/${data_type}/${set}/feat.sp to ./data/${data_type}/${set}/feat_len.scp"
    ${python} ${speechain}/utilbox/data_len_generator.py \
      --data_path data/${data_type}/${set}/feat.scp \
      --feat_type ${feat_config} \
      --ncpu ${ncpu}
  done
fi


### Step6: Token Dictionary Generation ###
if [ ${start_step} -le 6 ] && [ ${stop_step} -ge 6 ]; then
  for set in ${subsets}; do
    echo "Generating token dictionary from ./data/raw${sample_rate}/${set}/text to ./data/${token_type}/${set}/dict"
    mkdir -p data/${token_type}/${set}

    ${python} ${speechain}/utilbox/token_generator.py \
      --text data/raw${sample_rate}/${set}/text \
      --output_path data/${token_type}/${set} \
      --token_type ${token_type}
  done
fi
