#  Author: Heli Qi
#  Affiliation: NAIST
#  Date: 2022.07

# --- Involved Arguments --- #
# the path to place the downloaded dataset, default to the current path
download_path=${PWD}

function print_help_message {
  echo "usage:
  $0 \\ (The arguments in [] are optional while other arguments must be given by your run.sh.)
    --download_path DOWNLOAD_PATH      # The path to place the downloaded dataset. (default: \$PWD)" >&2
  exit 1
}

### get args from the command line ###
while getopts ":h-:" optchar; do
  case "${optchar}" in
    -)
      case "${OPTARG}" in
        download_path)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          download_path=${val}
          ;;
        help)
          print_help_message
          ;;
        *)
          echo "Unknown variable --$OPTARG"
          exit 1 ;;
      esac
      ;;
    h)
      print_help_message
      ;;
    *)
      echo "Please refer to an argument by '--'."
      exit 1 ;;
  esac
done


# create the data folder to place the dataset
mkdir -p ${download_path}/data

# skip the downloading and unzipping if 'wav' has already been downloaded and unzipped
if [ ! -d "${download_path}/data/wav" ]; then
  # download the data package if it doesn't exist
  if [ ! -f "${download_path}/data/LJSpeech-1.1.tar.bz2" ]; then
    echo "Download data from https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 to ${download_path}/data/LJSpeech-1.1.tar.bz2"
    wget -P ${download_path}/data https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
  else
    echo "${download_path}/data/LJSpeech-1.1.tar.bz2 has already existed. Skipping data downloading~~~"
  fi

  # unzip the downloaded data package
  if [ ! -d "${download_path}/data/LJSpeech-1.1" ]; then
    # for unzipping .tar.bz2, the arguments here need to be -xjf instead of -zxvf
    echo "Unzip the downloaded data from ${download_path}/data/LJSpeech-1.1.tar.bz2 to ${download_path}/data/wav"
    tar -xjf ${download_path}/data/LJSpeech-1.1.tar.bz2 -C ${download_path}/data
    mv ${download_path}/data/LJSpeech-1.1 ${download_path}/data/wav
  fi

else
  echo "${download_path}/data/wav has already existed. Skipping data downloading and unzipping~~~"
fi

# remove the compressed data package
echo "Remove the downloaded data ${download_path}/data/LJSpeech-1.1.tar.bz2"
rm ${download_path}/data/LJSpeech-1.1.tar.bz2

# Finally, remove the folder named LJSpeech-1.1 if needed
if [ -d "${download_path}/data/LJSpeech-1.1" ];then
  rm -rf "${download_path}/data/LJSpeech-1.1"
fi
