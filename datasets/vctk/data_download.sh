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
  if [ ! -f "${download_path}/data/VCTK-Corpus-0.92.zip" ]; then
    echo "Download data from https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip to ${download_path}/data/VCTK-Corpus-0.92.zip"
    wget -P ${download_path}/data https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip
  else
    echo "${download_path}/data/VCTK-Corpus-0.92.zip has already existed. Skipping data downloading~~~"
  fi

  # unzip the downloaded data package
  if [ ! -d "${download_path}/data/wav/txt" ] || [ ! -d "${download_path}/data/wav/wav48_silence_trimmed" ]; then
    # for unzipping .zip, we need to use unzip rather than tar
    echo "Unzip the downloaded data from ${download_path}/data/VCTK-Corpus-0.92.zip to ${download_path}/data/wav"
    mkdir -p ${download_path}/data/wav
    unzip ${download_path}/data/VCTK-Corpus-0.92.zip -d ${download_path}/data/wav
  fi

else
  echo "${download_path}/data/wav has already existed. Skipping data downloading and unzipping~~~"
fi

# remove the compressed data package
if [ -f ${download_path}/data/VCTK-Corpus-0.92.zip ];then
  echo "Remove the downloaded data ${download_path}/data/VCTK-Corpus-0.92.zip"
  rm ${download_path}/data/VCTK-Corpus-0.92.zip
fi
