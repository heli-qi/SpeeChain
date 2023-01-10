#  Author: Heli Qi
#  Affiliation: NAIST
#  Date: 2022.07

# --- Involved Arguments --- #
# the path to place the downloaded dataset, default to the current path
download_path=${PWD}
# which subsets of LibriSpeech you want to download, default to all the subsets
subsets="train-clean-100,train-clean-360,train-other-500,dev-clean,dev-other,test-clean,test-other"
# the separator used to separate the input 'subsets' argument from a string into an array of string, default to be comma
separator=','

function print_help_message {
  echo "usage:
  $0 \\ (The arguments in [] are optional while other arguments must be given by your run.sh.)
    --download_path DOWNLOAD_PATH \\       # The path to place the downloaded dataset. (default: \$PWD)
    --subsets SUBSETS \\                   # A comma-separated string which defines the subsets of LibriSpeech you want to download. (default: all subsets)
    [--separator SEPARATOR] \\             # The separator used to separate the input 'subsets' argument from a string into an array of string. (default: ',')" >&2
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
        subsets)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          subsets=${val}
          ;;
        separator)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          separator=${val}
          ;;
        help)
          print_help_message
          ;;
        ?)
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


# --- Argument Initialization ---#
# create the data folder to place the dataset
mkdir -p ${download_path}/data

# convert 'subsets' from a string to an array with ',' as the separator
ori_IFS=${IFS}
IFS=${separator}
read -ra subsets <<<"$subsets"
# recover IFS to its original value after separation
IFS=${ori_IFS}


# --- Argument Checking --- #
all_subsets=(
'train-clean-100'
'train-clean-360'
'train-other-500'
'dev-clean'
'dev-other'
'test-clean'
'test-other'
)
for (( n=0; n < ${#subsets[*]}; n++ )); do
  set=${subsets[n]}

  # check whether the current subset name is valid
  valid_flag=false
  for value in ${all_subsets[*]}; do
    if [ "${set}" == "${value}" ]; then
      valid_flag=true
      break
    fi
  done
  if ! ${valid_flag}; then
    echo "Your no.$((n+1)) input '${set}' is invalid. It must be one of [${all_subsets[*]}]!"
    exit 1
  fi

  # check whether the current subset name is repeated
  repeat_flag=false
  for (( m=0; m < ${#subsets[*]}; m++ )); do
    if [ "$m" != "$n" ]; then
      if [ "${set}" == "${subsets[m]}" ]; then
        repeat_flag=true
        break
      fi
    fi
  done
  if ${repeat_flag}; then
    echo "Your no.$((n+1)) input '${set}' is repeated!"
    exit 1
  fi
done


# --- Data Downloading --- #
# loop each target subset
for (( n=0; n < ${#subsets[*]}; n++ )); do
  set=${subsets[n]}
  # skip the subset that has already been downloaded and unzipped
  if [ ! -d "${download_path}/data/wav/${set}" ]; then
    # download the data package if it doesn't exist
    if [ ! -f "${download_path}/data/${set}.tar.gz" ]; then
      echo "Download data from https://www.openslr.org/resources/12/${set}.tar.gz to ${download_path}/data/${set}.tar.gz"
      wget -P ${download_path}/data https://www.openslr.org/resources/12/"${set}".tar.gz
    else
      echo "${download_path}/data/${set}.tar.gz has already existed. Skipping data downloading~~~"
    fi

    # unzip the downloaded data package
    if [ ! -d "${download_path}/data/LibriSpeech/${set}" ]; then
      echo "Unzip the downloaded data ${download_path}/data/${set}.tar.gz to ${download_path}/data/LibriSpeech/${set}"
      tar -zxvf ${download_path}/data/"${set}".tar.gz -C ${download_path}/data

    else
      echo "${download_path}/data/LibriSpeech/${set} has already existed. Skipping data unzipping~~~"
    fi

    # move the unzipped data folder to the wav folder
    echo "Move ${download_path}/data/LibriSpeech/${set} to ${download_path}/data/wav/${set}"
    # if wav doesn't exist, directly rename the folder to include those .TXT files
    if [ ! -d "${download_path}/data/wav" ]; then
      mv -f ${download_path}/data/LibriSpeech ${download_path}/data/wav
    # if wav has already existed, just move the subset folder to wav
    else
      mv -f ${download_path}/data/LibriSpeech/"${set}" ${download_path}/data/wav/
    fi

  else
    echo "${download_path}/data/wav/${set} has already existed. Skipping data downloading and unzipping~~~"
  fi

  # remove the compressed data package
  echo "Remove the downloaded data package ${download_path}/data/${set}.tar.gz"
  rm ${download_path}/data/"${set}".tar.gz
done

# Finally, remove the folder named LibriSpeech if needed
if [ -d "${download_path}/data/LibriSpeech" ];then
  rm -rf "${download_path}/data/LibriSpeech"
fi
