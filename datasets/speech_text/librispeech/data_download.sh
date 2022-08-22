#  Author: Heli Qi
#  Affiliation: NAIST
#  Date: 2022.07

subsets="train-clean-100 train-clean-360 train-other-500 train-960 dev-clean dev-other test-clean test-other "

mkdir -p data
for set in ${subsets}; do
  # data download
  echo "Download data from https://www.openslr.org/resources/12/${set}.tar.gz to ./data/${set}.tar.gz"
  wget -P data https://www.openslr.org/resources/12/${set}.tar.gz

  # data unzip
  echo "Unzip the downloaded data ./data/${set}.tar.gz to ./data/LibriSpeech/${set}"
  tar -zxvf data/${set}.tar.gz -C data

  echo "Remove the downloaded data ./data/${subsets}.tar.gz"
  rm data/${subsets}.tar.gz
done

echo "Rename ./data/LibriSpeech to ./data/wav"
mv data/LibriSpeech data/wav
