#  Author: Heli Qi
#  Affiliation: NAIST
#  Date: 2022.07

mkdir -p data
echo "Download data from https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 to ./data/LJSpeech-1.1.tar.bz2"
wget -P data https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2

# for unziping .tar.bz2, the arguments need to be -xjf instead of -zxvf
echo "Unzip the downloaded data from ./data to ./data/raw"
tar -xjf data/LJSpeech-1.1.tar.bz2 -C data
mv data/LJSpeech-1.1 data/raw

echo "Remove the downloaded data ./data/LJSpeech-1.1.tar.bz2"
rm data/LJSpeech-1.1.tar.bz2
