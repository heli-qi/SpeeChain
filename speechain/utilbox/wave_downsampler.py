"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import shutil
import librosa
import soundfile as sf
import argparse
import os
import numpy as np
from multiprocessing import Pool
from functools import partial

from speechain.utilbox.regex_util import regex_key_val

def parse() :
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--src_file', type=str, required=True)
    parser.add_argument('--tgt_path', type=str, required=True)
    parser.add_argument('--ncpu', type=int, default=16)
    return parser.parse_args()


def waveform_downsample(path, args):
    """
    sf.read(path) + sf.write(path, sample_rate) doesn't work because the values of waveforms remain the same.
    The only thing that is changed is the sampling rate of the waveform files.

    Args:
        path:
        args:

    Returns:

    """
    wav = librosa.core.load(path, sr=args.sample_rate)[0]
    file_name = path.split('/')[-1]
    file_format = file_name.split('.')[-1].upper()
    sf.write(file=os.path.join(args.tgt_path, file_name), data=wav, samplerate=args.sample_rate,
             format=file_format, subtype=sf.default_subtype(file_format))


if __name__ == '__main__' :
    args = parse()
    if not os.path.exists(args.tgt_path):
        os.makedirs(args.tgt_path)

    kv_list = regex_key_val.findall(open(args.src_file).read())
    k_list, v_list = zip(*kv_list)

    # saving the downsampled audio files to the disk
    with Pool(args.ncpu) as executor:
        waveform_downsample_args = partial(waveform_downsample, args=args)
        executor.map(waveform_downsample_args, v_list)

    idx2wav = []
    for file in os.listdir(args.tgt_path):
        file_path = args.tgt_path + '/' + file
        idx2wav.append([file.split('.')[0], os.path.abspath(file_path)])

    print(f"Copying statistic information from {os.path.dirname(args.src_file)} to {args.tgt_path}")
    np.savetxt(os.path.abspath(args.tgt_path) + "/idx2wav", sorted(idx2wav, key=lambda x: x[0]), fmt='%s')
    shutil.copy(os.path.dirname(args.src_file) + "/idx2sent", args.tgt_path)
    shutil.copy(os.path.dirname(args.src_file) + "/text", args.tgt_path)