"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import shutil
from typing import List

import librosa
import soundfile as sf
import argparse
import os
import numpy as np

from multiprocessing import Pool
from functools import partial


def parse():
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--src_file', type=str, required=True)
    parser.add_argument('--tgt_path', type=str, required=True)
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help="The number of waveform instances packaged in a chunk when downsamping by multiple "
                             "processes. (default: 1000)")
    parser.add_argument('--ncpu', type=int, default=8,
                        help="The number of processes you want to use to save the chunk files. (default: 8)")
    return parser.parse_args()


def waveform_downsample(idx2src_wav: List[List[str]], tgt_path: str, sample_rate: int) -> List[List[str]]:
    """
    sf.read(path) + sf.write(path, sample_rate) doesn't work because the values of waveforms remain the same.
    The only thing that is changed is the sampling rate of the waveform files.

    Args:
        idx2src_wav:
        tgt_path:
        sample_rate:

    Returns:

    """
    idx2tgt_wav = []
    # loop each source data wav in the given chunk
    for idx, src_wav_path in idx2src_wav:
        file_name = src_wav_path.split('/')[-1]
        tgt_wav_path = os.path.join(tgt_path, file_name)
        # create the downsampled waveform file
        if not os.path.exists(tgt_wav_path):
            src_wav = librosa.core.load(src_wav_path, sr=sample_rate)[0]
            wav_format = file_name.split('.')[-1].upper()
            sf.write(file=tgt_wav_path, data=src_wav, samplerate=sample_rate,
                     format=wav_format, subtype=sf.default_subtype(wav_format))
        # record the target waveform path
        idx2tgt_wav.append([idx, tgt_wav_path])

    return idx2tgt_wav


def main(src_file: str, tgt_path: str, sample_rate: int, ncpu: int, chunk_size: int = 1000):
    src_file, tgt_path = os.path.abspath(src_file), os.path.abspath(tgt_path)
    os.makedirs(tgt_path, exist_ok=True)

    # skip the dowmsampling process if there has already been an idx2wav
    idx2tgt_wav_path = os.path.join(tgt_path, "idx2wav")
    if not os.path.exists(idx2tgt_wav_path):
        # reshape the source waveform paths into individual chunks by the given chunk_size
        idx2src_wav = np.loadtxt(src_file, delimiter=" ", dtype=str)
        _residue = len(idx2src_wav) % chunk_size
        idx2src_wav_chunk = idx2src_wav[:-_residue].reshape(-1, chunk_size, idx2src_wav.shape[-1]).tolist()
        idx2src_wav_chunk.append(idx2src_wav[-_residue:].tolist())

        # saving the downsampled audio files to the disk
        with Pool(ncpu) as executor:
            waveform_downsample_func = partial(waveform_downsample, tgt_path=tgt_path, sample_rate=sample_rate)
            idx2tgt_wav_chunk = executor.map(waveform_downsample_func, idx2src_wav_chunk)

        idx2tgt_wav = []
        for chunk in idx2tgt_wav_chunk:
            idx2tgt_wav += chunk
        np.savetxt(idx2tgt_wav_path, sorted(idx2tgt_wav, key=lambda x: x[0]), fmt='%s')
    else:
        print(f"Downsampled waveforms have already existed in {args.tgt_path}, so the dowmsampling process is skipped.")

    print(f"Copying statistic information from {os.path.dirname(src_file)} to {tgt_path}")
    src_dir = os.path.dirname(src_file)
    for file in os.listdir(src_dir):
        # skip idx2wav, folders, and the files with a suffix
        if file in ['idx2wav', 'idx2wav_len', 'idx2feat', 'idx2feat_len'] or \
                os.path.isdir(os.path.join(src_dir, file)) or '.' in file:
            continue
        shutil.copy(os.path.join(src_dir, file), tgt_path)


if __name__ == '__main__':
    args = parse()
    main(**vars(args))
