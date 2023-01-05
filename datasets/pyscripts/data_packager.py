"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.11
"""
import argparse
import os
import warnings

import numpy as np
import h5py
import soundfile as sf

from typing import Dict
from functools import partial
from multiprocessing import Pool
from speechain.utilbox.dump_util import parse_readable_number, get_readable_memory


def parse():
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--src_path', type=str, required=True,
                        help="The source path where the raw data to be packaged is placed.")
    parser.add_argument('--feat_type', type=str, required=True,
                        help="The type of feature you want to package.")
    parser.add_argument('--comp_chunk_ext', type=str, default='npz',
                        help="The extension of the compressed chunk files you want to specify. (default: npz)")
    parser.add_argument('--chunk_size', type=str, default=None,
                        help="The size of compressed chunk files you want to create. "
                             "If 'idx2{feat_type}_len' is given, chunk_size means the overall length of all data "
                             "instances in a chunk; If not, chunk_size means the number of the data instances in a "
                             "chunk. This argument can be given in the form of 'XXbXXXmXXXkXhXX' where b, m, k, "
                             "h mean billion, million, kilo, hundred respectively. "
                             "(default: '40m' if 'idx2{feat_type}_len' is given else '2000')")
    parser.add_argument('--ncpu', type=int, default=8,
                        help="The number of processes you want to use to save the chunk files. (default: 8)")
    return parser.parse_args()


def open_file_writer(file_path: str, file_ext: str):
    if file_ext == 'hdf5':
        return h5py.File(file_path, 'w')
    else:
        # 'w' will cause 'TypeError: write() argument must be str, not bytes'; 'wb' means write + byte
        return open(file_path, mode='wb')


def save_chunk(chunk_num: int, chunk_dict: Dict, feat_type: str, comp_chunk_ext: str,
               save_path: str, remove_ori_data: bool):
    idx2data_comp_chunk = dict()
    chunk_path = os.path.join(save_path, f'chunk_{chunk_num}.{comp_chunk_ext}')

    if os.path.exists(chunk_path):
        for idx in chunk_dict.keys():
            # record the address of compressed data
            idx2data_comp_chunk[idx] = ':'.join([chunk_path, idx])

    else:
        # inject the file into the compressed package
        with open_file_writer(chunk_path, comp_chunk_ext) as writer:
            for idx, data_path in chunk_dict.items():
                # read the raw data
                if feat_type == 'wav':
                    data = sf.read(data_path)[0]
                elif feat_type == 'feat':
                    data = np.load(data_path)['feat']
                else:
                    raise ValueError

                # write the data into the compressed file
                if comp_chunk_ext == 'hdf5':
                    writer.create_dataset(idx, data=data, compression='gzip', compression_opts=9)
                elif comp_chunk_ext == 'npz':
                    np.savez_compressed(writer, **{idx: data})
                else:
                    raise ValueError

                # record the address of compressed data
                idx2data_comp_chunk[idx] = ':'.join([chunk_path, idx])

    # remove all the original data files in the chunk after finishing this chunk file
    if remove_ori_data:
        for data_path in chunk_dict.values():
            os.remove(data_path)

    return idx2data_comp_chunk


def main(src_path: str, feat_type: str, comp_chunk_ext: str,
         chunk_size: int or str = None, ncpu: int = 8, remove_ori_data: bool = False):
    """

    Args:
        src_path:
        feat_type:
        comp_chunk_ext:
        chunk_size:
        ncpu:
        remove_ori_data:

    Returns:

    """
    # --- 0. Information Initialization --- #
    # compressed data folder creating
    assert comp_chunk_ext in ['npz', 'hdf5'], \
        f"Data compression file extension should be one of {['npz', 'hdf5']}, but got {comp_chunk_ext}."
    comp_save_path = os.path.join(src_path, f'{feat_type}_{comp_chunk_ext}')
    os.makedirs(comp_save_path, exist_ok=True)

    # data path reading
    idx2data = os.path.join(src_path, f'idx2{feat_type}')
    if os.path.exists(idx2data):
        idx2data = np.loadtxt(idx2data, dtype=str, delimiter=" ")
        idx2data = dict(zip(idx2data[:, 0], idx2data[:, 1]))
    else:
        raise RuntimeError("Please create 'idx2wav' or 'idx2feat' before data compression!")

    # data length reading
    idx2data_len = os.path.join(src_path, f'idx2{feat_type}_len')
    # sort idx2data in descending order by their length if data_len is given
    if os.path.exists(idx2data_len):
        idx2data_len = np.loadtxt(idx2data_len, dtype=str, delimiter=" ")
        idx2data_len = dict(sorted(idx2data_len, key=lambda x: int(x[1]), reverse=True))
        idx2data = {idx: idx2data[idx] for idx in idx2data_len.keys()}
    else:
        idx2data_len = None
        warnings.warn(f"Data length file 'idx2{feat_type}_len' is not found in {src_path}, "
                      f"so the data compressed into the same chunk may not have the similar lengths.")

    # different default values for fixed-length chunk and fixed-number chunk
    if chunk_size is None:
        chunk_size = '40m' if idx2data_len is not None else '2k'
    # turn the readable value to the raw integer value
    if isinstance(chunk_size, str):
        chunk_size = parse_readable_number(chunk_size)

    # --- 1. Collect Chunk Information --- #
    # loop each wav file
    chunk_dict, curr_chunk_num, curr_chunk_size = dict(), 0, 0
    for idx, data_path in idx2data.items():
        # create the current chunk sub-dict
        if curr_chunk_num not in chunk_dict.keys():
            chunk_dict[curr_chunk_num] = dict()

        # update the current data instance to the current chunk
        chunk_dict[curr_chunk_num][idx] = idx2data[idx]
        curr_chunk_size += int(idx2data_len[idx]) if idx2data_len is not None else 1

        # move to the next chunk if the current chunk is full
        if curr_chunk_size >= chunk_size:
            curr_chunk_num += 1
            curr_chunk_size = 0

    # --- 2. Write the collected data instances into compressed chunk files --- #
    chunk_list = list(chunk_dict.items())
    with Pool(ncpu) as executor:
        save_chunk_func = partial(save_chunk, feat_type=feat_type, comp_chunk_ext=comp_chunk_ext,
                                  save_path=comp_save_path, remove_ori_data=remove_ori_data)
        idx2data_comp_chunk = executor.starmap(save_chunk_func, chunk_list)

    total_chunk_num = len(chunk_list)
    chunk_memories = [os.path.getsize(os.path.join(comp_save_path, f'chunk_{chunk_num}.{comp_chunk_ext}'))
                      for chunk_num in range(total_chunk_num)]
    print(f"Successfully save {len(chunk_list)} chunk files in {comp_save_path}!\n"
          f"Average size: {get_readable_memory(sum(chunk_memories) / total_chunk_num)}; "
          f"Total size: {get_readable_memory(sum(chunk_memories))}.")

    # --- 3. Save the new data addressed after compression --- #
    idx2data_comp = dict()
    for chunk in idx2data_comp_chunk:
        idx2data_comp.update(chunk.items())
    np.savetxt(os.path.join(src_path, f'idx2{feat_type}_{comp_chunk_ext}'),
               sorted(idx2data_comp.items(), key=lambda x: x[0]), fmt='%s')


if __name__ == '__main__':
    args = parse()
    main(**vars(args))
