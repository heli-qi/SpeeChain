"""
    Author: Sashi Novitasari
    Affiliation: NAIST
    Date: 2022.07
"""
import soundfile as sf
import argparse
import os
import numpy as np
from multiprocessing import Pool
from functools import partial

from euterpe.utilbox.regex_util import regex_key_val


def get_feat_length(path, args):
    feat = sf.read(path)[0] if args.feat_type == 'raw' else np.load(path)['feat']
    return feat.shape[0]

def parse():
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--feat_type', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--ncpu', type=int, default=16)
    return parser.parse_args()
    pass

if __name__ == '__main__':
    args = parse()

    folder = os.path.dirname(args.data_path)
    kv_list = regex_key_val.findall(open(args.data_path).read())
    k_list, v_list = zip(*kv_list)

    # For running
    with Pool(args.ncpu) as executor:
        get_feat_length_args = partial(get_feat_length, args=args)
        output_result = executor.map(get_feat_length_args, v_list)

    output_result = np.concatenate((np.array(k_list).reshape(-1, 1), np.array(output_result).reshape(-1, 1)), axis=1)
    np.savetxt('{}_len{}'.format(*os.path.splitext(args.data_path)), output_result, fmt="%s")