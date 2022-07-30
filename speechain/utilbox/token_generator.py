"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import argparse
from collections import Counter
import time
import numpy as np
import os

def parse():
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--text', type=str, default="../../../egs/librispeech/data/raw/train_clean_100/text")
    parser.add_argument('--output_path', type=str, default="../../../egs/librispeech/data/char/train_clean_100")
    parser.add_argument('--token_type', type=str, default="char")
    return parser.parse_args()


def generate_dict_char(text):
    # collect all chars and count their occurrences
    tokens = []
    for i in text:
        tokens.append(list(i[1]))
    tokens = Counter(np.concatenate(tokens))
    tokens = sorted(list(tokens.keys()))

    # 0 is designed for the blank (the padding index)
    # -2 is designed for the unknowns
    # -1 is designed for the beginning and end of sentence
    dic = ["<blank>"] + tokens + ["<unk>", "<sos/eos>"]
    return np.array(dic)


if __name__ == '__main__':
    args = parse()
    token_type = args.token_type
    output_path = args.output_path
    text = np.loadtxt(args.text, delimiter="\n", dtype=str)
    text = np.stack(np.chararray.split(text, maxsplit=1))

    if os.path.exists(output_path):
        assert os.path.isdir(output_path), "output_path must be a folder"
    else :
        os.makedirs(output_path, mode=0o755, exist_ok=False)

    start_time = time.time()

    if token_type == 'char':
        dict = generate_dict_char(text)
    else:
        raise NotImplementedError

    np.savetxt(output_path + "/dict", np.array(dict), fmt="%s")

    print("Time elapsed %.2f secs"%(time.time() - start_time))
    print("=== FINISH ===")
    pass