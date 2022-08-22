"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import argparse
import numpy as np
import os
import pandas as pd

def parse():
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--src_path', type=str, default='/ahc/work4/heli-qi/euterpe-heli-qi/datasets/speech_text/ljspeech/data/wav')
    parser.add_argument('--valid_size', type=int, default=400)
    parser.add_argument('--test_size', type=int, default=400)
    return parser.parse_args()
    pass


def main(args):
    src_path = args.src_path
    valid_size = args.valid_size
    test_size = args.test_size

    # metadata.csv contains a lot of '"', which is taken as the default quotechar by read_csv().
    # To correctly load those '"' for processing, we need to set quotechar=None and quoting=3.
    idx2sent = pd.read_csv(src_path + "/metadata.csv", delimiter='|', quotechar=None, quoting=3, header=None).to_numpy(dtype=str)
    idx2sent = np.delete(idx2sent, 1, axis=1)

    # turn capital letters into their lowercase
    idx2sent[:, 1] = np.char.lower(idx2sent[:, 1])
    # turn semicolon into comma
    idx2sent[:, 1] = np.char.replace(idx2sent[:, 1], ';', ',')
    # turn quotation mark and full-width symbols into apostrophe
    idx2sent[:, 1] = np.char.replace(idx2sent[:, 1], '"', '\'')
    idx2sent[:, 1] = np.char.replace(idx2sent[:, 1], '“', '\'')
    idx2sent[:, 1] = np.char.replace(idx2sent[:, 1], '”', '\'')
    idx2sent[:, 1] = np.char.replace(idx2sent[:, 1], '’', '\'')
    # delete horizontal lines as well as parentheses and brackets
    idx2sent[:, 1] = np.char.replace(idx2sent[:, 1], '-', '')
    idx2sent[:, 1] = np.char.replace(idx2sent[:, 1], '(', '')
    idx2sent[:, 1] = np.char.replace(idx2sent[:, 1], ')', '')
    idx2sent[:, 1] = np.char.replace(idx2sent[:, 1], '[', '')
    idx2sent[:, 1] = np.char.replace(idx2sent[:, 1], ']', '')
    # turn some non-English letters into English letters
    idx2sent[:, 1] = np.char.replace(idx2sent[:, 1], 'ü', 'u')
    idx2sent[:, 1] = np.char.replace(idx2sent[:, 1], 'é', 'e')
    idx2sent[:, 1] = np.char.replace(idx2sent[:, 1], 'ê', 'e')
    idx2sent[:, 1] = np.char.replace(idx2sent[:, 1], 'è', 'e')
    idx2sent[:, 1] = np.char.replace(idx2sent[:, 1], 'â', 'a')
    idx2sent[:, 1] = np.char.replace(idx2sent[:, 1], 'à', 'a')

    # Now, the characters contain 26 lowercase English letters + space + 6 punctuation symbols (' : ! ? . ,)
    # These punctuation symbols are helpful for TTS modeling.

    idx2wav = []
    for file in os.listdir(src_path + "/wavs"):
        file_path = src_path + "/wavs/" + file
        idx2wav.append([file.split('.')[0], os.path.abspath(file_path)])

    if not os.path.exists(src_path + "/train"):
        os.makedirs(src_path + "/train")
    if not os.path.exists(src_path + f"/valid"):
        os.makedirs(src_path + "/valid")
    if not os.path.exists(src_path + f"/test"):
        os.makedirs(src_path + "/test")


    # idx2wav
    idx2wav = np.array(idx2wav, dtype=str)
    idx2wav = idx2wav[idx2wav[:, 0].argsort()]
    # idx2sent
    idx2sent = idx2sent[idx2sent[:, 0].argsort()]


    print(f"Saving statistic information of subset train to {src_path}/train")
    np.savetxt(f"{src_path}/train/idx2wav", idx2wav[: -(valid_size + test_size)], fmt='%s')
    np.savetxt(f"{src_path}/train/idx2sent", idx2sent[: -(valid_size + test_size)], fmt='%s')
    np.savetxt(f"{src_path}/train/text", idx2sent[: -(valid_size + test_size), 1], fmt='%s')

    print(f"Saving statistic information of subset valid to {src_path}/train")
    np.savetxt(f"{src_path}/valid/idx2wav", idx2wav[-(valid_size + test_size): -test_size], fmt='%s')
    np.savetxt(f"{src_path}/valid/idx2sent", idx2sent[-(valid_size + test_size): -test_size], fmt='%s')
    np.savetxt(f"{src_path}/valid/text", idx2sent[-(valid_size + test_size): -test_size, 1], fmt='%s')

    print(f"Saving statistic information of subset test to {src_path}/train")
    np.savetxt(f"{src_path}/test/idx2wav", idx2wav[-test_size:], fmt='%s')
    np.savetxt(f"{src_path}/test/idx2sent", idx2sent[-test_size:], fmt='%s')
    np.savetxt(f"{src_path}/test/text", idx2sent[-test_size:, 1], fmt='%s')


if __name__ == '__main__':
    main(parse())