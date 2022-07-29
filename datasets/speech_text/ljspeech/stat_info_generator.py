"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import argparse
import numpy as np
import os
import pandas as pd


def main(args):
    src_path = args.src_path
    valid_size = args.valid_size
    test_size = args.test_size

    # metadata.csv contains a lot of '"', which is taken as the default quotechar by read_csv().
    # To correctly load those '"' for processing, we need to set quotechar=None and quoting=3.
    text = pd.read_csv(src_path + "/metadata.csv", delimiter='|', quotechar=None, quoting=3, header=None).to_numpy(dtype=str)
    text = np.delete(text, 1, axis=1)

    # turn capital letters into their lowercase
    text[:, 1] = np.char.lower(text[:, 1])
    # turn semicolon into comma
    text[:, 1] = np.char.replace(text[:, 1], ';', ',')
    # turn quotation mark and full-width symbols into apostrophe
    text[:, 1] = np.char.replace(text[:, 1], '"', '\'')
    text[:, 1] = np.char.replace(text[:, 1], '“', '\'')
    text[:, 1] = np.char.replace(text[:, 1], '”', '\'')
    text[:, 1] = np.char.replace(text[:, 1], '’', '\'')
    # delete parentheses and brackets
    text[:, 1] = np.char.replace(text[:, 1], '(', '')
    text[:, 1] = np.char.replace(text[:, 1], ')', '')
    text[:, 1] = np.char.replace(text[:, 1], '[', '')
    text[:, 1] = np.char.replace(text[:, 1], ']', '')
    # turn some non-English letters into English letters
    text[:, 1] = np.char.replace(text[:, 1], 'ü', 'u')
    text[:, 1] = np.char.replace(text[:, 1], 'é', 'e')
    text[:, 1] = np.char.replace(text[:, 1], 'ê', 'e')
    text[:, 1] = np.char.replace(text[:, 1], 'è', 'e')
    text[:, 1] = np.char.replace(text[:, 1], 'â', 'a')
    text[:, 1] = np.char.replace(text[:, 1], 'à', 'a')


    feat_scp = []
    for file in os.listdir(src_path + "/wavs"):
        file_path = src_path + "/wavs/" + file
        feat_scp.append([file.split('.')[0], os.path.abspath(file_path)])

    if not os.path.exists(src_path + "/train"):
        os.makedirs(src_path + "/train")
    if not os.path.exists(src_path + f"/valid"):
        os.makedirs(src_path + "/valid")
    if not os.path.exists(src_path + f"/test"):
        os.makedirs(src_path + "/test")


    # feat.scp
    feat_scp = np.array(feat_scp, dtype=str)
    feat_scp = feat_scp[feat_scp[:, 0].argsort()]
    # text
    text = text[text[:, 0].argsort()]


    print(f"Saving statistic information of subset train to {src_path}/train")
    np.savetxt(f"{src_path}/train/feat.scp", feat_scp[: -(valid_size + test_size)], fmt='%s')
    np.savetxt(f"{src_path}/train/text", text[: -(valid_size + test_size)], fmt='%s')

    print(f"Saving statistic information of subset valid to {src_path}/train")
    np.savetxt(f"{src_path}/valid/feat.scp", feat_scp[-(valid_size + test_size): -test_size], fmt='%s')
    np.savetxt(f"{src_path}/valid/text", text[-(valid_size + test_size): -test_size], fmt='%s')

    print(f"Saving statistic information of subset test to {src_path}/train")
    np.savetxt(f"{src_path}/test/feat.scp", feat_scp[-test_size:], fmt='%s')
    np.savetxt(f"{src_path}/test/text", text[-test_size:], fmt='%s')


def parse():
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--src_path', type=str, required=True)
    parser.add_argument('--valid_size', type=int, default=400)
    parser.add_argument('--test_size', type=int, default=400)
    return parser.parse_args()
    pass


if __name__ == '__main__':
    main(parse())