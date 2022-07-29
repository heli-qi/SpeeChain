"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import argparse
import os
import numpy as np
import pandas as pd

def main(args):
    src_path = args.src_path
    feat_scp = {
        "train_clean_100": [],
        "train_clean_360": [],
        "train_clean_460": [],
        "train_other_500": [],
        "train_960": [],
        "dev_clean": [],
        "dev_other": [],
        "test_clean": [],
        "test_other": []
    }

    utt2spk = {
        "train_clean_100": [],
        "train_clean_360": [],
        "train_clean_460": [],
        "train_other_500": [],
        "train_960": [],
        "dev_clean": [],
        "dev_other": [],
        "test_clean": [],
        "test_other": []
    }

    text = {
        "train_clean_100": [],
        "train_clean_360": [],
        "train_clean_460": [],
        "train_other_500": [],
        "train_960": [],
        "dev_clean": [],
        "dev_other": [],
        "test_clean": [],
        "test_other": []
    }

    ### Collect speech and text data ###
    # looping for each subset
    for dset in os.listdir(src_path):
        # skip those .TXT files
        if '.' in dset:
            continue

        # replace '-' with '_' if needed
        if '-' in dset:
            os.rename(src_path + '/' + dset, src_path + '/' + dset.replace('-', '_'))
            dset = dset.replace('-', '_')
        dset_path = src_path + '/' + dset
        print(f"Collecting statistic information of subset {dset} in {dset_path}")

        # looping for each speaker
        for spk in os.listdir(dset_path):
            if not spk.isdigit():
                continue
            spk_path = dset_path + '/' + spk

            # looping for each chapter made by the specific speaker
            for chp in os.listdir(spk_path):
                chp_path = spk_path + '/' + chp

                # looping for each audio file of the specific chapter
                for file in os.listdir(chp_path):
                    file_path = chp_path + '/' + file

                    # text data
                    if file.endswith('.txt'):
                        chp_text = np.loadtxt(file_path, delimiter="\n", dtype=str)
                        if len(chp_text.shape) == 0:
                            chp_text = np.array(chp_text.tolist().split(" ", 1)).reshape(1, -1)
                        else:
                            chp_text = np.stack(np.chararray.split(chp_text, maxsplit=1))

                        text[dset].append(chp_text)

                        if dset in ["train_clean_100", "train_clean_360"]:
                            text["train_clean_460"].append(chp_text)
                            text["train_960"].append(chp_text)

                        elif dset in ["train_other_500"]:
                            text["train_960"].append(chp_text)

                    # speech data
                    else:
                        scp_item = [file.split('.')[0], os.path.abspath(file_path)]
                        feat_scp[dset].append(scp_item)

                        spk_item = [file.split('.')[0], spk]
                        utt2spk[dset].append(spk_item)

                        if dset in ["train_clean_100", "train_clean_360"]:
                            feat_scp["train_clean_460"].append(scp_item)
                            feat_scp["train_960"].append(scp_item)
                            utt2spk["train_clean_460"].append(spk_item)
                            utt2spk["train_960"].append(spk_item)

                        elif dset in ["train_other_500"]:
                            feat_scp["train_960"].append(scp_item)
                            utt2spk["train_960"].append(spk_item)


    ### Collect speaker2gender data ###
    spk2gen = []
    spk_txt = open(src_path + '/SPEAKERS.TXT', 'r').readlines()[12:]
    for line in spk_txt:
        line = line.split("|")
        spk2gen.append([line[0].replace(" ", ""), line[1].replace(" ", "")])
    spk2gen = np.array(spk2gen)
    spk2gen = pd.Series(spk2gen[:, 1], index=spk2gen[:, 0])



    ### Save all the files of each subset in LibriSpeech ###
    for dset in feat_scp.keys():
        if not os.path.exists(f"{src_path}/{dset}"):
            os.makedirs(f"{src_path}/{dset}")
        print(f"Saving statistic information of subset {dset} to {src_path}/{dset}/")

        # feat.scp
        scp = np.array(feat_scp[dset], dtype=str)
        idx = scp[:, 0].argsort()
        scp = scp[idx]
        np.savetxt(f"{src_path}/{dset}/feat.scp", scp, fmt='%s')

        # utt2spk
        u2s = np.array(utt2spk[dset], dtype=str)
        u2s = u2s[idx]
        np.savetxt(f"{src_path}/{dset}/utt2spk", u2s, fmt='%s')

        # utt2gen
        u2s[:, 1] = np.array(spk2gen[u2s[:, 1]])
        np.savetxt(f"{src_path}/{dset}/utt2gen", u2s, fmt='%s')

        # text
        txt = np.concatenate(text[dset], axis=0)
        idx = txt[:, 0].argsort()
        txt = txt[idx]
        np.savetxt(f"{src_path}/{dset}/text", txt, fmt='%s')


def parse():
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--src_path', type=str, required=True)
    return parser.parse_args()
    pass


if __name__ == '__main__':
    main(parse())