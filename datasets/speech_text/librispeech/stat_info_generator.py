"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import argparse
import os
import numpy as np
import pandas as pd


def parse():
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--src_path', type=str, default="/ahc/work4/heli-qi/euterpe-heli-qi/datasets/speech_text/librispeech/data/raw")
    return parser.parse_args()


def main(src_path: str):
    """
    Extract the statistical information from each subset of the LibriSpeech dataset.
    Statistical information contains speech and text data as well as meta data (speaker id and gender).

    Args:
        src_path: str
            The path where the original dataset is placed.

    """
    # register Dicts for the dataset information
    idx2wav = {
        "train_clean_100": [],
        "train_clean_360": [],
        "train_other_500": [],
        "dev_clean": [],
        "dev_other": [],
        "test_clean": [],
        "test_other": []
    }

    idx2spk = {
        "train_clean_100": [],
        "train_clean_360": [],
        "train_other_500": [],
        "dev_clean": [],
        "dev_other": [],
        "test_clean": [],
        "test_other": []
    }

    idx2gen = {
        "train_clean_100": [],
        "train_clean_360": [],
        "train_other_500": [],
        "dev_clean": [],
        "dev_other": [],
        "test_clean": [],
        "test_other": []
    }

    idx2sent = {
        "train_clean_100": [],
        "train_clean_360": [],
        "train_other_500": [],
        "dev_clean": [],
        "dev_other": [],
        "test_clean": [],
        "test_other": []
    }


    # --- Collect speech and text data --- #
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

                        idx2sent[dset].append(chp_text)

                    # speech data
                    else:
                        scp_item = [file.split('.')[0], os.path.abspath(file_path)]
                        idx2wav[dset].append(scp_item)

                        spk_item = [file.split('.')[0], spk]
                        idx2spk[dset].append(spk_item)


    # --- Collect meta data (speaker id and gender) --- #
    spk2gen = []
    spk_txt = open(src_path + '/SPEAKERS.TXT', 'r').readlines()[12:]
    for line in spk_txt:
        line = line.split("|")
        spk2gen.append([line[0].replace(" ", ""), line[1].replace(" ", "")])
    spk2gen = np.array(spk2gen)
    spk2gen = pd.Series(spk2gen[:, 1], index=spk2gen[:, 0])


    # --- Save all the statistical information of each subset in LibriSpeech --- #
    for dset in idx2wav.keys():
        os.makedirs(f"{src_path}/{dset}", exist_ok=True)
        print(f"Saving statistic information of subset {dset} to {src_path}/{dset}/")

        # idx2wav
        np.savetxt(f"{src_path}/{dset}/idx2wav", sorted(idx2wav[dset], key=lambda x:x[0]), fmt='%s')

        # idx2spk
        np.savetxt(f"{src_path}/{dset}/idx2spk", sorted(idx2spk[dset], key=lambda x:x[0]), fmt='%s')

        # idx2gen
        idx = list(map(lambda x: x[0], idx2spk[dset]))
        gen = np.array(spk2gen[list(map(lambda x: x[1], idx2spk[dset]))]).tolist()
        idx2gen[dset] = list(zip(idx, gen))
        np.savetxt(f"{src_path}/{dset}/idx2gen", sorted(idx2gen[dset], key=lambda x:x[0]), fmt='%s')

        # idx2sent
        idx2sent[dset] = np.concatenate(idx2sent[dset], axis=0).tolist()
        _tmp = sorted(idx2sent[dset], key=lambda x: x[0])
        np.savetxt(f"{src_path}/{dset}/idx2sent", _tmp, fmt='%s')
        np.savetxt(f"{src_path}/{dset}/text", [item[1] for item in _tmp], fmt='%s')


    # --- train_clean_460 and train_960 data files production --- #
    os.makedirs(f"{src_path}/train_clean_460", exist_ok=True)
    os.makedirs(f"{src_path}/train_960", exist_ok=True)
    for file in [idx2wav, idx2spk, idx2gen, idx2sent]:
        file['train_clean_460'] = file['train_clean_100'] + file['train_clean_360']
        file['train_960'] = file['train_clean_460'] + file['train_other_500']

    # train_clean_460
    print(f"Saving statistic information of subset train_clean_460 to {src_path}/train_clean_460/")
    np.savetxt(f"{src_path}/train_clean_460/idx2wav", sorted(idx2wav['train_clean_460'], key=lambda x: x[0]), fmt='%s')
    np.savetxt(f"{src_path}/train_clean_460/idx2spk", sorted(idx2spk['train_clean_460'], key=lambda x: x[0]), fmt='%s')
    np.savetxt(f"{src_path}/train_clean_460/idx2gen", sorted(idx2gen['train_clean_460'], key=lambda x: x[0]), fmt='%s')
    _tmp = sorted(idx2sent['train_clean_460'], key=lambda x: x[0])
    np.savetxt(f"{src_path}/train_clean_460/idx2sent", _tmp, fmt='%s')
    np.savetxt(f"{src_path}/train_clean_460/text", [item[1] for item in _tmp], fmt='%s')

    # train_960
    print(f"Saving statistic information of subset train_960 to {src_path}/train_960/")
    np.savetxt(f"{src_path}/train_960/idx2wav", sorted(idx2wav['train_960'], key=lambda x: x[0]), fmt='%s')
    np.savetxt(f"{src_path}/train_960/idx2spk", sorted(idx2spk['train_960'], key=lambda x: x[0]), fmt='%s')
    np.savetxt(f"{src_path}/train_960/idx2gen", sorted(idx2gen['train_960'], key=lambda x: x[0]), fmt='%s')
    _tmp = sorted(idx2sent['train_960'], key=lambda x: x[0])
    np.savetxt(f"{src_path}/train_960/idx2sent", _tmp, fmt='%s')
    np.savetxt(f"{src_path}/train_960/text", [item[1] for item in _tmp], fmt='%s')


if __name__ == '__main__':
    args = parse()
    main(args.src_path)
