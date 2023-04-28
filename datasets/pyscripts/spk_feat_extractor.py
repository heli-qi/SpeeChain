"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.12
"""
import argparse
import os
import numpy as np
from functools import partial
from multiprocessing import Pool

from speechain.utilbox.data_loading_util import parse_path_args, load_idx2data_file
from speechain.utilbox.import_util import get_idle_gpu
from speechain.utilbox.spk_util import extract_spk_feat, average_spk_feat


def parse():
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument("--src_file", type=str, required=True,
                        help="Path to the idx2wav file containing the source waveforms for speaker embedding extraction.")
    parser.add_argument("--result_path", type=str, default=None,
                        help="Path to save the extracted speaker embeddings. "
                             "If not provided, embeddings will be saved in the same directory as 'src_file'.")
    parser.add_argument("--spk_emb_model", type=str, required=True,
                        help="Speaker recognition model to use for extracting speaker embeddings. "
                             "Must be 'xvector' or 'ecapa'.")
    parser.add_argument('--batch_size', type=int, default=10,
                        help="Number of utterances to pass to the speaker embedding model in a batch for parallel "
                             "computation. (default: 10)")
    parser.add_argument("--ncpu", type=int, default=8,
                        help="Number of CPU processes for extracting speaker embeddings. "
                             "Ignored if 'ngpu' is provided. (default: 8)")
    parser.add_argument("--ngpu", type=int, default=0,
                        help="Number of GPUs for extracting speaker embeddings. "
                             "If not provided, extraction will be done on CPUs. (default: 0)")

    return parser.parse_args()

def main(src_file: str, result_path: str = None, spk_emb_model: str = 'ecapa', downsample_package: str = 'torchaudio',
         batch_size: int = 10, ncpu: int = 8, ngpu: int = 0):
    """
        Main function to extract speaker embeddings from audio waveforms.

        Args:
            src_file (str):
                Path to the idx2wav file containing the source waveforms for speaker embedding extraction.
            result_path (str):
                Path to save the extracted speaker embeddings.
            spk_emb_model (str):
                Speaker recognition model to use for extracting speaker embeddings.
            ncpu (int):
                Number of CPU processes for extracting speaker embeddings.
            ngpu (int):
                Number of GPUs for extracting speaker embeddings.
    """
    spk_emb_model = spk_emb_model.lower()
    assert spk_emb_model in ['ecapa', 'xvector'], \
        f"Your input spk_emb_model should be one of ['ecapa', 'xvector'], but got {spk_emb_model}!"

    # initialize the result path
    src_idx2wav = parse_path_args(src_file)
    if result_path is None:
        result_path = os.path.dirname(src_idx2wav)
    else:
        result_path = parse_path_args(result_path)

    # --- 1. Speaker Embedding Feature Extraction for Individual Utterances --- #
    idx2spk_feat_path = os.path.join(result_path, f'idx2{spk_emb_model}_spk_feat')
    # skip if idx2spk_feat has already existed
    if not os.path.exists(idx2spk_feat_path):
        # read the source idx2wav file into a Dict, str -> Dict[str, str]
        src_idx2wav = load_idx2data_file(src_idx2wav)

        # initialize the arguments for the execution function
        extract_spk_feat_func = partial(extract_spk_feat, spk_emb_model=spk_emb_model,
                                        downsample_package=downsample_package, batch_size=batch_size,
                                        save_path=os.path.join(result_path, spk_emb_model))

        device_list = get_idle_gpu(ngpu, id_only=True) if ngpu > 0 else [-1 for _ in range(ncpu)]
        n_proc = len(device_list) if ngpu > 0 else ncpu
        src_idx2wav = list(src_idx2wav.items())
        func_args = [[dict(src_idx2wav[i::n_proc]), device_list[i]] for i in range(n_proc)]

        # start the executing jobs
        with Pool(n_proc) as executor:
            extraction_results = executor.starmap(extract_spk_feat_func, func_args)

        # gather the results from all the processes
        idx2spk_feat = {}
        for _idx2spk_feat in extraction_results:
            idx2spk_feat.update(_idx2spk_feat)

        # record the address of the .npy file of each vector
        np.savetxt(idx2spk_feat_path, sorted(idx2spk_feat.items(), key=lambda x: x[0]), fmt='%s')

    # --- 2. Average Speaker Embedding Feature Extraction for Individual Speakers --- #
    spk2aver_spk_feat_path = os.path.join(result_path, f'spk2aver_{spk_emb_model}_spk_feat')
    if not os.path.exists(spk2aver_spk_feat_path):
        # read the idx2spk_feat file into a Dict, str -> Dict[str, str]
        idx2spk_feat = load_idx2data_file(idx2spk_feat_path)
        # read the idx2spk file in the directory of idx2wav into a Dict, str -> Dict[str, str]
        idx2spk = load_idx2data_file(os.path.join(os.path.dirname(src_idx2wav), 'idx2spk'))

        # initialize the arguments for the execution function
        average_spk_feat_func = partial(average_spk_feat,
                                        idx2spk=idx2spk, idx2spk_feat=idx2spk_feat,
                                        save_path=os.path.join(result_path, f'aver_{spk_emb_model}'))
        spk_list = load_idx2data_file(os.path.join(os.path.dirname(src_idx2wav), 'spk_list'))
        spk_list = list(spk_list.values())
        func_args = [spk_list[i::ncpu] for i in range(ncpu)]

        # start the executing jobs
        with Pool(ncpu) as executor:
            average_results = executor.map(average_spk_feat_func, func_args)

        # gather the results from all the processes
        spk2aver_spk_feat = {}
        for _spk2aver_spk_feat in average_results:
            spk2aver_spk_feat.update(_spk2aver_spk_feat)

        # record the address of the .npy file of each vector
        np.savetxt(spk2aver_spk_feat_path, sorted(spk2aver_spk_feat.items(), key=lambda x: x[0]), fmt='%s')


if __name__ == '__main__':
    args = parse()
    main(**vars(args))
