"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.12
"""
import argparse
import os
import librosa
import torch
import numpy as np

from functools import partial
from multiprocessing import Pool
from typing import Dict, List

from speechbrain.pretrained import EncoderClassifier

from speechain.utilbox.data_loading_util import parse_path_args, load_idx2data_file, read_data_by_path
from speechain.utilbox.data_saving_util import save_data_by_format
from speechain.utilbox.import_util import get_idle_gpu
from speechain.utilbox.tensor_util import to_cpu


def parse():
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument("--src_file", type=str, required=True,
                        help="The path of the idx2wav file for the source waveforms from which you want to extract the "
                             "speaker embeddings. This argument is required.")
    parser.add_argument("--result_path", type=str, default=None,
                        help="The path where the extracted speaker embeddings are placed. If not given, the speaker "
                             "embeddings will be saved to the same directory as 'src_file'. (default: None)")
    parser.add_argument("--spk_emb_model", type=str, required=True,
                        help="The speaker recognition model you want to use to extract the speaker embeddings. This "
                             "argument is required and its value must be either 'xvector' or 'ecapa'.")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="The number of utterances you want to pass to the speaker embedding model in a batch for "
                             "parallel computation. (default: 1)")
    parser.add_argument("--ncpu", type=int, default=8,
                        help="The number of processes you want to use to extract speaker embeddings. If ngpu is given, "
                             "this argument won't be used. (default: 8)")
    parser.add_argument("--ngpu", type=int, default=0,
                        help="The number of GPUs you want to use to extract speaker embeddings. If not given, the "
                             "extraction will be done by CPUs. (default: 0)")

    return parser.parse_args()


def proc_curr_batch(curr_batch: List, device: str, spk_emb_func, save_path: str) -> Dict[str, str]:

    idx_list, wav_list = [j[0] for j in curr_batch], [j[1] for j in curr_batch]
    wav_len = torch.LongTensor([w.size(0) for w in wav_list]).to(device)
    batch_size, max_wav_len, wav_dim = len(wav_list), wav_len.max().item(), wav_list[0].size(-1)

    # padding all the feature vectors into a matrix
    wav_matrix = torch.zeros((batch_size, max_wav_len, wav_dim), device=device)
    for j in range(len(wav_list)):
        wav_matrix[j][:wav_len[j]] = wav_list[j]

    spk_feat = spk_emb_func(wavs=wav_matrix.squeeze(-1), wav_lens=wav_len / max_wav_len)
    idx2spk_feat = save_data_by_format(file_format='npy', save_path=save_path, file_name_list=idx_list,
                                       file_content_list=[to_cpu(s_f, tgt='numpy') for s_f in spk_feat])
    return idx2spk_feat


def extract_spk_feat(idx2wav: Dict, gpu_id: int, batch_size: int, speechbrain_args: Dict, save_path: str):

    device = f"cuda:{gpu_id}" if gpu_id >= 0 else 'cpu'
    speechbrain_args.update(
        run_opts=dict(device=device)
    )
    speechbrain_model = EncoderClassifier.from_hparams(**speechbrain_args)
    kwargs = dict(device=device, spk_emb_func=speechbrain_model.encode_batch, save_path=save_path)

    curr_batch, wav_results, idx2spk_feat = [], [], {}
    for i, (idx, wav_path) in enumerate(idx2wav.items()):
        # collect the data into the current batch
        wav, sample_rate = read_data_by_path(wav_path, return_tensor=True, return_sample_rate=True)
        wav = wav.to(device)
        if sample_rate > 16000:
            wav = librosa.resample(to_cpu(wav.squeeze(-1), tgt='numpy'), orig_sr=sample_rate, target_sr=16000)
            wav = torch.from_numpy(wav).unsqueeze(-1).to(device)
        elif sample_rate < 16000:
            raise RuntimeError()
        curr_batch.append([idx, wav])

        # process the batch if it meets the given size
        if len(curr_batch) == batch_size:
            _idx2spk_feat = proc_curr_batch(curr_batch=curr_batch, **kwargs)
            idx2spk_feat.update(_idx2spk_feat)
            # refresh the current batch
            curr_batch = []

    # the redundant incomplete batch
    if len(curr_batch) != 0:
        _idx2spk_feat = proc_curr_batch(curr_batch=curr_batch, **kwargs)
        idx2spk_feat.update(_idx2spk_feat)

    return idx2spk_feat


def main(src_file: str, result_path: str, spk_emb_model: str, batch_size: int, ncpu: int, ngpu: int):

    # initialize the result path
    src_idx2wav = parse_path_args(src_file)
    if result_path is None:
        result_path = os.path.dirname(src_idx2wav)
    else:
        result_path = parse_path_args(result_path)
    # read the source idx2wav file into a Dict, str -> Dict[str, str]
    src_idx2wav = load_idx2data_file(src_idx2wav)

    # initialize the speaker embedding model and downloading path for speechbrain API
    spk_emb_model = spk_emb_model.lower()
    download_dir = parse_path_args("datasets/speech_text/spk_emb_models")
    if spk_emb_model == 'ecapa':
        speechbrain_args = dict(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.join(download_dir, 'spkrec-ecapa-voxceleb')
        )
    elif spk_emb_model == 'xvector':
        speechbrain_args = dict(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir=os.path.join(download_dir, 'spkrec-xvect-voxceleb')
        )
    else:
        raise ValueError(f"Unknown speaker embedding model ({spk_emb_model})! "
                         f"Currently, spk_emb_model should be one of ['ecapa', 'xvector'].")

    # initialize the arguments for the execution function
    extract_spk_feat_func = partial(extract_spk_feat, batch_size=batch_size, speechbrain_args=speechbrain_args,
                                    save_path=os.path.join(result_path, spk_emb_model))

    device_list = get_idle_gpu(ngpu, id_only=True) if ngpu > 0 else [-1 for _ in range(ncpu)]
    n_proc = len(device_list) if ngpu > 0 else ncpu
    src_idx2wav = list(src_idx2wav.items())
    func_args = [[dict(src_idx2wav[i::n_proc]), device_list[i]] for i in range(n_proc)]

    # # debugging use
    # extraction_results = [extract_spk_feat_func(*i) for i in func_args]

    # start the executing jobs
    with Pool(n_proc) as executor:
        extraction_results = executor.starmap(extract_spk_feat_func, func_args)

    # gather the results from all the processes
    idx2spk_feat = {}
    for _idx2spk_feat in extraction_results:
        idx2spk_feat.update(_idx2spk_feat)

    # record the address of the .npy file of each vector
    np.savetxt(os.path.join(result_path, f'idx2{spk_emb_model}_spk_feat'),
               sorted(idx2spk_feat.items(), key=lambda x: x[0]), fmt='%s')


if __name__ == '__main__':
    args = parse()
    main(**vars(args))
