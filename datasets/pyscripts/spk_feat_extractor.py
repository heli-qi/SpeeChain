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

from tqdm import tqdm
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


def proc_curr_batch(curr_batch: List, device: str, spk_emb_func, save_path: str) -> Dict[str, str]:
    """
        Process the current batch of audio waveforms.

        Args:
            curr_batch (List):
                List of audio waveform tensors and their indices.
            device (str):
                Device to process the audio data on, either 'cpu' or 'cuda'.
            spk_emb_func:
                Function for extracting speaker embeddings from audio data.
            save_path (str):
                Path to save the extracted speaker embeddings.

        Returns:
            Dict[str, str]: Dictionary mapping audio waveform indices to extracted speaker embeddings.
    """
    idx_list, wav_list = [j[0] for j in curr_batch], [j[1] for j in curr_batch]
    wav_len = torch.LongTensor([w.size(0) for w in wav_list]).to(device)
    batch_size, max_wav_len, wav_dim = len(wav_list), wav_len.max().item(), wav_list[0].size(-1)

    # Pad feature vectors into a matrix
    wav_matrix = torch.zeros((batch_size, max_wav_len, wav_dim), device=device)
    for j in range(len(wav_list)):
        wav_matrix[j][:wav_len[j]] = wav_list[j]

    spk_feat = spk_emb_func(wavs=wav_matrix.squeeze(-1), wav_lens=wav_len / max_wav_len)
    idx2spk_feat = save_data_by_format(file_format='npy', save_path=save_path, file_name_list=idx_list,
                                       file_content_list=[to_cpu(s_f, tgt='numpy') for s_f in spk_feat])
    return idx2spk_feat


def extract_spk_feat(idx2wav: Dict, gpu_id: int, batch_size: int, speechbrain_args: Dict, save_path: str):
    """
        Extract speaker features from audio waveforms.

        Args:
            idx2wav (Dict):
                Dictionary mapping indices to audio waveform file paths.
            gpu_id (int):
                ID of the GPU to use for processing, or -1 to use CPU.
            batch_size (int):
                Number of utterances to process in a batch for parallel computation.
            speechbrain_args (Dict):
                Dictionary of arguments for the SpeechBrain model.
            save_path (str):
                Path to save the extracted speaker embeddings.

        Returns:
            Dict[str, str]: Dictionary mapping audio waveform indices to extracted speaker embeddings.
    """
    device = f"cuda:{gpu_id}" if gpu_id >= 0 else 'cpu'
    speechbrain_args.update(
        run_opts=dict(device=device)
    )
    speechbrain_model = EncoderClassifier.from_hparams(**speechbrain_args)
    kwargs = dict(device=device, spk_emb_func=speechbrain_model.encode_batch, save_path=save_path)

    curr_batch, wav_results, idx2spk_feat = [], [], {}
    for idx, wav_path in tqdm(idx2wav.items()):
        # Collect the data into the current batch
        wav, sample_rate = read_data_by_path(wav_path, return_tensor=True, return_sample_rate=True)
        wav = wav.to(device)
        if sample_rate > 16000:
            wav = librosa.resample(to_cpu(wav.squeeze(-1), tgt='numpy'), orig_sr=sample_rate, target_sr=16000)
            wav = torch.from_numpy(wav).unsqueeze(-1).to(device)
        elif sample_rate < 16000:
            raise RuntimeError()
        curr_batch.append([idx, wav])

        # Process the batch if it meets the given size
        if len(curr_batch) == batch_size:
            _idx2spk_feat = proc_curr_batch(curr_batch=curr_batch, **kwargs)
            idx2spk_feat.update(_idx2spk_feat)
            # refresh the current batch
            curr_batch = []

    # Process the remaining incomplete batch
    if len(curr_batch) != 0:
        _idx2spk_feat = proc_curr_batch(curr_batch=curr_batch, **kwargs)
        idx2spk_feat.update(_idx2spk_feat)

    return idx2spk_feat


def average_spk_feat(spk_list: List[str], idx2spk: Dict[str, str], idx2spk_feat: Dict[str, str], save_path: str):

    spk2aver_spk_feat = {}
    for spk_id in tqdm(spk_list):
        aver_spk_feat = np.mean([read_data_by_path(idx2spk_feat[spk_feat_id])
                          for spk_feat_id in idx2spk_feat.keys() if idx2spk[spk_feat_id] == spk_id], axis=0)
        spk2aver_spk_feat.update(
            save_data_by_format(
                file_format='npy', save_path=save_path, file_name_list=spk_id, file_content_list=aver_spk_feat
            )
        )

    return spk2aver_spk_feat


def main(src_file: str, result_path: str, spk_emb_model: str, batch_size: int, ncpu: int, ngpu: int):
    """
        Main function to extract speaker embeddings from audio waveforms.

        Args:
            src_file (str):
                Path to the idx2wav file containing the source waveforms for speaker embedding extraction.
            result_path (str):
                Path to save the extracted speaker embeddings.
            spk_emb_model (str):
                Speaker recognition model to use for extracting speaker embeddings.
            batch_size (int):
                Number of utterances to process in a batch for parallel computation.
            ncpu (int):
                Number of CPU processes for extracting speaker embeddings.
            ngpu (int):
                Number of GPUs for extracting speaker embeddings.
    """

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
        average_spk_feat_func = partial(average_spk_feat, save_path=os.path.join(result_path, f'aver_{spk_emb_model}'))
        spk_list = load_idx2data_file(os.path.join(os.path.dirname(src_idx2wav), 'spk_list'))
        spk_list = list(spk_list.values())
        func_args = [[spk_list[i::ncpu], idx2spk, idx2spk_feat] for i in range(ncpu)]

        # start the executing jobs
        with Pool(ncpu) as executor:
            average_results = executor.starmap(average_spk_feat_func, func_args)

        # gather the results from all the processes
        spk2aver_spk_feat = {}
        for _spk2aver_spk_feat in average_results:
            spk2aver_spk_feat.update(_spk2aver_spk_feat)

        # record the address of the .npy file of each vector
        np.savetxt(spk2aver_spk_feat_path, sorted(spk2aver_spk_feat.items(), key=lambda x: x[0]), fmt='%s')


if __name__ == '__main__':
    args = parse()
    main(**vars(args))
