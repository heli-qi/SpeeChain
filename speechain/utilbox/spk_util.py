import os
from typing import Dict, List, Union

import librosa
import numpy as np
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from tqdm import tqdm

from speechain.utilbox.data_loading_util import read_data_by_path
from speechain.utilbox.data_saving_util import save_data_by_format
from speechain.utilbox.import_util import parse_path_args
from speechain.utilbox.tensor_util import to_cpu


def extract_spk_feat(idx2wav: Dict, gpu_id: int, spk_emb_model: str, save_path: str = None, batch_size: int = 10,
                     downsample_package: str = 'librosa') -> Dict:
    """
        Extract speaker features using a specified speaker embedding model and save them.

        Args:
            idx2wav (Dict):
                A dictionary mapping unique IDs to waveform file paths.
            gpu_id (int):
                The GPU device ID to use. Set to -1 for CPU.
            spk_emb_model (str):
                The speaker embedding model to use (either 'ecapa' or 'xvector').
            save_path (str, optional):
                The path to save the extracted speaker features. If not given, the extracted features will be stored
                in memory. Defaults to None.
            batch_size (int, optional):
                The batch size for processing. Defaults to 10.
            downsample_package (str, optional):
                The library to use for downsampling ('torchaudio' or 'librosa'). Defaults to 'librosa'.

        Returns:
            Dict: A dictionary mapping unique IDs to the corresponding extracted speaker features.
        """

    def proc_curr_batch():
        """
            Process the current batch of waveforms and extract speaker features.
        """
        idx_list, wav_list = [i[0] for i in curr_batch], [i[1] for i in curr_batch]
        wav_len = torch.LongTensor([w.size(0) for w in wav_list]).to(device)
        max_wav_len = wav_len.max().item()

        # Pad feature vectors into a matrix
        wav_matrix = torch.zeros((wav_len.size(0), max_wav_len), device=device)
        for i in range(len(wav_list)):
            wav_matrix[i][:wav_len[i]] = wav_list[i]

        spk_feat = speechbrain_model.encode_batch(wavs=wav_matrix, wav_lens=wav_len / max_wav_len)
        if save_path is None:
            idx2spk_feat.update(dict(
                zip(idx_list, [to_cpu(s_f, tgt='numpy') for s_f in spk_feat]))
            )
        else:
            idx2spk_feat.update(save_data_by_format(
                file_format='npy', save_path=save_path, file_name_list=idx_list,
                file_content_list=[to_cpu(s_f, tgt='numpy') for s_f in spk_feat])
            )

    # initialize the speaker embedding model and downloading path for speechbrain API
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

    device = f"cuda:{gpu_id}" if gpu_id >= 0 else 'cpu'
    speechbrain_args.update(
        run_opts=dict(device=device)
    )
    speechbrain_model = EncoderClassifier.from_hparams(**speechbrain_args)

    curr_batch, wav_results, idx2spk_feat, resamplers = [], [], {}, {}
    for idx, wav_path in tqdm(idx2wav.items()):
        # Collect the data into the current batch
        wav, sample_rate = read_data_by_path(wav_path, return_tensor=True, return_sample_rate=True)
        wav = wav.squeeze(-1).to(device)
        if sample_rate > 16000:
            if downsample_package == 'torchaudio':
                if sample_rate not in resamplers.keys():
                    resamplers[sample_rate] = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000).to(device)
                wav = resamplers[sample_rate](wav)
            elif downsample_package == 'librosa':
                wav = librosa.resample(to_cpu(wav, tgt='numpy'), orig_sr=sample_rate, target_sr=16000)
                wav = torch.from_numpy(wav).to(device)
            else:
                raise ValueError

        elif sample_rate < 16000:
            raise RuntimeError

        curr_batch.append([idx, wav])
        # Process the batch if it meets the given size
        if len(curr_batch) == batch_size:
            proc_curr_batch()
            # refresh the current batch
            curr_batch = []

    # Process the remaining incomplete batch
    if len(curr_batch) != 0:
        proc_curr_batch()

    return idx2spk_feat


def average_spk_feat(spk_list: List[str], idx2spk: Dict[str, str], idx2spk_feat: Dict[str, Union[str, np.ndarray]],
                     save_path: str = None) -> Dict:
    """
        Compute the average speaker features for a list of speakers.

        Args:
            spk_list (List[str]):
                A list of speaker IDs to compute average features for.
            idx2spk (Dict[str, str]):
                A dictionary mapping unique feature IDs to speaker IDs.
            idx2spk_feat (Dict[str, Union[str, np.ndarray]]):
                A dictionary mapping unique feature IDs to speaker features or their file paths.
            save_path (str):
                The path to save the average speaker features. If not given, the computed average features will be
                stored in memory. Defaults to None.

        Returns:
            Dict: A dictionary mapping speaker IDs to their corresponding average speaker features or physical storage
            address depend on in_memory is set to True or False.
    """
    spk2aver_spk_feat = {}
    for spk_id in tqdm(spk_list):
        # Compute the average speaker features for the current speaker
        aver_spk_feat = np.mean([read_data_by_path(idx2spk_feat[spk_feat_id])
                                 if isinstance(idx2spk_feat[spk_feat_id], str) else idx2spk_feat[spk_feat_id]
                                 for spk_feat_id in idx2spk_feat.keys() if idx2spk[spk_feat_id] == spk_id], axis=0)

        # Save the average speaker features in memory or to disk
        if save_path is None:
            spk2aver_spk_feat[spk_id] = aver_spk_feat
        else:
            spk2aver_spk_feat.update(
                save_data_by_format(
                    file_format='npy', save_path=save_path, file_name_list=spk_id, file_content_list=aver_spk_feat
                )
            )
    return spk2aver_spk_feat
