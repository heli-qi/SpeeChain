import os
from typing import Dict, Tuple

import numpy as np
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from tqdm import tqdm

from speechain.utilbox.data_loading_util import read_data_by_path
from speechain.utilbox.data_saving_util import save_data_by_format
from speechain.utilbox.import_util import parse_path_args
from speechain.utilbox.tensor_util import to_cpu


def extract_spk_feat(spk2wav_dict: Dict[str, Dict[str, str]], gpu_id: int, spk_emb_model: str,
                     save_path: str = None, batch_size: int = 10) -> Tuple[Dict, Dict]:
    """
        Extract speaker features using a specified speaker embedding model and save them.

        Args:
            spk2wav_dict (Dict):
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

        Returns:
            Tuple[Dict, Dict]:
                - A dictionary mapping unique IDs to the corresponding extracted speaker features.
                - A dictionary mapping speaker IDs to the corresponding average speaker features.
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
                file_format='npy', save_path=save_path, group_ids=spk_id, file_name_list=idx_list,
                file_content_list=[to_cpu(s_f, tgt='numpy') for s_f in spk_feat])
            )

        # refresh the current batch
        return []


    # initialize the speaker embedding model and downloading path for speechbrain API
    download_dir = parse_path_args("datasets/spk_emb_models")
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

    idx2spk_feat, spk2aver_spk_feat, resamplers = {}, {}, {}
    # loop each speaker
    for spk_id, wav_dict in tqdm(spk2wav_dict.items()):
        # a batch contains only waveforms for a single speaker
        curr_batch = []
        # loop each waveform file for each speaker
        for wav_idx, wav_path in wav_dict.items():
            # Collect the data into the current batch
            wav, sample_rate = read_data_by_path(wav_path, return_tensor=True, return_sample_rate=True)
            wav = wav.squeeze(-1).to(device)
            if sample_rate > 16000:
                if sample_rate not in resamplers.keys():
                    resamplers[sample_rate] = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000).to(device)
                wav = resamplers[sample_rate](wav)

            elif sample_rate < 16000:
                raise RuntimeError

            curr_batch.append([wav_idx, wav])
            # Process the batch if it meets the given size
            if len(curr_batch) == batch_size:
                # refresh the current batch
                curr_batch = proc_curr_batch()

        # Process the remaining incomplete batch
        if len(curr_batch) != 0:
            curr_batch = proc_curr_batch()

        # calculate the average speaker embedding for each speaker
        spk_feat_list, failed_idx_list = [], []
        # loop each waveform file for each speaker and check the saved data
        for wav_idx in wav_dict.keys():
            if isinstance(idx2spk_feat[wav_idx], str):
                try:
                    spk_feat_list.append(read_data_by_path(idx2spk_feat[wav_idx]))
                except ValueError:
                    # record the waveform index whose speaker embedding dumping failed
                    failed_idx_list.append(wav_idx)
            else:
                spk_feat_list.append(idx2spk_feat[wav_idx])

        # keep looping until no failed waveforms remain
        while len(failed_idx_list) != 0:
            # loop each failed waveform
            for wav_idx in failed_idx_list:
                wav, sample_rate = read_data_by_path(wav_dict[wav_idx], return_tensor=True, return_sample_rate=True)
                wav = wav.squeeze(-1).to(device)
                if sample_rate > 16000:
                    wav = resamplers[sample_rate](wav)
                curr_batch.append([wav_idx, wav])

            # reprocess the failed waveforms and check again
            curr_batch = proc_curr_batch()
            for wav_idx in failed_idx_list:
                try:
                    spk_feat_list.append(idx2spk_feat[wav_idx])
                except ValueError:
                    # the waveform remains in the failed list if the error happens again
                    pass
                else:
                    # remove the waveform if no error happens
                    failed_idx_list.remove(wav_idx)

        aver_spk_feat = np.mean(spk_feat_list, axis=0)
        # Save the average speaker features in memory or to disk
        if save_path is None:
            spk2aver_spk_feat[spk_id] = aver_spk_feat
        else:
            spk2aver_spk_feat.update(
                save_data_by_format(
                    file_format='npy', save_path=save_path, group_ids=spk_id, file_name_list=spk_id,
                    file_content_list=aver_spk_feat
                )
            )
    return idx2spk_feat, spk2aver_spk_feat
