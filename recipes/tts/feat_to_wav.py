"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.12
"""
import argparse
import os

import numpy as np
import torch

from tqdm import tqdm
from typing import Dict, List
from functools import partial
from multiprocessing import Pool
from speechbrain.pretrained import HIFIGAN

from speechain.utilbox.type_util import str2dict
from speechain.utilbox.data_loading_util import parse_path_args, load_idx2data_file, read_data_by_path, search_file_in_subfolder
from speechain.utilbox.yaml_util import load_yaml
from speechain.utilbox.import_util import get_idle_gpu
from speechain.utilbox.data_saving_util import save_data_by_format

from speechain.module.frontend.speech2mel import Speech2MelSpec
from speechain.module.frontend.speech2linear import Speech2LinearSpec


class SpeechBrainWrapper:
    """
    A wrapper class for the vocoder forward function of the speechbrain package.

    Before wrapping:
        feat -> vocode_func -> wav
    After wrapping:
        feat, feat_len -> SpeechBrainWrapper(vocode_func) -> wav, wav_len

    """
    def __init__(self, vocode_func):
        self.vocode_func = vocode_func

    def __call__(self, feat: torch.Tensor, feat_len: torch.Tensor):
        wav = self.vocode_func(feat.transpose(-2, -1)).transpose(-2, -1)
        # the lengths of the shorter utterances in the batch are estimated by their feature lengths
        wav_len = (feat_len * (wav.size(1) / feat.size(1))).long()
        # make sure that the redundant parts are set to silence
        for i in range(len(wav_len)):
            wav[i][wav_len[i]:] = 0
        return wav, wav_len


def parse():
    parser = argparse.ArgumentParser(description='params')

    # Shared Arguments
    group = parser.add_argument_group("Shared Arguments")
    group.add_argument('--vocoder', type=str, default='hifigan',
                       help="The type of the vocoder you want to use to generate waveforms. (default: hifigan)")
    group.add_argument('--feat_path', type=str, required=True,
                       help="The path of your TTS experimental folder. All the files named 'idx2feat' will be "
                            "automatically found out and used for waveform generation. You can also specify the path "
                            "of your target 'idx2feat' file by this argument.")
    group.add_argument('--wav_path', type=str, default=None,
                       help="The path where the generated waveforms are placed. If not given, the results will be "
                            "saved to the same directory as your given 'hypo_idx2feat'. (default: None)")
    group.add_argument('--batch_size', type=int, default=1,
                       help="The number of utterances you want to pass to the vocoder in a batch for parallel "
                            "computation. We recommend you to set this argument to 1 if you want to use hifigan for "
                            "vocoding for accurate waveform length recording. (default: 1)")
    group.add_argument('--ngpu', type=int, default=0,
                       help="The number of GPUs you want to use to generate waveforms. If not given, the vocoding "
                            "process will be done by CPUs. (default: 0)")
    group.add_argument('--ncpu', type=int, default=8,
                       help="The number of processes you want to use to generate waveforms. If ngpu is given, this "
                            "argument won't be used. (default: 8)")

    # GL-specific Arguments
    group = parser.add_argument_group("GL-specific Arguments")
    group.add_argument('--tts_model_cfg', type=str2dict, default=None,
                       help="The path of the configuration file of your TTS model for GL vocoding. If not given, the "
                            "file named 'train_cfg.yaml' will be automatically found out in your given 'feat_path'. "
                            "(default: None)")

    # Neural Vocoder-specific Arguments
    group = parser.add_argument_group("Neural Vocoder-specific Arguments")
    group.add_argument('--sample_rate', type=int, default=22050,
                       help="The sampling rate of generated waveforms. (default: 22050)")
    group.add_argument('--vocoder_train_data', type=str, default='ljspeech',
                       help="The dataset used to train the neural vocoder. "
                            "This argument is required if your input vocoder is not 'gl'. (default: ljspeech)")
    return parser.parse_args()


def proc_curr_batch(curr_batch: List, device: str, sample_rate: int, save_path: str, feat_to_wav_func):

    idx_list = [j[0] for j in curr_batch]
    feat_list = [j[1] for j in curr_batch]
    feat_len = torch.LongTensor([j[2] for j in curr_batch]).to(device)
    batch_size, max_feat_len, feat_dim = len(feat_list), feat_len.max().item(), feat_list[0].size(-1)

    # padding all the feature vectors into a matrix
    feat = torch.zeros((batch_size, max_feat_len, feat_dim), device=device)
    for j in range(len(feat_list)):
        feat[j][:feat_len[j]] = feat_list[j]

    # recover acoustic features back to waveforms
    wav, wav_len = feat_to_wav_func(feat, feat_len)
    idx2wav = save_data_by_format(file_format='wav', save_path=save_path, sample_rate=sample_rate,
                                  file_name_list=idx_list,
                                  file_content_list=[wav[i][:wav_len[i]] for i in range(len(wav))])
    idx2wav_len = dict(zip(idx_list, wav_len.tolist()))

    return idx2wav, idx2wav_len


def convert_feat_to_wav(idx2feat: Dict, device: str, batch_size: int, sample_rate: int, save_path: str,
                        feat_to_wav_func):

    curr_batch, wav_results, idx2wav, idx2wav_len = [], [], {}, {}
    kwargs = dict(device=device, sample_rate=sample_rate, save_path=save_path, feat_to_wav_func=feat_to_wav_func)

    for idx, feat_path in tqdm(idx2feat.items()):
        # collect the data into the current batch
        feat = read_data_by_path(feat_path, return_tensor=True).to(device)
        curr_batch.append([idx, feat, feat.size(0)])

        # process the batch if it meets the given size
        if len(curr_batch) == batch_size:
            _idx2wav, _idx2wav_len = proc_curr_batch(curr_batch=curr_batch, **kwargs)
            idx2wav.update(_idx2wav)
            idx2wav_len.update(_idx2wav_len)
            # refresh the current batch
            curr_batch = []

    # the redundant incomplete batch
    if len(curr_batch) != 0:
        _idx2wav, _idx2wav_len = proc_curr_batch(curr_batch=curr_batch, **kwargs)
        idx2wav.update(_idx2wav)
        idx2wav_len.update(_idx2wav_len)

    return idx2wav, idx2wav_len


def vocode_by_gl(idx2feat: Dict, gpu_id: int, batch_size: int, save_path: str, frontend_cfg: Dict) -> (Dict, Dict):

    if frontend_cfg['type'] == 'mel_fbank':
        frontend = Speech2MelSpec(**frontend_cfg['conf'])
    elif frontend_cfg['type'] == 'stft':
        frontend = Speech2LinearSpec(**frontend_cfg['conf'])
    else:
        raise ValueError()
    device = f"cuda:{gpu_id}" if gpu_id >= 0 else 'cpu'
    frontend = frontend.to(device)

    # convert idx2feat into idx2wav by batches
    idx2wav, idx2wav_len = convert_feat_to_wav(idx2feat=idx2feat, device=device, batch_size=batch_size,
                                               sample_rate=frontend.get_sample_rate(),
                                               save_path=os.path.join(save_path, 'gl_wav'),
                                               feat_to_wav_func=frontend.recover)
    return idx2wav, idx2wav_len


def vocode_by_hifigan(idx2feat: Dict, gpu_id: int,
                      batch_size: int, save_path: str, sample_rate: int, vocoder_train_data: str):
    """

    Args:
        idx2feat:
        gpu_id:
        batch_size:
        save_path:
        sample_rate:
        vocoder_train_data:

    Returns:

    """
    # initialize the HiFiGAN model
    device = f"cuda:{gpu_id}" if gpu_id >= 0 else 'cpu'
    download_dir = parse_path_args("recipes/tts/speechbrain_vocoder")
    if vocoder_train_data == 'ljspeech':
        hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", run_opts=dict(device=device),
                                        savedir=os.path.join(download_dir, 'hifigan-ljspeech'))
    elif vocoder_train_data == 'libritts':
        if sample_rate == 16000:
            hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz",
                                            savedir=os.path.join(download_dir, 'hifigan-libritts-16kHz'),
                                            run_opts=dict(device=device))
        elif sample_rate == 22050:
            hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-22050Hz",
                                            savedir=os.path.join(download_dir, 'hifigan-libritts-22050Hz'),
                                            run_opts=dict(device=device))
        else:
            raise ValueError
    else:
        raise NotImplementedError

    # convert idx2feat into idx2wav by batches
    idx2wav, idx2wav_len = convert_feat_to_wav(idx2feat=idx2feat, device=device, batch_size=batch_size,
                                               sample_rate=sample_rate,
                                               save_path=os.path.join(save_path, 'hifigan_wav'),
                                               feat_to_wav_func=SpeechBrainWrapper(hifi_gan.decode_batch))
    return idx2wav, idx2wav_len


def main(vocoder: str, feat_path: str, wav_path: str, batch_size: int, ngpu: int, ncpu: int,
         tts_model_cfg: Dict or str, sample_rate: int, vocoder_train_data: str):

    feat_path = parse_path_args(feat_path)
    # for folder input, automatically find out all the idx2feat candidates as hypo_idx2feat
    if os.path.isdir(feat_path):
        hypo_idx2feat_list = search_file_in_subfolder(feat_path, lambda x: x == 'idx2feat')
    # for file input, directly use it as hypo_idx2feat
    else:
        hypo_idx2feat_list = [feat_path]

    # loop each candidate idx2feat for hypothesis acoustic feature
    for hypo_idx2feat in hypo_idx2feat_list:
        if os.path.exists(os.path.join(os.path.dirname(hypo_idx2feat), f'idx2{vocoder}_wav')):
            print(f"The waveform files have already existed. So {hypo_idx2feat} will be skipped.")
        else:
            print(f"Start to generate the waveforms by {hypo_idx2feat}.")

            # initialize the result path
            hypo_idx2feat = parse_path_args(hypo_idx2feat)
            if wav_path is None:
                save_path = os.path.dirname(hypo_idx2feat)
            else:
                save_path = parse_path_args(wav_path)

            # go through different branches by vocoder
            vocoder = vocoder.lower()
            if vocoder == 'gl':
                if tts_model_cfg is None:
                    cand_model_cfg_list = search_file_in_subfolder(feat_path, lambda x: x == 'train_cfg.yaml')
                    if len(cand_model_cfg_list) == 1:
                        tgt_model_cfg = parse_path_args(cand_model_cfg_list[0])
                    else:
                        tgt_model_cfg = None
                        for cand_model_cfg in cand_model_cfg_list:
                            cand_model_cfg = parse_path_args(cand_model_cfg)
                            if hypo_idx2feat.startswith(os.path.dirname(cand_model_cfg)):
                                tgt_model_cfg = cand_model_cfg
                                break

                        if tgt_model_cfg is None:
                            raise RuntimeError(
                                f"None of the train_cfg.yaml files in {feat_path} matches the chosen idx2feat file "
                                f"{hypo_idx2feat}! Please check your arguments.")
                else:
                    tgt_model_cfg = parse_path_args(tts_model_cfg)

                tts_model_cfg_dict = load_yaml(tgt_model_cfg)
                if 'model' in tts_model_cfg_dict.keys():
                    frontend_cfg = tts_model_cfg_dict['model']['module_conf']['frontend']
                else:
                    frontend_cfg = tts_model_cfg_dict
                assert 'type' in frontend_cfg.keys() and 'conf' in frontend_cfg.keys(), \
                    "tts_model_cfg must contain 'type' and 'conf' as necessary key-value items!"

                vocode_func = partial(vocode_by_gl,
                                      batch_size=batch_size, save_path=save_path, frontend_cfg=frontend_cfg)

            elif vocoder == 'hifigan':
                assert vocoder_train_data is not None, \
                    "If you choose 'hifigan' as the vocoder, " \
                    "please specify 'vocoder_train_data' to pick up the target hifigan model file."
                vocoder_train_data = vocoder_train_data.lower()

                if vocoder_train_data == 'ljspeech':
                    assert sample_rate == 22050, \
                        "If you choose 'ljspeech' as vocoder_train_data, sample_rate must be 22050."
                elif vocoder_train_data == 'libritts':
                    assert sample_rate in [16000, 22050], \
                        "If you choose 'libritts' as vocoder_train_data, sample_rate must be either 16000 or 22050."
                else:
                    raise NotImplementedError(f"Unknown vocoder_train_data ({vocoder_train_data})! "
                                              f"vocoder_train_data should be one of ['ljspeech', 'libritts'].")

                vocode_func = partial(vocode_by_hifigan, batch_size=batch_size, save_path=save_path,
                                      sample_rate=sample_rate, vocoder_train_data=vocoder_train_data)

            else:
                raise NotImplementedError(
                    "Currently, we only support Griffin-Lim ('gl') and HiFiGAN ('hifigan') as the vocoder.")

            # read the idx2feat file into a Dict, str -> Dict[str, str]
            hypo_idx2feat = load_idx2data_file(hypo_idx2feat)

            # initialize the arguments for vocoder execution function
            device_list = get_idle_gpu(ngpu, id_only=True) if ngpu > 0 else [-1 for _ in range(ncpu)]
            n_proc = len(device_list) if ngpu > 0 else ncpu
            hypo_idx2feat_list = list(hypo_idx2feat.items())
            func_args = [[dict(hypo_idx2feat_list[i::n_proc]), device_list[i]] for i in range(n_proc)]

            # # debugging use
            # vocode_results = [vocode_func(*i) for i in func_args]

            # start the executing jobs
            with Pool(n_proc) as executor:
                vocode_results = executor.starmap(vocode_func, func_args)

            # gather the results from all the processes
            idx2wav, idx2wav_len = {}, {}
            for _idx2wav, _idx2wav_len in vocode_results:
                idx2wav.update(_idx2wav)
                idx2wav_len.update(_idx2wav_len)

            # save the address and length of each synthetic utterance
            idx2wav_path = os.path.join(save_path, f'idx2{vocoder}_wav')
            np.savetxt(idx2wav_path, sorted(idx2wav.items(), key=lambda x: x[0]), fmt='%s')
            print(f"idx2wav file has been saved to {idx2wav_path}.")

            idx2wav_len_path = os.path.join(save_path, f'idx2{vocoder}_wav_len')
            np.savetxt(idx2wav_len_path, sorted(idx2wav_len.items(), key=lambda x: x[0]), fmt='%s')
            print(f"idx2wav_len file has been saved to {idx2wav_len_path}.")

        print("\n")


if __name__ == '__main__':
    args = parse()
    main(**vars(args))
