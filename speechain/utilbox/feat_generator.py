"""
    Author: Sashi Novitasari
    Affiliation: NAIST
    Date: 2022.07
"""
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import tempfile
from scipy.io import wavfile
import python_speech_features as pyspfeat
import librosa
import librosa.filters
from tqdm import tqdm
from speechain.utilbox import signal_util


############
### UTIL ###
############

def linear2mu(signal, mu=255) :
    """
    -1 < signal < 1 
    """
    assert signal.max() <= 1 and signal.min() >= -1
    y = np.sign(signal) * np.log(1.0 + mu * np.abs(signal))/np.log(mu+1)
    return y

### FUNC FOR NPZ -- CREATE & SAVE ###
def save_arr(key, feat, path, compress=False):
    if compress:
        np.savez_compressed(path, feat=feat, key=key)
    else:
        np.savez(path, feat=feat, key=key)
    return key, path

def gen_and_save_arr(key, wav_file, path, cfg, compress=False) :
    feat = generate_feat_opts(wav_file, cfg)
    # TODO : flexible np.floatx
    feat = feat.astype(np.float32)
    path = os.path.abspath(path)
    save_arr(key, feat, path, compress=compress)
    return key, path


##################
### ALL IN ONE ###
##################

def generate_feat_opts(path, cfg):
    '''

    Args:
        path:
        cfg:

    Returns:

    '''
    cfg = dict(cfg)

    if cfg['pkg'] == 'pysp' : # python_speech_features #
        rate, signal = wavfile.read(path)
            
        if cfg['type'] == 'logfbank' :
            feat_mat = pyspfeat.base.logfbank(signal, rate, nfilt=cfg.get('nfilt', 40))
        elif cfg['type'] == 'mfcc' :
            feat_mat = pyspfeat.base.mfcc(signal, rate,
                    numcep=cfg.get('nfilt', 26)//2, nfilt=cfg.get('nfilt', 26))
        elif cfg['type'] == 'wav' :
            feat_mat = pyspfeat.base.sigproc.framesig(signal, 
                    frame_len=cfg.get('frame_len', 400), 
                    frame_step=cfg.get('frame_step', 160))
        else :
            raise NotImplementedError("feature type {} is not implemented/available".format(cfg['type']))
            pass
        # delta #
        comb_feat_mat = [feat_mat]
        delta = cfg['delta']
        if delta > 0 :
            delta_feat_mat = pyspfeat.base.delta(feat_mat, 2)
            comb_feat_mat.append(delta_feat_mat)
        if delta > 1 :
            delta2_feat_mat = pyspfeat.base.delta(delta_feat_mat, 2)
            comb_feat_mat.append(delta2_feat_mat)
        if delta > 2 :
            raise NotImplementedError("max delta is 2, larger than 2 is not normal setting")
        return np.hstack(comb_feat_mat)

    elif cfg['pkg'] == 'rosa':
        # load the audio signal file, signal downsampling is done here
        signal, rate = librosa.core.load(path, sr=cfg['sample_rate'])
        assert rate == cfg['sample_rate'], "sample rate is different with current data"

        # apply the preemphasis to the audio signal
        if cfg.get('preemphasis', None) is not None:
            signal = signal_util.preemphasis(signal, cfg['preemphasis'])

        # apply the pre-normalization to the audio signal
        if cfg.get('pre', None) == 'meanstd':
            signal = (signal - signal.mean()) / signal.std()
        elif cfg.get('pre', None) == 'norm':
            signal = (signal - signal.min()) / (signal.max() - signal.min()) * 2 - 1 

        ### waveform-based acoustic feature ###
        if cfg['type'] == 'wav' :
            if cfg.get('post', None) == 'mu':
                signal = linear2mu(signal)
            
            feat_mat = pyspfeat.base.sigproc.framesig(signal, 
                    frame_len=cfg.get('frame_len', 400), 
                    frame_step=cfg.get('frame_step', 160))
            return feat_mat

        ### spectrogram-based acoustic feature ###
        # extract linear spectrogram from signal
        raw_spec = signal_util.rosa_spectrogram(signal,
                                                n_fft=cfg.get('n_fft', 2048),
                                                hop_length=cfg.get('hop_length', None),
                                                win_length=cfg.get('win_length', None),
                                                power=cfg.get('power', 2))[0]

        # extract mel-scale spectrogram from linear spectrogram
        if cfg['type'] in ['logmelfbank', 'melfbank']:
            mel_spec = signal_util.rosa_spec2mel(raw_spec,
                                                 sr=cfg.get('sample_rate', 16000),
                                                 n_mels=cfg.get('n_mels', 80),
                                                 fmin=cfg.get('fmin', 0),
                                                 fmax=cfg.get('fmax', None),
                                                 clip=cfg.get('mel_clip', 1e-10))
            # apply the logrithm operation
            if cfg['type'] == 'logmelfbank':
                log_base = cfg.get('log_base', None)
                if log_base is None:
                    return np.log(mel_spec)
                elif log_base == 2:
                    return np.log2(mel_spec)
                elif log_base == 10:
                    return np.log10(mel_spec)
                else:
                    return np.log(mel_spec) / np.log(log_base)
            else :
                return mel_spec
        # log-scale linear spectrogram
        elif cfg['type'] == 'lograwfbank':
            return np.log(raw_spec)
        # linear spectrogram
        elif cfg['type'] == 'rawfbank' :
            return raw_spec
        # MFCC
        elif cfg['type'] == 'mfcc' :
            mfcc = librosa.feature.mfcc(S=librosa.power_to_db(raw_spec), n_mfcc=cfg['n_mfcc'])
            return mfcc
            # edit: sashi (S--> log mel spectrogram https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html)
            # mel_spec = signal_util.rosa_spec2mel(raw_spec, nfilt=cfg['nfilt'])
            # mfcc = librosa.feature.mfcc(S=np.log(mel_spec), n_mfcc=cfg['n_mfcc'])
            # return np.transpose(mfcc)
        else :
            raise NotImplementedError()

    elif cfg['pkg'] == 'taco' :
        # SPECIAL FOR TACOTRON #
        tacohelper = TacotronHelper(cfg)
        signal = tacohelper.load_wav(path)
        if "snr" in cfg.keys():
            print('snr')
            noise = np.random.randn(len(signal))
            signal = tacohelper.add_noise(signal, noise, cfg['snr'])

        assert len(signal) != 0, ('file {} is empty'.format(path))

        try:
            if cfg['type'] == 'raw' : 
                feat = tacohelper.spectrogram(signal).T
            elif cfg['type'] == 'mel' :
                feat = tacohelper.melspectrogram(signal).T
            elif 'mel' in cfg['type'] and 'pitch' in cfg['type']:

                feat_mel = tacohelper.melspectrogram(signal).T
                feat_pitch = librosa.core.piptrack(y=signal)
                
                #print(feat_mel.shape)
                #print(feat_pitch.shape)
                #if 'pitch_delta' in cfg['type']:
                #    feat_pitch_delta = librosa.feature.delta(feat_pitch)
            else :
                raise NotImplementedError()
        except :
            import ipdb; ipdb.set_trace()
            pass

        if 'delta' in cfg.keys():
            
            if cfg['delta']>=1:
                feat_size = feat.shape[1]
                feat_d = librosa.feature.delta(feat)
                feat = np.concatenate((feat,feat_d),axis=1)
            
            if cfg['delta']==2:
                feat_dd = librosa.feature.delta(feat[:,feat_size:])
                feat = np.concatenate((feat,feat_dd),axis=1)

        return feat

    elif cfg['pkg'] == 'world' :
        if path is None :
            with tempfile.NamedTemporaryFile() as tmpfile :
                wavfile.write(tmpfile.name, rate, signal)
                logf0, bap, mgc = world_vocoder_util.world_analysis(tmpfile.name, cfg['mcep'])
        else :
            logf0, bap, mgc = world_vocoder_util.world_analysis(path, cfg['mcep'])
        
        vuv, f0, bap, mgc = world_vocoder_util.world2feat(logf0, bap, mgc)

        # ignore delta, avoid curse of dimensionality #
        return vuv, f0, bap, mgc

    else :
        raise NotImplementedError()
        pass


def generate_feat_standard_npfile(key_list, wav_list, output_path, cfg, ncpu=16, compress=False):
    '''
    Extract acoustic features from raw waveform files and save the features in the form of numpy files.

    Args:
        key_list: List or numpy.array
            The list of waveform file names.
        wav_list: List or numpy.array
            The list of waveform file addresses.
        output_path: str
            The output path used to place the extracted feature files.
        cfg:
            The config file of extracting acoustic features
        ncpu: int
            The number of cpu used to extract acoustic features
        compress: bool
            Whether feature files are compressed or not.

    Returns:
        None
    '''
    assert len(key_list) == len(wav_list)
    assert len(set(key_list)) == len(set(wav_list)), "number of key & wav unique set is not same"
    assert cfg['pkg'] in ['pysp', 'rosa', 'taco']
    if os.path.exists(output_path):
        assert os.path.isdir(output_path), "output_path must be a folder"
    else :
        os.makedirs(output_path, mode=0o755, exist_ok=False)

    executor = ProcessPoolExecutor(max_workers=ncpu)
    list_execs = []
    file_kv = open(os.path.join(output_path, 'feat.scp'), 'w')

    for ii in range(len(key_list)):
        path_ii = os.path.join(output_path, key_list[ii]+'.npz')
        # # For run
        list_execs.append(executor.submit(partial(gen_and_save_arr, key_list[ii], wav_list[ii], path_ii, cfg=cfg, compress=compress)))
        # # For debug
        # print(f"{ii}/{len(key_list)}")
        # gen_and_save_arr(key_list[ii], wav_list[ii], path_ii, cfg, compress)


    list_result = []
    for item in tqdm(list_execs) :
        list_result.append(item.result())
    for item in list_result :
        file_kv.write('{} {}\n'.format(*item))
    file_kv.flush()
    file_kv.close()
    pass



import argparse
import yaml
import time

def parse():
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--wav_scp', type=str, default="../../egs/librispeech/data/raw/train_clean_100/wav.scp")
    parser.add_argument('--output_path', type=str, default="../../egs/librispeech/data/rosa_16k_logmel_v1/train_clean_100")
    parser.add_argument('--cfg', type=str, default="../../config/feat/rosa_16k_logmel_v1.json", help='feature config file')
    parser.add_argument('--ncpu', type=int, default=32)
    parser.add_argument('--compress', action='store_true', default=False)
    parser.add_argument('--type', type=str, choices=['h5', 'npz'], default='npz')
    return parser.parse_args()
    pass

if __name__ == '__main__':
    args = parse()
    output_path = args.output_path
    cfg = yaml.load(open(args.cfg))

    wav_scp = np.loadtxt(args.wav_scp, dtype=str)
    key_list = wav_scp[:, 0]
    wav_list = wav_scp[:, 1]

    start_time = time.time()
    if cfg['pkg'] in ['pysp', 'rosa', 'taco'] :
        generate_feat_standard_npfile(key_list,
                                      wav_list,
                                      output_path=output_path,
                                      cfg=cfg,
                                      ncpu=args.ncpu,
                                      compress=args.compress)
    else :
        raise ValueError("pkg name is not included")
        pass

    print("Time elapsed %.2f secs"%(time.time() - start_time))
    print("=== FINISH ===")
    pass