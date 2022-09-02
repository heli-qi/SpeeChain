"""
    Author: Andros Tjandra
    Affiliation: NAIST (-2020)
    Date: 2017
"""

import numpy as np
import librosa
import librosa.filters
import soundfile
import math
import warnings
import random
from scipy import signal

#########################
### TACOTRON HELPER   ###
#########################
class TacotronHelper() :

    def __init__(self, cfg) :
        self.cfg = cfg
        if 'encoding' not in cfg:
            self.cfg['encoding']="PCM_16"
        pass

    def load_wav(self, path):
        wav = librosa.core.load(path, sr=self.cfg['sample_rate'])[0]
        if self.cfg.get('trim_db', None) is not None :
            print('trim_db\n\n')
            _, trim_idx = librosa.effects.trim(wav, top_db=self.cfg['trim_db'], frame_length=self.cfg['sample_rate']//10)
            if self.cfg.get('trim_part', None) is not None :
                assert len(set(self.cfg.get('trim_part')).difference(set(['start', 'end']))) == 0, "invalid trim_part parameter in feat_cfg"

                if self.cfg.get('trim_offset', None) is not None :
                    assert self.cfg['trim_offset'][0] < 0 and self.cfg['trim_offset'][1] > 0, "(negative, positive) number offset"
                    start_offset = int((self.cfg['trim_offset'][0] / 1000.0) * self.cfg['sample_rate'])
                    end_offset = int((self.cfg['trim_offset'][1] / 1000.0) * self.cfg['sample_rate'])
                    trim_idx[0] = max(trim_idx[0] + start_offset, 0)
                    trim_idx[1] = min(trim_idx[1] + end_offset, len(wav))

                if len(self.cfg['trim_part']) == 0 :
                    warnings.warn('no trimming was done in current config')
                if 'end' in self.cfg['trim_part'] :
                    wav = wav[0:trim_idx[1]]
                if 'start' in self.cfg['trim_part'] :
                    wav = wav[trim_idx[0]:]
        return wav

    def save_wav(self, wav, path) :
        soundfile.write(path, wav, self.cfg['sample_rate'], subtype=self.cfg['encoding'])
        '''
        wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        try :
            librosa.output.write_wav(path, wav.astype(np.int16), self.cfg['sample_rate'])
        except :
            soundfile.write(path, wav.astype(np.int16), self.cfg['sample_rate'])
        '''

    def preemphasis(self, x):
        return signal.lfilter([1, -self.cfg['preemphasis']], [1], x)


    def inv_preemphasis(self, x):
        return signal.lfilter([1], [1, -self.cfg['preemphasis']], x)


    def spectrogram(self, y):
        D = self._stft(self.preemphasis(y))
        S = self._amp_to_db(np.abs(D)) - self.cfg['ref_level_db']
        return self._normalize(S)


    def inv_spectrogram(self, spectrogram):
        '''Converts spectrogram to waveform using librosa'''
        S = self._db_to_amp(self._denormalize(spectrogram) + self.cfg['ref_level_db']) # Convert back to linear
        return self.inv_preemphasis(self._griffin_lim(S ** self.cfg['power'])) # Reconstruct phase

    def melspectrogram(self, y):
        D = self._stft(self.preemphasis(y))
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.cfg['ref_level_db']
        return self._normalize(S)


    # DO NOT USE THIS #
    def find_endpoint(self, wav, threshold_db=-40, min_silence_sec=0.8):
        window_length = int(self.cfg['sample_rate'] * min_silence_sec)
        hop_length = int(window_length / 4)
        threshold = self._db_to_amp(threshold_db)
        for x in range(hop_length, len(wav) - window_length, hop_length):
            if np.max(wav[x:x+window_length]) < threshold:
                return x + hop_length
        return len(wav)


    def _griffin_lim(self, S):
        '''librosa implementation of Griffin-Lim
        Based on https://github.com/librosa/librosa/issues/434
        '''
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex)
        y = self._istft(S_complex * angles)
        for i in range(self.cfg['griffin_lim_iters']):
            angles = np.exp(1j * np.angle(self._stft(y)))
            y = self._istft(S_complex * angles)
        return y

    def _stft(self, y):
        n_fft, hop_length, win_length = self._stft_parameters()
        return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


    def _istft(self, y):
        _, hop_length, win_length = self._stft_parameters()
        return librosa.istft(y, hop_length=hop_length, win_length=win_length)

    def _stft_parameters(self):
        n_fft = (self.cfg['num_freq'] - 1) * 2
        hop_length = int(self.cfg['frame_shift_ms'] / 1000 * self.cfg['sample_rate'])
        win_length = int(self.cfg['frame_length_ms'] / 1000 * self.cfg['sample_rate'])
        return n_fft, hop_length, win_length


    # Conversions:
    def _linear_to_mel(self, spectrogram):
        _mel_basis = None
        if _mel_basis is None:
            _mel_basis = self._build_mel_basis()
        return np.dot(_mel_basis, spectrogram)

    def _build_mel_basis(self):
        n_fft = (self.cfg['num_freq'] - 1) * 2
        return librosa.filters.mel(sr=self.cfg['sample_rate'], n_fft=n_fft, n_mels=self.cfg['num_mels'])

    def _amp_to_db(self, x):
        return 20 * np.log10(np.maximum(1e-5, x))

    def _db_to_amp(self, x):
        return np.power(10.0, x * 0.05)

    def _normalize(self, S):
        return np.clip((S - self.cfg['min_level_db']) / -self.cfg['min_level_db'], 0, 1)

    def _denormalize(self, S):
        return (np.clip(S, 0, 1) * -self.cfg['min_level_db']) + self.cfg['min_level_db']


    def cal_adjusted_rms(self, clean_rms, snr):
        a = float(snr) / 20
        noise_rms = clean_rms / (10**a) 
        return noise_rms

    def cal_amp(self, wf):
        buffer = wf.readframes(wf.getnframes())
        # The dtype depends on the value of pulse-code modulation. The int16 is set for 16-bit PCM.
        amptitude = (np.frombuffer(buffer, dtype="int16")).astype(np.float64)
        return amptitude

    def cal_rms(self, amp):
        return np.sqrt(np.mean(np.square(amp), axis=-1))

    def add_noise(self, clean_amp, noise_amp, snr):
        clean_rms = self.cal_rms(clean_amp)

        start = random.randint(0, len(noise_amp)-len(clean_amp))
        divided_noise_amp = noise_amp[start: start + len(clean_amp)]
        noise_rms = self.cal_rms(divided_noise_amp)

        adjusted_noise_rms = self.cal_adjusted_rms(clean_rms, snr)
        
        adjusted_noise_amp = divided_noise_amp * (adjusted_noise_rms / noise_rms) 
        return (clean_amp + adjusted_noise_amp)


####################################
### MULTISPEAKER TACOTRON HELPER ###
####################################

