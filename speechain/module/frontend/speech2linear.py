"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import torch
import numpy as np
import torch.nn.functional as F
from speechain.module.abs import Module

class Speech2LinearSpec(Module):
    """
    The acoustic frontend where the input is raw speech waveforms and the output is linear spectrogram.

    """
    def module_init(self,
                    n_fft: int,
                    hop_length: int,
                    win_length: int,
                    preemphasis: float = None,
                    pre_stft_norm: str = None,
                    window: str = "hann",
                    center: bool = True,
                    normalized: bool = False,
                    onesided: bool = True,
                    mag_spec: bool = False):
        """
        The process from waveform to linear spectrogram has 4 steps:
            1. waveform preemphasis (implemented by Conv1d layer)
            2. waveform normalization
            3. STFT

        Args:
            n_fft: int
                The number of Fourier point used for STFT
            hop_length: int or float
                the distance between neighboring sliding window frames for STFT.
                int means the absolute number of sampling point,
                float means the duration of the speech segment (in seconds).
            win_length: int or float
                the size of window frame for STFT.
                int means the absolute number of sampling point,
                float means the duration of the speech segment (in seconds).
            preemphasis: float
                The preemphasis coefficient before STFT.
            pre_stft_norm: str
                The normalization method for the speech waveforms before STFT.
            window: str
                The window type for STFT.
            center: bool
                 whether to pad input on both sides so that the t-th frame is centered at time t × hop_length.
            normalized: bool
                controls whether to return the normalized STFT results
            onesided: bool
                controls whether to return half of results to avoid redundancy for real inputs.
            mag_spec: bool
                controls whether to calculate the linear magnitude spectrogram during STFT.
                True feeds the linear magnitude spectrogram into mel-fbank.
                False feeds the linear energy spectrogram into mel-fbank.

        """

        # preemphasis filter initialization
        self.preemphasis = preemphasis
        if preemphasis is not None:
            _preemph_filter = torch.nn.Conv1d(1, 1, kernel_size=2, bias=False)
            _filter_weight = torch.Tensor(np.array([-self.preemphasis, 1])).reshape(1, 1, 2)
            _preemph_filter.weight = torch.nn.Parameter(_filter_weight, requires_grad=False)

            # move the weight from _parameters to _buffers so that these parameters won't influence the training
            _para_keys = list(_preemph_filter._parameters.keys())
            for name in _para_keys:
                _preemph_filter._buffers[name] = _preemph_filter._parameters.pop(name)
            self.preemph_filter = _preemph_filter

        # normalization style before STFT
        self.pre_stft_norm = pre_stft_norm

        # during stft
        self.stft_config = dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            normalized=normalized,
            onesided=onesided,
            return_complex=True
        )

        # True=magnitude spectrogram, False=energy spectrogram
        self.mag_spec = mag_spec


    def forward(self, speech: torch.Tensor, speech_len: torch.Tensor):
        """

        Args:
            speech: (batch, speech_maxlen, 1) or (batch, speech_maxlen)
                The input speech data.
            speech_len: (batch,)
                The lengths of input speech data

        Returns:
            The linear spectrograms (energy or magnitude) with their lengths.

        """

        # preparatory checking for the input speech data
        if speech.dim() == 2:
            speech.unsqueeze(-1)
        elif speech.dim() == 3 and speech.size(-1) != 1:
            raise RuntimeError(f"If the speech is given in 3D vectors, the last dimension must be 1. "
                               f"But got speech.shape={speech.shape}.")

        # apply preemphasis if specified
        if self.preemphasis is not None:
            _previous_speech = F.pad(speech.transpose(1, 2), (1, 0))
            speech = self.preemph_filter(_previous_speech).transpose(1, 2)
            # remove redundant preemphasis calculations (there is one meaningless point at the end of some utterances)
            for i in range(speech_len.size(0)):
                if speech_len[i] < speech.size(1):
                    speech[i][speech_len[i]] = 0.0

        # normalization for audio signals before STFT
        if self.pre_stft_norm is not None:
            if self.pre_norm == 'mean_std':
                speech = (speech - speech.mean(dim=-1)) / speech.std(dim=-1)
            elif self.pre_norm == 'norm':
                speech = (speech - speech.min(dim=-1)) / (speech.max(dim=-1) - speech.min(dim=-1)) * 2 - 1

        # initialize the window function at the first step of the training phase (borrowed from ESPNET)
        if isinstance(self.stft_config['window'], str):
            window_func = getattr(torch, f"{self.stft_config['window']}_window")
            self.stft_config['window'] = window_func(
                self.stft_config['win_length'], dtype=speech.dtype, device=speech.device
            )
        # extract linear spectrogram from signal by stft
        stft_feat = torch.stft(speech.squeeze(-1), **self.stft_config).transpose(1, 2)

        # calculate the number of frames after STFT (borrowed from ESPNET)
        if self.stft_config['center']:
            # speech_len += 2 * (self.stft_config['n_fft'] // 2)
            speech_len += 2 * torch.div(self.stft_config['n_fft'], 2, rounding_mode='floor')

        # feat_len = (speech_len - self.stft_config['n_fft']) // self.stft_config['hop_length'] + 1
        feat_len = torch.div(speech_len - self.stft_config['n_fft'], self.stft_config['hop_length'], rounding_mode='floor') + 1

        # get rid of imaginary parts
        linear_spec = stft_feat.real ** 2 + stft_feat.imag ** 2
        # prevent extremely small numbers
        linear_spec = torch.clamp(input=linear_spec, min=torch.finfo().eps)
        # convert the energy spectrogram to the magnitude spectrogram if specified
        if self.mag_spec:
            linear_spec = torch.sqrt(linear_spec)

        # make sure all the silence parts are zeros
        for i in range(feat_len.size(0)):
            if feat_len[i] < linear_spec.size(1):
                linear_spec[i][feat_len[i]:] = 0.0

        return linear_spec, feat_len