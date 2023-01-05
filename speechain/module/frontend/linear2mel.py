"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import math
import torch
import torchaudio
from speechain.module.abs import Module


class LinearSpec2MelSpec(Module):
    """
    The input is linear spectrogram extracted by STFT and the output is (log-)mel spectrogram
    The mel-fbank is implemented by a linear layer without the bias vector.

    """

    def module_init(self,
                    n_fft: int,
                    n_mels: int,
                    sr: int = 16000,
                    fmin: float = 0.0,
                    fmax: float = None,
                    clamp: float = 1e-10,
                    logging: bool = True,
                    log_base: float = 10.0,
                    mel_scale: str = 'slaney',
                    mel_norm: bool = True,
                    mag_spec: bool = False):
        """

        The difference between two different options of mel_scale, i.e., 'htk' and 'slaney', is the relationship between
        the linear frequency (Hz) and mel frequency.
            1. For 'htk', the mel frequency is always logarithmic to the linear frequency by the following formula:
                mel = 2595.0 * np.log10(1.0 + hz / 700.0)
            2. For 'slaney', the mel frequency is linear to the linear frequency below 1K Hz and logarithmic above 1K Hz
        In the initalization function, the default configuration is mel_scale = 'slaney' and mel_norm=True (the filters
        will be normalized by the filter width).

        A simple calculation procedure of 'htk'-scaled mel-fbank is shown below. For details about mel-scales, please
        refer to http://librosa.org/doc/latest/generated/librosa.mel_frequencies.html?highlight=mel_frequencies#librosa.mel_frequencies
            >>> def hz2mel(hz: float or torch.Tensor):
            ...     return 2595 * math.log10(1 + hz / 700)
            >>> def mel2hz(mel: float or torch.Tensor):
            ...     return 700 * (10 ** (mel / 2595) - 1) \

            >>> # --- Initialization for Mel-Fbank Matrix Production --- #
            ... # frequency axis of the linear spectrogram
            ... src_hz_points = torch.linspace(0, self.sr // 2, self.stft_dim).repeat(self.n_mels, 1)
            ... # mel-frequency axis of the mel spectrogram, [mel(0), ..., mel(stft_dim + 1)]
            ... # Note: there are two auxiliary points mel(0) and mel(stft_dim + 1)
            ... mel_ranges = torch.linspace(hz2mel(self.fmin), hz2mel(self.fmax), n_mels + 2)
            ... # frequency axis of the mel spectrogram
            ... hz_ranges = mel2hz(mel_ranges)

            >>> # --- Left Slope Calculation --- #
            ... # left mel-band width, [mel(1) - mel(0), ..., mel(stft_dim) - mel(stft_dim - 1)]
            ... mel_left_hz_bands = (hz_ranges[1:] - hz_ranges[:-1])[:-1].repeat(self.stft_dim, 1).transpose(0, 1)
            ... # left-shifted mel-frequency, [mel(0), ..., mel(stft_dim - 1)]
            ... mel_left_hz_points = hz_ranges[: -2].repeat(self.stft_dim, 1).transpose(0, 1)
            ... # slope values of the left mel-band
            ... # i.e. (hz - mel(m - 1)) / (mel(m) - mel(m - 1)) where m in [1, ..., stft_dim]
            ... left_slopes = (src_hz_points - mel_left_hz_points) / mel_left_hz_bands
            ... # slope masks of the left mel-band
            ... # True for the frequency in [mel(m - 1), mel(m)] where m in [1, ..., stft_dim]
            ... left_masks = torch.logical_and(left_slopes >= 0, left_slopes <= 1)

            >>> # --- Right Slope Calculation --- #
            ... # right mel-band width, [mel(2) - mel(1), ..., mel(stft_dim + 1) - mel(stft_dim)]
            ... mel_right_hz_bands = (hz_ranges[1:] - hz_ranges[:-1])[1:].repeat(self.stft_dim, 1).transpose(0, 1)
            ... # right-shifted mel-frequency, [mel(2), ..., mel(stft_dim + 1)]
            ... mel_right_hz_points = hz_ranges[2:].repeat(self.stft_dim, 1).transpose(0, 1)
            ... # slope values of the right mel-band
            ... # i.e. (mel(m + 1) - hz) / (mel(m + 1) - mel(m)) where m in [1, ..., stft_dim]
            ... right_slopes = (mel_right_hz_points - src_hz_points) / mel_right_hz_bands
            ... # slope masks of the right mel-band
            ... # True for the frequency in [mel(m), mel(m + 1)] where m in [1, ..., stft_dim]
            ... right_masks = torch.logical_and(right_slopes >= 0, right_slopes < 1)

            >>> # --- Mel-Fbank Matrix Generation --- #
            ... mel_matrix = torch.zeros(self.n_mels, self.stft_dim)
            ... mel_matrix[left_masks] = left_slopes[left_masks]
            ... mel_matrix[right_masks] = right_slopes[right_masks]

        Args:
            sr: int
                The sampling rate of the input speech waveforms.
            n_fft: int
                The number of Fourier point used for STFT
            n_mels: int
                The number of filters in the mel-fbank
            fmin: float
                The minimal frequency for the mel-fbank
            fmax: float
                The maximal frequency for the mel-fbank
            clamp: float
                The minimal number for the log-mel spectrogram. Used for numerical stability.
            logging: bool
                Controls whether to take log for the mel spectrogram.
            log_base: float
                The log base for the log-mel spectrogram. None means the natural log base e.
                This argument is effective when mel_norm=True (ESPNET style)
            mel_scale: str
                The tyle of mel-scale of the mel-fbank. 'htk' for SpeechBrain style and 'slaney' for ESPNET style.
            mel_norm: bool
                Whether perform the area normalization to the mel-fbank filters.
                True for ESPNET style and False for SpeechBrain style.
            mag_spec: bool
                Whether the input linear spectrogram is the magnitude. Used for decibel calculation.
                This argument is effective when mel_norm=False (SpeechBrain style)

        """
        # fundamental arguments
        self.sr = sr
        self.stft_dim = n_fft // 2 + 1
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sr // 2
        assert self.fmax > self.fmin, \
            f"fmax must be larger than fmin, but got fmax={self.fmax} and fmin={self.fmin}!"

        # mel-scale-related arguments
        assert mel_scale in ['htk', 'slaney'], \
            f"mel_scale must be either 'htk' or 'slaney', but got mel_scale={mel_scale}"
        self.mel_scale = mel_scale
        self.mel_norm = mel_norm

        # mel-fbank generation
        mel_matrix = torchaudio.functional.melscale_fbanks(sample_rate=self.sr,
                                                           n_mels=self.n_mels, n_freqs=self.stft_dim,
                                                           f_min=self.fmin, f_max=self.fmax,
                                                           norm='slaney' if self.mel_norm else None,
                                                           mel_scale=self.mel_scale).T

        # implement mel-fbank extraction by a linear layer
        mel_fbanks = torch.nn.Linear(in_features=self.stft_dim, out_features=self.n_mels, bias=False)
        mel_fbanks.weight = torch.nn.Parameter(mel_matrix, requires_grad=False)

        # move the weight from _parameters to _buffers so that these parameters won't influence the training
        _para_keys = list(mel_fbanks._parameters.keys())
        for name in _para_keys:
            mel_fbanks._buffers[name] = mel_fbanks._parameters.pop(name)
        self.mel_fbanks = mel_fbanks

        # logging-related arguments
        self.clamp = clamp
        self.logging = logging
        self.log_base = log_base
        self.mag_spec = mag_spec

    def forward(self, feat: torch.Tensor, feat_len: torch.Tensor):
        """

        Args:
            feat: (batch, speech_maxlen, stft_dim)
                The input linear spectrograms
            feat_len: (batch,)
                The lengths of the input linear spectrograms

        Returns:
            The log-mel spectrograms with their lengths.

        """
        # extract mel-scale spectrogram
        feat = self.mel_fbanks(feat)

        # take the logarithm operation
        if self.logging:
            # pre-log clamping for numerical stability
            feat = torch.clamp(input=feat, min=self.clamp)

            # go through the logarithm operation for the mel-fbanks
            feat = feat.log()
            if self.log_base is not None:
                feat /= math.log(self.log_base)

        return feat, feat_len

    def recover(self, feat: torch.Tensor, feat_len: torch.Tensor):
        """

        Args:
            feat: (batch_size, feat_maxlen, mel_dim)
            feat_len: (batch_size,)

        Returns:

        """
        # recover the logarithm operation
        if self.logging:
            feat = torch.pow(
                torch.full_like(feat, fill_value=torch.e if self.log_base is None else self.log_base), feat
            )

        # recover the mel spectrograms back to linear spectrograms
        feat = torch.linalg.lstsq(
            self.mel_fbanks.weight.data.unsqueeze(0), feat.transpose(-2, -1),
            # the default driver for CPU data is 'gelsy' which doesn't work and the result is zero
            driver='gels'
        ).solution.transpose(-2, -1)

        # turn the silence part of the shorter utterances to zeros
        for i in range(len(feat_len)):
            if feat_len[i] != feat_len.max():
                feat[i][feat_len[i]:] = 0

        # clamp the spectrogram for numerical stability
        return torch.clamp(feat, min=1e-10)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n" \
               f"stft_dim={self.stft_dim}, " \
               f"n_mels={self.n_mels}, " \
               f"fmin={self.fmin}, " \
               f"fmax={self.fmax}, " \
               f"mel_scale={self.mel_scale}, " \
               f"mel_norm={self.mel_norm}" \
               f"\n)"
