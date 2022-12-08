"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import math
import torch
import torchaudio
import torch.nn.functional as F
from speechain.module.abs import Module


class Speech2LinearSpec(Module):
    """
    The acoustic frontend where the input is raw speech waveforms and the output is linear spectrogram.

    """

    def module_init(self,
                    hop_length: int,
                    win_length: int,
                    sr: int = 16000,
                    n_fft: int = None,
                    preemphasis: float = None,
                    pre_stft_norm: str = None,
                    window: str = "hann",
                    center: bool = True,
                    normalized: bool = False,
                    onesided: bool = True,
                    mag_spec: bool = False,
                    clamp: float = 1e-10,
                    logging: bool = False,
                    log_base: float = None):
        """
        The transformation from waveform to linear spectrogram has 4 steps:
            1. (optional) waveform pre-emphasis (implemented by Conv1d layer)
            2. (optional) waveform pre-normalization (not recommended for TTS model)
            3. STFT processing (implemented by torch.stft())
            4. STFT postprocessing: zero masking, (optional)sqrt for magnitude, (optional)clamping + logging

        Args:
            hop_length: int or float
                the distance between neighboring sliding window frames for STFT.
                int means the absolute number of sampling point,
                float means the duration of the speech segment (in seconds).
            win_length: int or float
                the size of window frame for STFT.
                int means the absolute number of sampling point,
                float means the duration of the speech segment (in seconds).
            sr: int
                The sampling rate of the input speech waveforms. Only used for window calculation.
            n_fft: int
                The number of Fourier point used for STFT
            preemphasis: float
                The preemphasis coefficient before STFT.
            pre_stft_norm: str
                The normalization type for the speech waveforms before STFT.
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
            clamp: float
                The minimal number for the log-linear spectrogram. Used for numerical stability.
            logging: bool
                Controls whether to take log for the mel spectrogram.
            log_base: float
                The log base for the log-mel spectrogram. None means the natural log base e.

        """
        # if hop_length and win_length are given in the unit of seconds, turn them into the corresponding time steps
        hop_length = int(hop_length * sr) if isinstance(hop_length, float) else hop_length
        win_length = int(win_length * sr) if isinstance(win_length, float) else win_length

        # if n_fft is not given, it will be initialized to the window length
        if n_fft is None:
            n_fft = win_length

        # para recording
        self.output_size = n_fft // 2 + 1 if onesided else n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft

        # preemphasis filter initialization
        self.preemphasis = preemphasis
        if preemphasis is not None:
            _preemph_filter = torch.nn.Conv1d(1, 1, kernel_size=2, bias=False)
            _filter_weight = torch.Tensor([-self.preemphasis, 1]).reshape(1, 1, 2)
            _preemph_filter.weight = torch.nn.Parameter(_filter_weight, requires_grad=False)

            # move the weight from _parameters to _buffers so that these parameters won't influence the training
            _para_keys = list(_preemph_filter._parameters.keys())
            for name in _para_keys:
                _preemph_filter._buffers[name] = _preemph_filter._parameters.pop(name)
            self.preemph_filter = _preemph_filter

        # normalization type before STFT
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

        # logging-related arguments
        self.clamp = clamp
        self.logging = logging
        self.log_base = log_base

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
            raise RuntimeError(f"Currently, we don't support multi-channel speech waveforms. "
                               f"If the speech is given in 3D vectors, the last dimension must be 1. "
                               f"But got speech.shape={speech.shape}.")

        # --- Waveform Pre-Emphasis --- #
        # apply preemphasis if specified
        if self.preemphasis is not None:
            _previous_speech = F.pad(speech.transpose(1, 2), (1, 0))
            speech = self.preemph_filter(_previous_speech).transpose(1, 2)
            # remove redundant preemphasis calculations (there is one meaningless point at the end of some utterances)
            for i in range(speech_len.size(0)):
                if speech_len[i] < speech.size(1):
                    speech[i][speech_len[i]:] = 0.0

        # --- Waveform Pre-Normalization --- #
        # normalization for audio signals before STFT
        if self.pre_stft_norm is not None:
            if self.pre_stft_norm == 'mean_std':
                speech = (speech - speech.mean(dim=1)) / speech.std(dim=1)
            elif self.pre_stft_norm == 'min_max':
                speech_min, speech_max = speech.min(dim=1, keepdim=True)[0], speech.max(dim=1, keepdim=True)[0]
                speech = (speech - speech_min) / (speech_max - speech_min) * 2 - 1
            else:
                raise ValueError

        # --- STFT Processing --- #
        # initialize the window function lazily at the first training step
        # borrowed from https://github.com/espnet/espnet/blob/80e042099655822d6543c256910ae655a1a056fd/espnet2/layers/stft.py#L83
        if isinstance(self.stft_config['window'], str):
            window_func = getattr(torch, f"{self.stft_config['window']}_window")
            self.stft_config['window'] = window_func(
                self.stft_config['win_length'], dtype=speech.dtype, device=speech.device
            )
        # extract linear spectrogram from signal by stft
        stft_feat = torch.stft(speech.squeeze(-1), **self.stft_config).transpose(1, 2)

        # calculate the number of frames after STFT
        if self.stft_config['center']:
            speech_len += 2 * torch.div(self.stft_config['n_fft'], 2, rounding_mode='floor')
        feat_len = torch.div(speech_len - self.stft_config['n_fft'], self.stft_config['hop_length'],
                             rounding_mode='floor') + 1
        # get the energy spectrogram
        linear_spec = stft_feat.real ** 2 + stft_feat.imag ** 2

        # --- STFT Post-Processing --- #
        # mask all the silence parts of the linear spectrograms to zeros
        for i in range(feat_len.size(0)):
            if feat_len[i] < linear_spec.size(1):
                linear_spec[i][feat_len[i]:] = 0.0

        # convert the energy spectrogram to the magnitude spectrogram if specified
        if self.mag_spec:
            linear_spec = torch.sqrt(linear_spec)

        # take the logarithm operation
        if self.logging:
            # pre-log clamping for numerical stability
            linear_spec = torch.clamp(input=linear_spec, min=self.clamp)
            linear_spec = linear_spec.log()
            if self.log_base is not None:
                linear_spec /= math.log(self.log_base)

        return linear_spec, feat_len

    def recover(self, feat: torch.Tensor, feat_len: torch.Tensor, inv_preemph_winlen: int = 100):
        """

        Args:
            feat:
            feat_len:
            inv_preemph_winlen:

        Returns:

        """
        # --- STFT Recovery by the GL Algorithm --- #
        # 1. Randomly initialize the phase between 0 and 2Π
        # 2. Recover the waveform by the magnitude and phase
        # 3. Process the synthetic waveform and get the new magnitude and phase
        # 4. Iteratively do step2 and step3 by the original magnitude and new phase
        #
        # Pseudo codes of GL algorithm could be as follow:
        #     angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        #     S_complex = np.abs(S).astype(np.complex)
        #     y = _istft(S_complex * angles, hparams)
        #     for i in range(hparams.griffin_lim_iters):
        #         angles = np.exp(1j * np.angle(_stft(y, hparams)))
        #         y = _istft(S_complex * angles, hparams)
        # ----------------------------------------- #
        if not hasattr(self, 'griffin_lim'):
            # lazily initialize the linear-to-waveform transformation
            self.griffin_lim = torchaudio.transforms.GriffinLim(
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                window_fn=getattr(torch, f"{self.stft_config['window']}_window"),
                power=1 if self.mag_spec else 2
            )
            self.griffin_lim.window = self.griffin_lim.window.cuda(feat.device)
        wav = self.griffin_lim(feat.transpose(-2, -1))
        wav_len = (feat_len - 1) * self.hop_length
        assert wav_len.max() == wav.size(1), \
            "Something wrong happens when calculating the length of synthetic utterances."

        # pre-stft normalization cannot be recovered
        assert self.pre_stft_norm is None, "waveform pre-stft normalization cannot be recovered for TTS synthesis."

        # --- Pre-Emphasis Recovery --- #
        # Pre-emphasis: Y[n] = X[n] - 0.97 * X[n-1] where Y is the pre-emphasized signal and X is the original signal
        # Inverse Pre-emphasis: X[n] = Y[n] + 0.97 * X[n-1] = Y[n] + 0.97 * Y[n-1] + ... + (0.97)^n * Y[0]
        # However, since the signal is usually very long (n is in the unit of 10k), the power of n will infinitely
        # approach 0 as n grows. So, a slide window is used to perform inverse pre-emphasis which only considers the
        # previous time steps in a given range.
        # ----------------------------- #
        if self.preemphasis is not None:
            # lazily initialize the inverse preemphasis filters (implemented by Conv1d)
            if not hasattr(self, 'inv_preemph'):
                inv_preemph_filter = torch.nn.Conv1d(1, 1, kernel_size=inv_preemph_winlen, bias=False)

                # get the sliding window for the inverse pre-emphasis
                inv_preemph_win = torch.pow(
                    torch.full((inv_preemph_winlen,), fill_value=self.preemphasis),
                    torch.arange(start=inv_preemph_winlen - 1, end=-1, step=-1, dtype=torch.int),
                ).reshape(1, 1, -1).to(wav.device)
                inv_preemph_filter.weight = torch.nn.Parameter(inv_preemph_win, requires_grad=False)

                # move the weight from _parameters to _buffers so that these parameters won't influence the training
                _para_keys = list(inv_preemph_filter._parameters.keys())
                for name in _para_keys:
                    inv_preemph_filter._buffers[name] = inv_preemph_filter._parameters.pop(name)
                self.inv_preemph_filter = inv_preemph_filter

            wav = F.pad(wav.unsqueeze(1), (inv_preemph_winlen - 1, 0))
            wav = self.inv_preemph_filter(wav).transpose(-2, -1)

        # make sure that the redundant parts are set to silence
        for i in range(len(wav_len)):
            wav[i][wav_len[i]:] = 0

        return wav, wav_len

    def __repr__(self) -> str:
        string = f"{self.__class__.__name__}(\n" \
                 f"win_length={self.win_length}, " \
                 f"hop_length={self.hop_length}, " \
                 f"n_fft={self.n_fft}, " \
                 f"mag_spec={self.mag_spec}, "

        if self.preemphasis is not None:
            string += f"preemphasis={self.preemphasis}, "
        if self.pre_stft_norm is not None:
            string += f"pre_stft_norm={self.pre_stft_norm}, "

        return string + '\n)'
