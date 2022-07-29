"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import torch
from speechain.module.abs import Module
from speechain.module.frontend.speech2linear import Speech2LinearSpec
from speechain.module.frontend.linear2mel import LinearSpec2MelSpec
from speechain.module.frontend.delta_feat import DeltaFeature

class Speech2MelSpec(Module):
    """
    The acoustic frontend where the input is raw speech waveforms and the output is log-mel spectrogram.

    The waveform is first converted into linear spectrogram by STFT. Then, the linear spectrogram is converted into
    log-mel spectrogram by mel-fbank. Finally, the delta features of log-mel spectrogram are calculated if specified.

    """
    def module_init(self,
                    n_mels: int,
                    n_fft: int,
                    hop_length: int or float,
                    win_length: int or float,
                    sr: int = 16000,
                    preemphasis: float = None,
                    pre_stft_norm: str = None,
                    window: str = "hann",
                    center: bool = True,
                    normalized: bool = False,
                    onesided: bool = True,
                    mag_spec: bool = False,
                    fmin: float = 0.0,
                    fmax: float = None,
                    clip: float = 1e-10,
                    log_base: float = None,
                    delta_order: int = None,
                    delta_N: int = 2):
        """

        Args:
            n_mels: int
                The number of filters in the mel-fbank
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
            sr: int
                The sampling rate of the input speech waveforms.
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
            fmin: float
                The minimal frequency for the mel-fbank
            fmax: float
                The maximal frequency for the mel-fbank
            clip: float
                The minimal number for the log-mel spectrogram. Used for stability.
            log_base: float
                The log base for the log-mel spectrogram. None means the natural log base e.
            delta_order: int
                The delta order you want to add to the original log-mel spectrogram.
                1 means original log-mel spectrogram + Δ Log-mel spectrogram
                2 means original log-mel spectrogram + Δ Log-mel spectrogram + ΔΔ log-mel spectrogram
            delta_N: int
                The number of neighbouring points used for calculating the delta features.

        """
        # para recording, used for returning output size
        self.output_size = n_mels

        # if hop_length and win_length are given in the unit of seconds, turn them into the corresponding time steps
        hop_length = int(hop_length * sr) if isinstance(hop_length, float) else hop_length
        win_length = int(win_length * sr) if isinstance(win_length, float) else win_length

        # Speech -> Linear Spectrogram
        self.speech2linear = Speech2LinearSpec(n_fft=n_fft,
                                                      hop_length=hop_length,
                                                      win_length=win_length,
                                                      preemphasis=preemphasis,
                                                      pre_stft_norm=pre_stft_norm,
                                                      window=window,
                                                      center=center,
                                                      normalized=normalized,
                                                      onesided=onesided,
                                                      mag_spec=mag_spec)
        # Linear Spectrogram -> Log-Mel Spectrogram
        self.linear2mel = LinearSpec2MelSpec(sr=sr,
                                                    n_fft=n_fft,
                                                    n_mels=n_mels,
                                                    fmin=fmin,
                                                    fmax=fmax,
                                                    clip=clip,
                                                    log_base=log_base)
        # (Optional) Log-Mel Spectrogram -> Log-Mel Spectrogram + Deltas
        self.delta_order = delta_order
        if delta_order is not None:
            self.delta = DeltaFeature(delta_order=delta_order, delta_N=delta_N)


    def forward(self, speech: torch.Tensor, speech_len: torch.Tensor):
        """

        Args:
            speech: (batch, speech_maxlen, 1) or (batch, speech_maxlen)
                The input speech data.
            speech_len: (batch,)
                The lengths of input speech data

        Returns:
            The log-mel spectrograms with their lengths.

        """

        # Speech -> Linear Spectrogram
        feat, feat_len = self.speech2linear(speech, speech_len)

        # Linear Spectrogram -> Log-Mel Spectrogram
        feat, feat_len = self.linear2mel(feat, feat_len)

        # Log-Mel Spectrogram -> Log-Mel Spectrogram + Deltas
        if self.delta_order is not None:
            feat, feat_len = self.delta(feat, feat_len)

        return feat, feat_len