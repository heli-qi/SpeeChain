"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import torch
import torchaudio
from speechain.module.abs import Module


class LinearSpec2MelSpec(Module):
    """
    The acoustic feature fronted where the input is linear spectrogram and the output is log-mel spectrogram
    This process is implemented by a linear layer without bias vector.

    """
    def module_init(self,
                    sr: int,
                    n_fft: int,
                    n_mels: int,
                    fmin: float = 0.0,
                    fmax: float = None,
                    clip: float = 1e-10,
                    log_base: float = 10.0,
                    mel_scale: str = 'slaney'):
        """
        The mel-fbank is implemented as a linear layer without the bias vector.

        There are two ways to obtain the weights matrix of the mel-scale filters:
            1. torch.Tensor(librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax))
            2. torchaudio.functional.melscale_fbanks(sample_rate=sr, n_mels=n_mels, n_freqs=n_fft // 2 + 1,
                                                     f_min=fmin, f_max=fmax)
        The second one directly gives torch.Tensor, so it is chosen in this function.

        Note: librosa.filters.mel() and torchaudio.functional.melscale_fbanks() have different default values for
        'htk' and 'norm'. librosa.filters.mel() uses 'slaney' norm without htk while torchaudio.functional.melscale_fbanks()
        applies htk without norm.

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
            clip: float
                The minimal number for the log-mel spectrogram. Used for stability.
            log_base: float
                The log base for the log-mel spectrogram. None means the natural log base e.
            mel_scale: str

        """

        # log-mel fbank initialization
        assert mel_scale in ['htk', 'slaney'], \
            f"mel_scale must be either 'htk' or 'slaney', but got mel_scale={mel_scale}"

        _norm = 'slaney' if mel_scale == 'slaney' else None
        _mel_matrix = torchaudio.functional.melscale_fbanks(sample_rate=sr, n_mels=n_mels, n_freqs=n_fft // 2 + 1,
                                                            f_min=fmin, f_max=fmax, norm=_norm, mel_scale=mel_scale).T
        mel_dim, linear_dim = _mel_matrix.size(0), _mel_matrix.size(1)

        _mel_fbanks = torch.nn.Linear(in_features=linear_dim, out_features=mel_dim, bias=False)
        _mel_fbanks.weight = torch.nn.Parameter(_mel_matrix, requires_grad=False)

        # move the weight from _parameters to _buffers so that these parameters won't influence the training
        _para_keys = list(_mel_fbanks._parameters.keys())
        for name in _para_keys:
            _mel_fbanks._buffers[name] = _mel_fbanks._parameters.pop(name)
        self.mel_fbanks = _mel_fbanks

        # clipping and logarithm coefficient
        self.clip = clip
        self.log_base = log_base


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

        # extract mel-scale spectrogram and do the clipping
        feat = self.mel_fbanks(feat)
        feat = torch.clamp(input=feat, min=self.clip)

        # take the logarithm operation
        feat = feat.log()
        if self.log_base is not None:
            feat /= torch.log(self.log_base)

        return feat, feat_len