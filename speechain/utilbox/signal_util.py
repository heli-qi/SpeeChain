"""
    Author: Sashi Novitasari
    Affiliation: NAIST
    Date: 2022.04
"""
import librosa
import numpy as np
from scipy import signal
np.random.seed(123)

"""
source : https://github.com/madebyollin/acapellabot/blob/master/conversion.py
"""

def preemphasis(x, coeff_preemph):
    '''

    Args:
        x:
        coeff_preemph:

    Returns:

    '''
    return signal.lfilter([1, -coeff_preemph], [1], x)

def zero_guard(x) :
    return np.where(x < np.finfo(np.float32).eps, np.finfo(np.float32).eps, x)

def rosa_spectrogram(signal, n_fft=512, hop_length=None, win_length=None, power=2) :
    '''

    Args:
        signal:
        n_fft:
        hop_length:
        win_length:
        power:

    Returns:

    '''
    spectrogram = librosa.stft(y=signal,
                               n_fft=n_fft,
                               hop_length=hop_length,
                               win_length=win_length)
    phase = np.imag(spectrogram)
    magnitude = np.abs(spectrogram)
    sqr_magnitude = np.power(magnitude, power)
    sqr_magnitude = np.where(sqr_magnitude < np.finfo(np.float32).eps, np.finfo(np.float32).eps, sqr_magnitude)
    return sqr_magnitude.T, phase.T

def rosa_spec2mel(sqr_magnitude, sr=16000, n_mels=80, fmin=0, fmax=None, clip=1e-10):
    '''

    Args:
        sqr_magnitude:
        sr:
        n_mels:
        fmin:
        fmax:
        clip:

    Returns:

    '''
    mel_spec = librosa.feature.melspectrogram(S=sqr_magnitude.T,
                                              sr=sr,
                                              n_mels=n_mels,
                                              fmin=fmin,
                                              fmax=fmax).T
    return np.maximum(mel_spec, float(clip))

def rosa_inv_spectrogram(spectrogram, n_fft=512, hop_length=None, win_length=None, power=2, phase_iter=50):
    '''

    Args:
        spectrogram:
        n_fft:
        hop_length:
        win_length:
        power:
        phase_iter:

    Returns:

    '''
    spectrogram = spectrogram.T
    magnitude = np.power(spectrogram, 1/power)
    magnitude = np.power(magnitude, 1.5) # TODO : tacotron tricks #
    for ii in range(phase_iter) :
        if ii == 0 :
            recons = np.pi * np.random.random_sample(magnitude.shape) + 1j * (2 * np.pi * np.random.random_sample(magnitude.shape) - np.pi)
        else :
            recons = librosa.stft(signal, n_fft, hop_length, win_length)
        spectrum = magnitude * np.exp(1j * np.angle(recons))
        signal = librosa.istft(spectrum, hop_length, win_length)
    return signal


def inv_preemphasis(x, coeff_preemph):
    '''

    Args:
        x:
        coeff_preemph:

    Returns:

    '''
    return signal.lfilter([1], [1, -coeff_preemph], x)
