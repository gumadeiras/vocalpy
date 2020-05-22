# -*- coding: utf-8 -*-
"""VocalPy - Vocal analysis framework"""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import numpy as np
from scipy import signal


def spectrogram(samples, fs, window_type, window_size, noverlap, nfft, low_frequency_cutoff, high_frequency_cutoff):
    """
    Computes the spectrogram, applies a frequency cutoff, and converts power values to decibel

    Parameters
    ----------
    samples
    fs : int
        audio sampling rate
    window_type : str
        window type (check available ones from scipy.sginal.get_window())
    window_size : int
        window size
    noverlap : int
        overlap between sliding windows
    nfft : int
        number of points to compute the stft
    low_frequency_cutoff : int
        frequencies lower than this limit will be removed
    high_frequency_cutoff : int
        frequencies higher than this limit will be removed
    """
    # -- compute spectrogram
    f, t, Pxx = signal.spectrogram(
        samples, fs=fs, window=signal.get_window(window_type, window_size), noverlap=noverlap, nfft=nfft, mode="psd"
    )

    # -- apply frequency cutoffs
    if low_frequency_cutoff > 0:
        Pxx = Pxx[(f > low_frequency_cutoff)]
        f = f[(f > low_frequency_cutoff)]
    if high_frequency_cutoff > 0:
        Pxx = Pxx[(f < high_frequency_cutoff)]
        f = f[(f < high_frequency_cutoff)]

    # -- convert to dB
    Pxx = 10 * np.log10(Pxx)

    return f, t, Pxx
