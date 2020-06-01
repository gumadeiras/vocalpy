# -*- coding: utf-8 -*-
"""VocalPy - Vocal analysis framework"""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import numpy as np
from scipy.signal import spectrogram, get_window, butter, lfilter


def compute_spectrogram(samples, fs, window_type, window_size, noverlap, nfft, low_frequency_cutoff, high_frequency_cutoff):
    """
    Computes the spectrogram, applies a frequency cutoff, and converts power values to decibel

    Parameters
    ----------
    samples : numpy.array
        audio samples time series
    fs : int
        sampling frequency of the audio
    window_type : str
        windowing function
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html#scipy.signal.get_window
    window_size : int
        window size
    noverlap : int
        number of points to overlap between segments
    nfft : int
        number of points to compute the stft
    low_frequency_cutoff : int
        frequencies lower than this limit will be removed
    high_frequency_cutoff : int
        frequencies higher than this limit will be removed

    Returns
    -------
    (f, t, Pxx) : numpy.array, numpy.array, numpy.array
        f contains the frequency bins
        t contains the time bins
        Pxx contains the power values for each bin in decibel
    """
    # -- compute spectrogram
    f, t, Pxx = spectrogram(
        x=samples, fs=fs, window=get_window(window_type, window_size), noverlap=noverlap, nfft=nfft, mode="psd"
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


def butter_bandpass(low_frequency_cutoff, high_frequency_cutoff, fs, order=25):
    nyquist_freq = 0.5 * fs
    low = low_frequency_cutoff / nyquist_freq
    high = high_frequency_cutoff / nyquist_freq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(samples, low_frequency_cutoff, high_frequency_cutoff, fs, order=25):
    b, a = butter_bandpass(low_frequency_cutoff, high_frequency_cutoff, fs, order=order)
    y = lfilter(b, a, samples)
    return y


def butter_highpass(high_frequency_cutoff, fs, order=25):
    nyquist_freq = 0.5 * fs
    high = high_frequency_cutoff / nyquist_freq
    b, a = butter(order, high, btype="high")
    return b, a


def butter_highpass_filter(samples, high_frequency_cutoff, fs, order=25):
    b, a = butter_highpass(high_frequency_cutoff, fs, order=order)
    y = lfilter(b, a, samples)
    return y
