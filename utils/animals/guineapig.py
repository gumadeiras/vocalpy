# -*- coding: utf-8 -*-
'''VocalPy - A python version based on VocalMat'''

__email__ = 'gustavo.santana@yale.edu'
__license__ = 'Apache License, Version 2.0'
__copyright__ = '2020 Dietrich Lab - Yale University School of Medicine'

import cv2
import logging
import warnings

import numpy as np

from time import time
from math import ceil
from scipy import signal, ndimage
from skimage import exposure, measure

from utils.processing import bradley_roth


def identifier(chunk):
    from classes.vocal import Vocal
    from classes.list_of_vocals import ListOfVocals
    timeBinA = time()

    # -- unwrap chunk
    output_dir, spectrogram_dir, mask_dir, sample_rate, sample_range, this_bin, start_range, end_range, bin_size, low_frequency_cutoff, high_frequency_cutoff, args = chunk

    if args.verbose:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s [%(levelname)-5.5s]  %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S',
                            handlers=[
                                logging.FileHandler(
                                    '{0}/{1}.log'.format(output_dir, 'output')),
                                logging.StreamHandler()
                            ])
    else:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s [%(levelname)-5.5s]  %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S',
                            handlers=[
                                logging.FileHandler(
                                    '{0}/{1}.log'.format(output_dir, 'output')),
                            ])

    logger = logging.getLogger()

    timeASpectrogram = time()
    fs = sample_rate
    window = signal.get_window('barthann', 1024)
    noverlap = 512
    nfft = 2048
    sample_range_secs = sample_range.shape[0] / sample_rate
    logger.info('[bin {}]: computing spectrogram for bin: {}; time range: {:.2f}s; audio range: {:.2f}-{:.2f}s'.format(this_bin, this_bin,
                                                                                                                       sample_range_secs,
                                                                                                                       start_range / sample_rate,
                                                                                                                       end_range / sample_rate))
    # -- compute spectrogram
    f, t, Pxx = signal.spectrogram(sample_range, fs=fs,
                                   window=window,
                                   noverlap=noverlap,
                                   nfft=nfft,
                                   mode='psd')

    # -- apply frequency cutoffs
    if low_frequency_cutoff > 0:
        Pxx = Pxx[(f > low_frequency_cutoff)]
        f = f[(f > low_frequency_cutoff)]
    if high_frequency_cutoff > 0:
        Pxx = Pxx[(f < high_frequency_cutoff)]
        f = f[(f < high_frequency_cutoff)]

    time_res = sample_range_secs / t.shape[0]
    freq_res = (np.max(f) - low_frequency_cutoff) / f.shape[0]

    logger.info('[bin {}]: spectrogram runtime: {:.2f}s'.format(this_bin, time() - timeASpectrogram))
    logger.info('[bin {}]: time resolution: {:.2f}ms'.format(this_bin, time_res * 1000))
    logger.info('[bin {}]: freq resolution: {:.2f}Hz'.format(this_bin, freq_res))

    # -- convert to dB
    Pxx = 10 * np.log10(Pxx)

    # -- normalize data
    B = np.abs(Pxx)
    B = B / np.max(B)

    # -- contrast adjustment
    p1, p99 = np.percentile(B, (1, 99))
    B[B < p1] = 0
    B[B > p99] = 1

    # -- binarize spectrogram
    B = bradley_roth(B, t=50)

    # -- median filter
    B = ndimage.median_filter(B, size=(4, 4))

    timeAConnectedComponents = time()
    connectivity = 4
    num_cc, output, stats, centroids = cv2.connectedComponentsWithStats(B, connectivity, cv2.CV_32S)
    # del erode14

    # -- remove background stats
    num_cc = num_cc - 1
    areas = stats[1:, 4]

    # -- filtered connected components placeholder
    grain = np.zeros((output.shape))

    # -- threshold connected components by minimum area
    min_area = 40
    for i in range(0, num_cc):
        if areas[i] >= min_area:
            grain[output == i + 1] = 255

    logger.info('[bin {}]: connected components runtime: {:.2f}s'.format(this_bin, time() - timeAConnectedComponents))

    # -- one more opening to make sure
    # -- segmentation covers *at least* the real area
    grain = grain.astype(np.uint8)
    kernel_cross = np.array([[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]], dtype=np.uint8)
    kernel_line3 = np.ones((1, 3), dtype=np.uint8)
    grain = cv2.dilate(grain, kernel_line3, iterations=1)

    # -- get connected components stats
    timeARegionProps = time()
    labels = measure.label(grain, background=0)

    props = measure.regionprops(labels,
                                intensity_image=Pxx,
                                cache=True,
                                coordinates='rc')
    props = sorted(props, key=lambda p: np.min(p.coords[:, 1]), reverse=False) # sort segments by time

    logger.info('[bin {}]: region props runtime: {:.2f}s'.format(this_bin, time() - timeARegionProps))
    del labels

    timeAVocal = time()
    vocal_id = 0
    vocal_list = []

    for prop in props:
        start = np.min(prop.coords[:, 1])
        try:
            interval = np.abs(end - start)
        except:
            interval = 0

        end = np.max(prop.coords[:, 1])
        duration = end - start

        if duration < 5:
            continue

        # -- get spectrogram and mask around each vocalization to compute intensity
        spectro_range = 25  # 2*25 =~ 250ms
        centroid_time = ceil(prop.centroid[1])

        # -- edge conditions, spectro_range goes over the spectrom vector limit (for this bin)
        # -- left edge: -200 is before vector start index
        # -- right edge: +200 is after vector end index
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                bg_intensity = np.mean(Pxx[:, centroid_time - spectro_range:centroid_time + spectro_range])
            except RuntimeWarning:
                left_end_idx = centroid_time - spectro_range
                if left_end_idx < 0:
                    centroid_time = centroid_time + np.abs(left_end_idx)
                    bg_intensity = np.mean(Pxx[:, centroid_time - spectro_range:centroid_time + spectro_range])
                else:
                    centroid_time = centroid_time - np.abs(left_end_idx)
                    bg_intensity = np.mean(Pxx[:, centroid_time - spectro_range:centroid_time + spectro_range])
            warnings.simplefilter('ignore')
            warnings.filterwarnings('ignore')

        # -- if contrast ratio is above treshold
        # -- then it's a false positive
        if (prop.mean_intensity / bg_intensity) > 0.91:
            continue

        if this_bin == 1:
            # first 0.5 were removed from recording as they are noisy
            # make this better
            start_time = (start * time_res) + ((this_bin - 1) * bin_size) + 0.5
            end_time = (end * time_res) + ((this_bin - 1) * bin_size) + 0.5
        else:
            start_time = (start * time_res) + ((this_bin - 1) * bin_size)
            end_time = (end * time_res) + ((this_bin - 1) * bin_size)

        min_freq_coord = np.min(prop.coords[:, 0])
        max_freq_coord = np.max(prop.coords[:, 0])
        min_freq = (min_freq_coord * freq_res) + low_frequency_cutoff
        max_freq = (max_freq_coord * freq_res) + low_frequency_cutoff
        avg_freq = (np.mean(prop.coords[:, 0]) * freq_res) + low_frequency_cutoff
        bandwidth = max_freq - min_freq

        new_vocal = Vocal(bin_number=this_bin,
                          start=start_time,
                          end=end_time,
                          start_coord=start,
                          end_coord=end,
                          duration=duration * time_res * 1000,
                          interval=interval * time_res,
                          min_freq=min_freq,
                          max_freq=max_freq,
                          min_freq_coord=min_freq_coord,
                          max_freq_coord=max_freq_coord,
                          avg_freq=avg_freq,
                          bandwidth=bandwidth,
                          min_intensity=prop.min_intensity,
                          max_intensity=prop.max_intensity,
                          avg_intensity=prop.mean_intensity,
                          bg_intensity=bg_intensity,
                          area=prop.area,
                          centroid=np.rint(prop.centroid).astype(int),
                          )

        vocal_list.append(new_vocal)
        vocal_id = vocal_id + 1

    del props

    # -- if list is not empty, create a list of vocals
    if len(vocal_list):
        vocal_list = ListOfVocals(vocals_in_recording=np.asarray(vocal_list))
        timeAConnectVocals = time()
        vocal_list.connect_vocals(animal='guineapig')
        vocal_list.connect_vocals(animal='guineapig')
        vocal_list.update_centroids()

        # -- rescale pixel values to save spectrograms in 8bits
        dtype = np.uint8
        Pxx = exposure.rescale_intensity(Pxx, in_range='image', out_range=dtype)
        vocal_list.add_spectrograms_to_vocals(full_spectrogram=np.flipud(Pxx), full_mask=np.flipud(grain), spec_range=100)

        logger.info('[bin {}]: connecting vocals runtime: {:.2f}s'.format(this_bin, time() - timeAConnectVocals))

    logger.info('[bin {}]: list of vocals runtime: {:.2f}s'.format(this_bin, time() - timeAVocal))
    logger.info('[bin {}]: raw number of vocals: {}'.format(this_bin, vocal_id))
    logger.info('[bin {}]: {}'.format(this_bin, vocal_list))
    logger.info('[bin {}]: bin runtime: {:.2f}s'.format(this_bin, time() - timeBinA))

    return vocal_list


def check_if_vocals_are_close(base_vocal, next_vocal):
    # -- conditions to check:
    # -- 1) next vocal starts within 100ms from base vocal start time AND
    # -- next vocal frequency is higher than the base vocal frequency
    # -- 2) next vocal starts within 100ms from base vocal end time AND
    # -- next vocal frequency is higher than the base vocal frequency
    # -- 3) next vocal starts/ends within base vocal start/end (harmonic)
    max_interval = 0.1  # 100ms
    condition_1 = (np.abs(base_vocal.end - next_vocal.start) < max_interval) and (next_vocal.min_freq > base_vocal.max_freq)
    condition_2 = (np.abs(base_vocal.start - next_vocal.start) < max_interval) and (next_vocal.min_freq > base_vocal.max_freq)
    condition_3 = (next_vocal.start >= base_vocal.start and next_vocal.end <= base_vocal.end)

    return True if (condition_1 or condition_2 or condition_3) else False
