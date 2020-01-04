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

from classes.vocal import Vocal
from classes.list_of_vocals import ListOfVocals


def bradley_roth(image, s=None, t=None):
    # -- from somewhere
    img = np.array(image).astype(np.float)

    # -- default window size is round(width/8)
    if s is None:
        s = np.round(img.shape[1] / 8)

    # -- default threshold is 15% of the total area in the window
    if t is None:
        t = 15.0

    # -- integral image
    intImage = np.cumsum(np.cumsum(img, axis=1), axis=0)

    # -- define grid of points
    (rows, cols) = img.shape[:2]
    (X, Y) = np.meshgrid(np.arange(cols), np.arange(rows))

    # -- make into 1D grid of coordinates for easier access
    X = X.ravel()
    Y = Y.ravel()

    # -- ensures is even so that we are able to index the image properly
    s = s + np.mod(s, 2)

    # -- access the four corners of each neighborhood area
    x1 = X - s / 2
    x2 = X + s / 2
    y1 = Y - s / 2
    y2 = Y + s / 2

    # -- assert no coordinates are out of bounds
    x1[x1 < 0] = 0
    x2[x2 >= cols] = cols - 1
    y1[y1 < 0] = 0
    y2[y2 >= rows] = rows - 1

    # -- assert coordinates are integers
    x1 = x1.astype(np.int)
    x2 = x2.astype(np.int)
    y1 = y1.astype(np.int)
    y2 = y2.astype(np.int)

    # -- count how many pixels are in each neighborhood
    count = (x2 - x1) * (y2 - y1)

    # -- compute the row and column coordinates to access each
    # -- corner of the neighborhood for the integral image
    f1_x = x2
    f1_y = y2
    f2_x = x2
    f2_y = y1 - 1
    f2_y[f2_y < 0] = 0
    f3_x = x1 - 1
    f3_x[f3_x < 0] = 0
    f3_y = y2
    f4_x = f3_x
    f4_y = f2_y

    # -- compute areas of each window
    sums = intImage[f1_y, f1_x] - intImage[f2_y, f2_x] - \
        intImage[f3_y, f3_x] + intImage[f4_y, f4_x]

    # -- compute thresholded image and reshape into a 2D grid
    out = np.zeros(rows * cols, dtype=np.bool)
    out[img.ravel() * count <= sums * (100.0 - t) / 100.0] = True

    # -- convert back to uint8
    out = np.reshape(out, (rows, cols)).astype(np.uint8)

    return out


def parallel_audio_processing(animal, chunk):
    if animal in ['mouse' or 'rat']:
        return mouse_rat_pipeline(chunk)
    elif animal == 'guineapig':
        return guinea_pig_pipeline(chunk)


def mouse_rat_pipeline(chunk):
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
    window = signal.get_window('hamming', 256)
    noverlap = 128
    nfft = 1024
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
    B = bradley_roth(B, t=20)

    # -- median filter
    B = ndimage.median_filter(B, size=(3, 3))

    # -- kernels for morphological operations
    kernel_rect = np.ones((4, 2), np.uint8)
    kernel_line1 = np.ones((4, 1), np.uint8)
    kernel_line2 = np.ones((5, 1), np.uint8)

    # -- morphological operations
    erode11 = cv2.erode(B, kernel_line1, iterations=1)
    del B

    dilate12 = cv2.dilate(erode11, kernel_rect, iterations=1)
    del erode11

    dilate13 = cv2.dilate(dilate12, kernel_line2, iterations=1)
    del dilate12

    erode14 = cv2.erode(dilate13, kernel_line1, iterations=2)
    del dilate13

    timeAConnectedComponents = time()
    connectivity = 4
    num_cc, output, stats, centroids = cv2.connectedComponentsWithStats(erode14, connectivity, cv2.CV_32S)
    del erode14

    # -- remove background stats
    num_cc = num_cc - 1
    areas = stats[1:, 4]

    # -- filtered connected components placeholder
    grain = np.zeros((output.shape))

    # -- threshold connected components by minimum area
    min_area = 20
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
        spectro_range = 200  # 2*200 * 0.51 = 205ms
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

        min_freq_coord = np.min(prop.coords[:, 0])
        max_freq_coord = np.max(prop.coords[:, 0])
        min_freq = (min_freq_coord * freq_res) + low_frequency_cutoff
        max_freq = (max_freq_coord * freq_res) + low_frequency_cutoff
        avg_freq = (np.mean(prop.coords[:, 0]) * freq_res) + low_frequency_cutoff
        bandwidth = max_freq - min_freq

        new_vocal = Vocal(bin_number=this_bin,
                          start=(start * time_res) + ((this_bin - 1) * bin_size),
                          end=(end * time_res) + ((this_bin - 1) * bin_size),
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
        vocal_list.connect_vocals(animal='mouse')
        vocal_list.connect_vocals(animal='mouse')
        vocal_list.update_centroids()

        # -- rescale pixel values to save spectrograms in 8bits
        dtype = np.uint8
        Pxx = exposure.rescale_intensity(Pxx, in_range='image', out_range=dtype)
        vocal_list.add_spectrograms_to_vocals(full_spectrogram=np.flipud(Pxx), full_mask=np.flipud(grain), spec_range=206) # 206 ~ 210ms @ 0.51ms resolution

        logger.info('[bin {}]: connecting vocals runtime: {:.2f}s'.format(this_bin, time() - timeAConnectVocals))

    logger.info('[bin {}]: list of vocals runtime: {:.2f}s'.format(this_bin, time() - timeAVocal))
    logger.info('[bin {}]: raw number of vocals: {}'.format(this_bin, vocal_id))
    logger.info('[bin {}]: {}'.format(this_bin, vocal_list))
    logger.info('[bin {}]: bin runtime: {:.2f}s'.format(this_bin, time() - timeBinA))

    return vocal_list


def guinea_pig_pipeline(chunk):
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

        min_freq_coord = np.min(prop.coords[:, 0])
        max_freq_coord = np.max(prop.coords[:, 0])
        min_freq = (min_freq_coord * freq_res) + low_frequency_cutoff
        max_freq = (max_freq_coord * freq_res) + low_frequency_cutoff
        avg_freq = (np.mean(prop.coords[:, 0]) * freq_res) + low_frequency_cutoff
        bandwidth = max_freq - min_freq

        new_vocal = Vocal(bin_number=this_bin,
                          start=(start * time_res) + ((this_bin - 1) * bin_size),
                          end=(end * time_res) + ((this_bin - 1) * bin_size),
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
        vocal_list.add_spectrograms_to_vocals(full_spectrogram=np.flipud(Pxx), full_mask=np.flipud(grain), spec_range=50)

        logger.info('[bin {}]: connecting vocals runtime: {:.2f}s'.format(this_bin, time() - timeAConnectVocals))

    logger.info('[bin {}]: list of vocals runtime: {:.2f}s'.format(this_bin, time() - timeAVocal))
    logger.info('[bin {}]: raw number of vocals: {}'.format(this_bin, vocal_id))
    logger.info('[bin {}]: {}'.format(this_bin, vocal_list))
    logger.info('[bin {}]: bin runtime: {:.2f}s'.format(this_bin, time() - timeBinA))

    return vocal_list