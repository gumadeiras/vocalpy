# -*- coding: utf-8 -*-
'''VocalPy - A python version based on VocalMat'''

__email__     = 'gustavo.santana@yale.edu'
__license__   = 'Apache License, Version 2.0'
__copyright__ = '2019 Dietrich Lab - Yale University School of Medicine'

import os
import cv2
import pickle
import logging
import argparse
import warnings

from vocal               import Vocal
from list_of_vocals      import ListOfVocals

import numpy             as     np
import pandas            as     pd
import matplotlib.pyplot as     plt

from PIL                 import Image
from time                import time
from math                import floor, ceil

from scipy               import signal, ndimage
from skimage             import exposure, measure


def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    plt.imshow(labeled_img)
    plt.show()


def create_logger(args=None, out_dir=None):
    if args.verbose:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s [%(levelname)-5.5s]  %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S',
                            handlers=[
                                logging.FileHandler('{0}/{1}.log'.format(out_dir, 'output')),
                                logging.StreamHandler()
                            ])
        logging.info('verbose output on')
    else:
        print('logging to file: {}'.format(os.path.join(out_dir,'output.log')))
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s [%(levelname)-5.5s]  %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S',
                            handlers=[
                                logging.FileHandler('{0}/{1}.log'.format(out_dir, 'output')),
                            ])

def save_file(file, filename, path):
    if os.path.exists(path)==False:
        raise ValueError("path does not existe: {}".format(path))

    pickle.dump(file, open(os.path.join(path, filename + '.vocalpy'),'wb'))

def load_file(filename, path):
    if os.path.exists(path)==False:
        raise ValueError("path does not existe: {}".format(path))

    return pickle.load(open(os.path.join(path, filename + '.vocalpy'), 'rb'))

def bradley_roth(image, s=None, t=None):
    # -- from somewhere
    img = np.array(image).astype(np.float)

    # -- default window size is round(width/8)
    if s is None:
        s = np.round(img.shape[1]/8)

    # -- default threshold is 15% of the total area in the window
    if t is None:
        t = 15.0

    # -- integral image
    intImage = np.cumsum(np.cumsum(img, axis=1), axis=0)

    # -- define grid of points
    (rows, cols) = img.shape[:2]
    (X, Y)       = np.meshgrid(np.arange(cols), np.arange(rows))

    # -- make into 1D grid of coordinates for easier access
    X = X.ravel()
    Y = Y.ravel()

    # -- ensures is even so that we are able to index the image properly
    s = s + np.mod(s, 2)

    # -- access the four corners of each neighborhood area
    x1 = X - s/2
    x2 = X + s/2
    y1 = Y - s/2
    y2 = Y + s/2

    # -- assert no coordinates are out of bounds
    x1[x1<0]     = 0
    x2[x2>=cols] = cols-1
    y1[y1<0]     = 0
    y2[y2>=rows] = rows-1

    # -- assert coordinates are integers
    x1 = x1.astype(np.int)
    x2 = x2.astype(np.int)
    y1 = y1.astype(np.int)
    y2 = y2.astype(np.int)

    # -- count how many pixels are in each neighborhood
    count = (x2 - x1) * (y2 - y1)

    # -- compute the row and column coordinates to access each corner of the neighborhood for the integral image
    f1_x           = x2
    f1_y           = y2
    f2_x           = x2
    f2_y           = y1 - 1
    f2_y[f2_y < 0] = 0
    f3_x           = x1-1
    f3_x[f3_x < 0] = 0
    f3_y           = y2
    f4_x           = f3_x
    f4_y           = f2_y

    # -- compute areas of each window
    sums = intImage[f1_y, f1_x] - intImage[f2_y, f2_x] - intImage[f3_y, f3_x] + intImage[f4_y, f4_x]

    # -- compute thresholded image and reshape into a 2D grid
    out  = np.zeros(rows*cols, dtype=np.bool)
    out[img.ravel()*count <= sums*(100.0 - t)/100.0] = True

    # -- convert back to uint8
    out  = np.reshape(out, (rows, cols)).astype(np.uint8)

    return out

def parallel_audio_processing(chunk):
    timeBinA = time()

    output_dir, spectrogram_dir, mask_dir, sample_rate, sample_range, this_bin, start_range, end_range, bin_size, frequency_cutoff, args = chunk

    if args.verbose:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s [%(levelname)-5.5s]  %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S',
                            handlers=[
                                logging.FileHandler('{0}/{1}.log'.format(output_dir,  'output')),
                                logging.StreamHandler()
                            ])
    else:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s [%(levelname)-5.5s]  %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S',
                            handlers=[
                                logging.FileHandler('{0}/{1}.log'.format(output_dir, 'output')),
                            ])

    logger = logging.getLogger()

    timeASpectrogram  = time()
    fs                = sample_rate
    window            = signal.get_window('hamming', 256)
    noverlap          = 128
    nfft              = 1024
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

    # -- apply frequency cutoff
    Pxx = Pxx[(f>frequency_cutoff)]
    f   = f[(f>frequency_cutoff)]

    time_res         = sample_range_secs/t.shape[0]
    freq_res         = (np.max(f) - frequency_cutoff) / f.shape[0]
    timeBSpectrogram = time()
    logger.info('[bin {}]: spectrogram runtime: {:.2f}s'.format(this_bin, timeBSpectrogram - timeASpectrogram))
    logger.info('[bin {}]: time resolution: {:.2f}ms'.format(this_bin, time_res * 1000))
    logger.info('[bin {}]: freq resolution: {:.2f}Hz'.format(this_bin, freq_res))

    # -- convert to dB
    Pxx = 10*np.log10(Pxx)
    if args.plot:
        plt.pcolormesh(t[35000:40000], f, Pxx[:,35000:40000], cmap='gray')
        plt.title('Pxx (dB)')
        plt.show()

    # -- normalize data
    # B = Pxx
    # Bmax = np.max(B)
    # Bmin = np.min(B)
    # B    = (B - Bmin) / (Bmax - Bmin)
    B = np.abs(Pxx)
    B = B/np.max(B)
    if args.plot:
        plt.pcolormesh(t[35000:40000], f, B[:,35000:40000], cmap='gray')
        plt.title('B normalized')
        plt.show()
        plt.hist(B.ravel(), bins=256, histtype='step', color='black')
        plt.title('B normalized histogram')
        plt.show()

    # -- contrast adjustment
    p1, p99 = np.percentile(B, (1, 99))
    B[B<p1] = 0
    B[B>p99]= 1
    if args.plot:
        plt.pcolormesh(t[35000:40000], f, B[:,35000:40000], cmap='gray')
        plt.title('B saturated')
        plt.show()
        plt.hist(B.ravel(), bins=256, histtype='step', color='black')
        plt.title('B saturated hist')
        plt.show()

    # -- binarize spectrogram
    B = bradley_roth(B, t=20)
    # B = bradley_roth(np.interp(B, (0,255), (0,1)), t=20)
    # B = ii8.max - B

    if args.plot:
        plt.pcolormesh(t[35000:40000], f, B[:,35000:40000], cmap='gray')
        plt.title('B binarized bradley roth')
        plt.show()    

    # -- median filter
    B = ndimage.median_filter(B, size=(3,3))

    # -- kernels for morphological operations
    kernel_rect  = np.ones((4,2), np.uint8)
    kernel_line1 = np.ones((4,1), np.uint8)
    kernel_line2 = np.ones((5,1), np.uint8)

    # -- morphological operations
    erode11  = cv2.erode(B, kernel_line1, iterations=1)
    dilate12 = cv2.dilate(erode11, kernel_rect, iterations=1)
    dilate13 = cv2.dilate(dilate12, kernel_line2, iterations=1)
    erode14  = cv2.erode(dilate13, kernel_line1, iterations=2)

    if args.plot:
        plt.subplot(511)
        plt.pcolormesh(t[35000:40000], f, B[:,35000:40000], cmap='gray')
        plt.title('B')
        plt.subplot(512)
        plt.pcolormesh(t[35000:40000], f, erode11[:,35000:40000], cmap='gray')
        plt.title('B + erode (4,1)')
        plt.subplot(513)
        plt.pcolormesh(t[35000:40000], f, dilate12[:,35000:40000], cmap='gray')
        plt.title('+ dilate (4,2)')
        plt.subplot(514)
        plt.pcolormesh(t[35000:40000], f, dilate13[:,35000:40000], cmap='gray')
        plt.title('+ dilate (5,1)')
        plt.subplot(515)
        plt.pcolormesh(t[35000:40000], f, erode14[:,35000:40000], cmap='gray')
        plt.title('+ erode (4,1)')
        plt.show()

    del B
    del erode11
    del dilate12
    del dilate13

    timeAConnectedComponents = time()
    connectivity = 4
    num_cc, output, stats, centroids = cv2.connectedComponentsWithStats(erode14, connectivity, cv2.CV_32S)
    del erode14

    # -- remove background stats
    num_cc = num_cc - 1
    areas  = stats[1:,4]
    # nr     = np.arange(num_cc)
    # ranked = sorted(zip(areas,nr))

    # -- filtered connected components placeholder
    grain = np.zeros((output.shape))

    # -- threshold connected components by minimum area
    min_area = 20
    for i in range(0, num_cc):
        if areas[i] >= min_area:
            grain[output == i + 1] = 255
    timeBConnectedComponents = time()
    logger.info('[bin {}]: connected components runtime: {:.2f}s'.format(this_bin, timeBConnectedComponents - timeAConnectedComponents))

    # -- one more opening to make sure segmentation covers *at least* the real area
    grain        = grain.astype(np.uint8)
    kernel_cross = np.array([[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]], dtype=np.uint8)
    kernel_line3 = np.ones((1,3), dtype=np.uint8)
    grain        = cv2.dilate(grain, kernel_line3, iterations=1)

    # -- rescale to save spectrograms in 8bit grayscale
    dtype      = np.uint8
    Pxx_scaled = exposure.rescale_intensity(Pxx, in_range='image', out_range=dtype)
    if args.plot:
        # -- get spectrogram area using the segmentation mask
        B_masked = Pxx_scaled * (((255 - grain) > 0) * 1)
        plt.subplot(411)
        plt.pcolormesh(t[35000:40000], f, Pxx[:,35000:40000], cmap='gray')
        # plt.pcolormesh(t, f, Pxx, cmap='gray')
        plt.title('Pxx')
        plt.subplot(412)
        plt.pcolormesh(t[35000:40000], f, B[:,35000:40000], cmap='gray')
        # plt.pcolormesh(t, f, B, cmap='gray')
        plt.title('B')
        plt.subplot(413)
        plt.pcolormesh(t[35000:40000], f, grain[:,35000:40000], cmap='gray')
        # plt.pcolormesh(t, f, grain, cmap='gray')
        plt.title('grain')
        plt.subplot(414)
        plt.pcolormesh(t[35000:40000], f, B_masked[:,35000:40000], cmap='gray')
        # plt.pcolormesh(t, f, B_masked, cmap='gray')
        plt.title('B_masked')
        plt.show()
        del B_masked

    # -- get connected components stats
    timeARegionProps = time()
    labels = measure.label(grain, background=0)

    props  = measure.regionprops(labels, intensity_image=Pxx, cache=True, coordinates='rc')
    props  = sorted(props, key=lambda p: np.min(p.coords[:,1]), reverse=False)
    timeBRegionProps = time()
    logger.info('[bin {}]: region props runtime: {:.2f}s'.format(this_bin, timeBRegionProps - timeARegionProps))
    if args.plot:
        plt.subplot(311)
        plt.pcolormesh(t[35000:40000], f, Pxx[:,35000:40000], cmap='gray')
        plt.title('Pxx')
        plt.subplot(312)
        plt.pcolormesh(t[35000:40000], f, grain[:,35000:40000], cmap='gray')
        plt.title('grain')
        plt.subplot(313)
        plt.pcolormesh(t[35000:40000], f, labels[:,35000:40000], cmap='nipy_spectral')
        plt.title('Labels')
        plt.show()

    timeAVocal = time()
    vocal_id   = 0
    vocal_list = []

    for prop in props:
        start = np.min(prop.coords[:,1])
        try:
            interval = np.abs(end - start)
        except:
            interval = 0

        end      = np.max(prop.coords[:,1])
        duration = end - start

        if duration < 5:
            continue

        # -- get spectrogram and mask around each vocalization
        spectro_range = 200 # 2*200 * 0.51 = 205ms
        centroid_time = ceil(prop.centroid[1])

        # -- edge conditions, spectro_range goes over the spectrom vector limit (for this bin)
        # with warnings.catch_warnings():
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                bg_intensity  = np.mean(Pxx[:,centroid_time-spectro_range:centroid_time+spectro_range])
            except RuntimeWarning:
                left_end_idx = centroid_time-spectro_range
                if left_end_idx < 0:
                    centroid_time = centroid_time + np.abs(left_end_idx)
                    bg_intensity  = np.mean(Pxx[:,centroid_time-spectro_range:centroid_time+spectro_range])
                else:
                    centroid_time = centroid_time - np.abs(left_end_idx)
                    bg_intensity  = np.mean(Pxx[:,centroid_time-spectro_range:centroid_time+spectro_range])
            warnings.simplefilter('ignore')
            warnings.filterwarnings('ignore')

        # ToDo implement local median filter
        if (prop.mean_intensity/bg_intensity) > 0.92:
            continue

        min_freq    = (np.min(prop.coords[:,0]) * freq_res) + frequency_cutoff
        max_freq    = (np.max(prop.coords[:,0]) * freq_res) + frequency_cutoff
        avg_freq    = (np.mean(prop.coords[:,0]) * freq_res) + frequency_cutoff
        median_freq = np.median(prop.coords[:,0])
        median_freq = (median_freq * freq_res) + frequency_cutoff
        bandwidth   = max_freq - min_freq

        new_vocal = Vocal(bin_number      = this_bin,
                          start           = (start * time_res) + ((this_bin - 1) * bin_size),
                          end             = (end * time_res) + ((this_bin - 1) * bin_size),
                          duration        = duration * time_res * 1000,
                          interval        = interval * time_res,
                          min_freq        = min_freq,
                          max_freq        = max_freq,
                          avg_freq        = avg_freq,
                          median_freq     = median_freq,
                          bandwidth       = bandwidth,
                          min_intensity   = prop.min_intensity,
                          max_intensity   = prop.max_intensity,
                          avg_intensity   = prop.mean_intensity,
                          bg_intensity    = bg_intensity,
                          area            = prop.area,
                          # centroid        = [ceil(prop.centroid[1]), ceil(prop.centroid[0])],
                          centroid        = np.rint(prop.centroid).astype(int),
                          )

        # img = np.flipud(Pxx_scaled[:,centroid_time-spectro_range:centroid_time+spectro_range])
        # new_vocal.spectrogram = img
        # new_vocal.save_spectrogram_as_image(path=output_dir, filename=str(vocal_id))

        # img = np.flipud(grain[:,centroid_time-spectro_range:centroid_time+spectro_range])
        # new_vocal.mask = img
        # new_vocal.save_mask_as_image(path=output_dir, filename=str(vocal_id))

        vocal_list.append(new_vocal)
        vocal_id = vocal_id + 1

    if len(vocal_list):
        # -- if list is not empty, create a list of vocals
        vocal_list = ListOfVocals(vocals_in_recording=np.asarray(vocal_list))
        timeAConnectVocals = time()
        vocal_list.connect_vocals()
        vocal_list.connect_vocals()
        vocal_list.add_spectrograms_to_vocals(full_spectrogram=np.flipud(Pxx_scaled), full_mask=grain, spec_range=200)
        timeBConnectVocals = time()
        logger.info('[bin {}]: connecting vocals runtime: {:.2f}s'.format(this_bin, timeBConnectVocals - timeAConnectVocals))

    timeBVocal = time()
    logger.info('[bin {}]: list of vocals runtime: {:.2f}s'.format(this_bin, timeBVocal - timeAVocal))

    timeBinB = time()
    logger.info('[bin {}]: raw number of vocals: {}'.format(this_bin, vocal_id))
    logger.info('[bin {}]: {}'.format(this_bin, vocal_list))
    logger.info('[bin {}]: bin runtime: {:.2f}s'.format(this_bin, timeBinB - timeBinA))

    return vocal_list