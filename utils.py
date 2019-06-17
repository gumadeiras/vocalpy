# -*- coding: utf-8 -*-
'''VocalPy Identifier - Finds candidate vocalizations in exoerimental recordings'''

__author__    = 'Gustavo Madeira Santana'
__email__     = 'gustavo.santana@yale.edu'
__copyright__ = '2019 Dietrich Lab - Yale University School of Medicine'

#ToDo
#Numba maybe

import os
import numpy             as     np
import pandas            as     pd

from   time              import time
from   math              import floor, ceil
from   PIL               import Image

import argparse
import logging

import multiprocessing
from   joblib            import Parallel, delayed

import matplotlib.pyplot as     plt

import cv2

from   scipy             import signal
from   scipy.io          import wavfile
from   skimage           import exposure, measure

def bradley_roth_numpy(image, s=None, t=None):

    # Convert image to numpy array
    img = np.array(image).astype(np.float)

    # Default window size is round(cols/8)
    if s is None:
        s = np.round(img.shape[1]/8)

    # Default threshold is 20% of the total
    # area in the window
    if t is None:
        t = 15.

    # Compute integral image
    intImage = np.cumsum(np.cumsum(img, axis=1), axis=0)

    # Define grid of points
    (rows,cols) = img.shape[:2]
    (X,Y)       = np.meshgrid(np.arange(cols), np.arange(rows))

    # Make into 1D grid of coordinates for easier access
    X = X.ravel()
    Y = Y.ravel()

    # Ensure s is even so that we are able to index into the image
    # properly
    s = s + np.mod(s,2)

    # Access the four corners of each neighbourhood
    x1 = X - s/2
    x2 = X + s/2
    y1 = Y - s/2
    y2 = Y + s/2

    # Ensure no coordinates are out of bounds
    x1[x1 < 0]     = 0
    x2[x2 >= cols] = cols-1
    y1[y1 < 0]     = 0
    y2[y2 >= rows] = rows-1

    # Ensures coordinates are integer
    x1 = x1.astype(np.int)
    x2 = x2.astype(np.int)
    y1 = y1.astype(np.int)
    y2 = y2.astype(np.int)

    # Count how many pixels are in each neighbourhood
    count = (x2 - x1) * (y2 - y1)

    # Compute the row and column coordinates to access
    # each corner of the neighbourhood for the integral image
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

    # Compute areas of each window
    sums = intImage[f1_y, f1_x] - intImage[f2_y, f2_x] - intImage[f3_y, f3_x] + intImage[f4_y, f4_x]

    # Compute thresholded image and reshape into a 2D grid
    out = np.ones(rows*cols, dtype=np.bool)
    out[img.ravel()*count <= sums*(100.0 - t)/100.0] = False

    # Also convert back to uint8
    out = 255*np.reshape(out, (rows, cols)).astype(np.uint8)

    # Return PIL image back to user
    return out


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


def parallel_audio_processing(chunk):
    sample_rate, time_range, this_bin, start_range, end_range, bin_size, args = chunk

    if args.verbose:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s [%(levelname)-5.5s]  %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S',
                            handlers=[
                                logging.FileHandler('{0}/{1}.log'.format('/Users/gustavo/Documents/git/vocalpy/outputs/', 'output')),
                                logging.StreamHandler()
                            ])
    else:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s [%(levelname)-5.5s]  %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S',
                            handlers=[
                                logging.FileHandler('{0}/{1}.log'.format('/Users/gustavo/Documents/git/vocalpy/outputs/', 'output')),
                            ])

    logger = logging.getLogger()

    timeA           = time()
    fs              = sample_rate
    window          = signal.get_window('hamming', 256)
    noverlap        = 128
    nfft            = 1024
    time_range_secs = time_range.shape[0] / sample_rate
    logger.info('[bin {}]: computing spectrogram for bin: {}; time range: {}s; audio range: {:.2f}-{:.2f}s'.format(this_bin, this_bin,
                                                                                                   time_range_secs,
                                                                                                   start_range / sample_rate,
                                                                                                   end_range / sample_rate))
    f, t, Sxx  = signal.spectrogram(time_range, fs=fs,
                                                window=window,
                                                noverlap=noverlap,
                                                nfft=nfft,
                                                mode='psd')
    # logger.info(t[bin {}]: .shape)
    # logger.info(f[bin {}]: .shape)

    # -- remove lower frequencies
    freq_cutoff = 45000
    Sxx         = Sxx[(f>freq_cutoff)]
    f           = f[(f>freq_cutoff)]
    # logger.info(S[bin {}]: xx.shape)
    # logger.info(n[bin {}]: p.min(Sxx))
    # logger.info(n[bin {}]: p.max(Sxx))

    time_res = time_range_secs/t.shape[0]
    freq_res = (np.max(f) - freq_cutoff) / f.shape[0]
    timeB    = time()
    logger.info('[bin {}]: spectrogram runtime: {:.2f}'.format(this_bin, timeB - timeA))
    logger.info('[bin {}]: time resolution: {:.2f}ms'.format(this_bin, time_res * 1000))
    logger.info('[bin {}]: freq resolution: {:.2f}Hz'.format(this_bin, freq_res))


    # -- convert to dB
    Pxx        = 10*np.log10(Sxx)
    # logger.info(n[bin {}]: p.min(Pxx))
    # logger.info(n[bin {}]: p.max(Pxx))
    if args.plot:
        plt.pcolormesh(t[35000:40000], f, Pxx[:,35000:40000], cmap='gray')
        plt.title('Pxx (dB)')
        plt.show()

    # -- rescale to save spectrograms in grayscale
    dtype      = np.uint8
    Pxx_scaled = exposure.rescale_intensity(Pxx, in_range='image', out_range=dtype)

    # -- normalize data
    B = np.abs(Pxx)
    B = B/np.max(B)
    if args.plot:
        plt.pcolormesh(t[35000:40000], f, B[:,35000:40000], cmap='gray')
        plt.title('B = Pxx normalized')
        plt.show()

    # -- rescale intensity values to be between (0,1)
    # B = exposure.rescale_intensity(B, in_range='image', out_range=(0,1))
    B = exposure.rescale_intensity(B, in_range='image', out_range=dtype)
    if args.plot:
        plt.pcolormesh(t[35000:40000], f, B[:,35000:40000], cmap='gray')
        plt.title('B intensity values rescaled to dtype')
        plt.show()

    # -- image complement
    ii8    = np.iinfo(dtype)
    B_orig = B
    B      = ii8.max - B
    if args.plot:
        plt.pcolormesh(t[35000:40000], f, B[:,35000:40000], cmap='gray')
        plt.title('B complement')
        plt.show()


    # -- contrast adjustment
    B = exposure.adjust_gamma(B)
    if args.plot:
        plt.pcolormesh(t[35000:40000], f, B[:,35000:40000], cmap='gray')
        plt.title('B contrast adjustment')
        plt.show()

    # -- binarize spectrogram
    BB = bradley_roth_numpy(B, t=10)
    BB = ii8.max - BB
    if args.plot:
        plt.pcolormesh(t[35000:40000], f, BB[:,35000:40000], cmap='gray')
        plt.title('BB = B binarized bradley roth')
        plt.show()

    # -- kernels for morphological operations
    kernel_rect  = np.ones((4,2), np.uint8)
    kernel_line1 = np.ones((4,1), np.uint8)
    kernel_line2 = np.ones((5,1), np.uint8)

    # -- morphological operations
    erode11  = cv2.erode(BB, kernel_line1, iterations=1)
    dilate12 = cv2.dilate(erode11, kernel_rect, iterations=1)
    dilate13 = cv2.dilate(dilate12, kernel_line2, iterations=1)
    erode14  = cv2.erode(dilate13, kernel_line1, iterations=2)
    if args.plot:
        plt.subplot(511)
        plt.pcolormesh(t[80000:85000], f, BB[:,80000:85000], cmap='gray')
        plt.title('BB')
        plt.subplot(512)
        plt.pcolormesh(t[80000:85000], f, erode11[:,80000:85000], cmap='gray')
        plt.title('BB + erode (4,1)')
        plt.subplot(513)
        plt.pcolormesh(t[80000:85000], f, dilate12[:,80000:85000], cmap='gray')
        plt.title('+ dilate (4,2)')
        plt.subplot(514)
        plt.pcolormesh(t[80000:85000], f, dilate13[:,80000:85000], cmap='gray')
        plt.title('+ dilate (5,1)')
        plt.subplot(515)
        plt.pcolormesh(t[80000:85000], f, erode14[:,80000:85000], cmap='gray')
        plt.title('+ erode (4,1)')
        plt.show()

    timeA  = time()
    connectivity = 4
    num_cc, output, stats, centroids = cv2.connectedComponentsWithStats(erode14, connectivity, cv2.CV_32S)

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
    timeB = time()
    logger.info('[bin {}]: connected components runtime: {:.2f}'.format(this_bin, timeB - timeA))

    # -- one more opening to make sure segmentation covers *at least* the real area
    grain        = grain.astype(np.uint8)
    kernel_cross = np.array([[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]], dtype=np.uint8)
    kernel_line3 = np.ones((1,3), dtype=np.uint8)
    grain        = cv2.dilate(grain, kernel_line3, iterations=1)

    # -- get spectrogram area using the segmentation mask
    B_masked = B * (((255 - grain) > 0) * 1)
    if args.plot:
        plt.subplot(411)
        plt.pcolormesh(t[80000:85000], f, Pxx[:,80000:85000], cmap='gray')
        # plt.pcolormesh(t, f, Pxx, cmap='gray')
        plt.title('Pxx')
        plt.subplot(412)
        plt.pcolormesh(t[80000:85000], f, BB[:,80000:85000], cmap='gray')
        # plt.pcolormesh(t, f, BB, cmap='gray')
        plt.title('BB')
        plt.subplot(413)
        plt.pcolormesh(t[80000:85000], f, grain[:,80000:85000], cmap='gray')
        # plt.pcolormesh(t, f, grain, cmap='gray')
        plt.title('grain')
        plt.subplot(414)
        plt.pcolormesh(t[80000:85000], f, B_masked[:,80000:85000], cmap='gray')
        # plt.pcolormesh(t, f, B_masked, cmap='gray')
        plt.title('B_masked')
        plt.show()

    # -- get connected components stats
    timeA     = time()
    labels = measure.label(grain, background=0)
    props  = measure.regionprops(labels, intensity_image=Pxx, cache=True, coordinates='rc')
    props  = sorted(props, key=lambda p: np.min(p.coords[:,1]), reverse=False)
    timeB    = time()
    logger.info('[bin {}]: region props runtime: {:.2f}'.format(this_bin, timeB - timeA))
    if args.plot:
        plt.subplot(311)
        plt.pcolormesh(t[35000:40000], f, Pxx[:,35000:40000], cmap='gray')
        plt.title('Pxx')
        plt.subplot(312)
        plt.pcolormesh(t[35000:40000], f, B[:,35000:40000], cmap='gray')
        plt.title('B')
        plt.subplot(313)
        plt.pcolormesh(t[35000:40000], f, labels[:,35000:40000], cmap='nipy_spectral')
        plt.title('Labels')
        plt.show()

    # -- region props available
    # area int
    # bbox tuple
    # bbox_area int
    # centroid array
    # coords
    # max_intensity float
    # mean_intensity float
    # min_intensity float
    # orientation float

    vocal_id = 0
    vocal_df = pd.DataFrame(columns=['start',
                                     'end',
                                     'duration',
                                     'interval',
                                     # 'min_freq_main',
                                     # 'max_freq_main',
                                     # 'avg_freq_main',
                                     'min_freq_all',
                                     'max_freq_all',
                                     'avg_freq_all',
                                     'bandwidth',
                                     'min_intensity',
                                     'max_intensity',
                                     'avg_intensity',
                                     'bg_intensity',
                                     'area',
                                     'points',
                                     'centroid',
                                     'orientation',
                                     ])
    
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

        min_freq_all = (np.min(prop.coords[:,0]) * freq_res) + freq_cutoff
        max_freq_all = (np.max(prop.coords[:,0]) * freq_res) + freq_cutoff
        avg_freq_all = (np.mean(prop.coords[:,0]) * freq_res) + freq_cutoff
        bandwidth    = max_freq_all - min_freq_all
        vocal_df     = vocal_df.append({'start': (start * time_res) + ((this_bin - 1) * bin_size),
                                        'end': (end * time_res) + ((this_bin - 1) * bin_size),
                                        'duration': duration * time_res * 1000,
                                        'interval': interval * time_res,
                                        'min_freq_all': min_freq_all,
                                        'max_freq_all': max_freq_all,
                                        'avg_freq_all': avg_freq_all,
                                        'bandwidth': bandwidth,
                                        'min_intensity': prop.min_intensity,
                                        'max_intensity': prop.max_intensity,
                                        'avg_intensity': prop.mean_intensity,
                                        'bg_intensity': 0,
                                        'area': prop.area,
                                        'points': prop.coords,
                                        'centroid': prop.centroid,
                                        'orientation': prop.orientation,
                                        }, ignore_index=True)

        # -- save spectrogram and mask around each vocalization
        centroid_time = ceil(prop.centroid[1])
        spectro_range = 200
        try:
            img = np.flipud(Pxx_scaled[:,centroid_time-200:centroid_time+200])
            img = Image.fromarray(img)
            img = img.convert('L')
            img.save('/Users/gustavo/Documents/git/vocalpy/outputs/all/' + str(this_bin) + '_' + str(vocal_id) + '.jpg')

            img = np.flipud(grain[:,centroid_time-200:centroid_time+200])
            img = Image.fromarray(img)
            img = img.convert('L')
            img.save('/Users/gustavo/Documents/git/vocalpy/outputs/all/mask/' + str(this_bin) + '_' + str(vocal_id) + '.jpg')

            img = np.flipud(B_masked[:,centroid_time-200:centroid_time+200])
            img = Image.fromarray(img)
            img = img.convert('L')
            img.save('/Users/gustavo/Documents/git/vocalpy/outputs/all/' + str(this_bin) + '_' + str(vocal_id) + '_overlay.jpg')
        except:
            logger.info('[bin {}]: ######## EXCEPT HERE FOR ID {}'.format(this_bin, vocal_id))
            img = np.flipud(Pxx_scaled[:,centroid_time-200:-1])
            img = Image.fromarray(img)
            img = img.convert('L')
            img.save('/Users/gustavo/Documents/git/vocalpy/outputs/all/' + str(this_bin) + '_' + str(vocal_id) + '.jpg')

            img = np.flipud(grain[:,centroid_time-200:-1])
            img = Image.fromarray(img)
            img = img.convert('L')
            img.save('/Users/gustavo/Documents/git/vocalpy/outputs/all/mask/' + str(this_bin) + '_' + str(vocal_id) + '.jpg')

            img = np.flipud(B_masked[:,centroid_time-200:-1])
            img = Image.fromarray(img)
            img = img.convert('L')
            img.save('/Users/gustavo/Documents/git/vocalpy/outputs/all/' + str(this_bin) + '_' + str(vocal_id) + '_overlay.jpg')
        vocal_id = vocal_id + 1

    return vocal_df