# -*- coding: utf-8 -*-
'''VocalPy - A python version based on VocalMat'''

__email__ = 'gustavo.santana@yale.edu'
__license__ = 'Apache License, Version 2.0'
__copyright__ = '2020 Dietrich Lab - Yale University School of Medicine'

import cv2
import logging
import numpy as np
import matplotlib.pyplot as plt

from os.path import join
from multiprocessing import cpu_count


def create_logger(args=None, out_dir=None):
    if args.verbose:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s [%(levelname)-5.5s]  %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S',
                            handlers=[
                                logging.FileHandler(
                                    '{0}/{1}.log'.format(out_dir, 'output')),
                                logging.StreamHandler()
                            ])
        logging.info('verbose output on')
    else:
        print('logging to file: {}'.format(join(out_dir, 'output.log')))
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s [%(levelname)-5.5s]  %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S',
                            handlers=[
                                logging.FileHandler(
                                    '{0}/{1}.log'.format(out_dir, 'output')),
                            ])


def get_core_count(threads=0):
    '''
    use number of threads provided by the user or get the cpu core count
    '''
    if threads > 0:
        return threads
    else:
        return cpu_count()


def validate_arguments(args, num_cores):
    validate_bin_size(args.bin_size)
    validate_frequency_range(args.frequency)
    validate_thread_count(args.threads, num_cores)
    return 0


def validate_bin_size(bin_size):
    if bin_size < 0:
        print('bin_size must be a positive integer.')
        print('provided value: {:2f}'.format(bin_size))
        exit()
    return 0


def validate_frequency_range(frequency_range):
    low_freq, high_freq = frequency_range
    if (low_freq > high_freq) & (high_freq != -1):
        print('low frequency cutoff must be lower than the high frequency cutoff.')
        print('provided values: low_freq={}; high_freq={}'.format(low_freq, high_freq))
        exit()
    return 0


def validate_thread_count(threads, num_cores):
    if threads < 0:
        print('number of threads must be a positive integer.')
        print('provded value: {}'.format(threads))
        print('computer core count: {}'.format(num_cores))
        exit()
    if threads > num_cores:
        print('WARNING: number of threads is higher than number of available cores.')
        print('provded value: {}'.format(threads))
        print('computer core count: {}'.format(num_cores))
    return 0


def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    plt.imshow(labeled_img)
    plt.show()