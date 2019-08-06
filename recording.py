# -*- coding: utf-8 -*-
'''VocalPy - A python version based on (VocalMat by Antonio Fonseca)'''

__author__    = 'Gustavo Madeira Santana'
__email__     = 'gustavo.santana@yale.edu'
__copyright__ = '2019 Dietrich Lab - Yale University School of Medicine'


import os
import argparse
import logging

import numpy    as     np
import pandas   as     pd

from   time     import time
from   math     import ceil
from   scipy.io import wavfile

class Recording(object):
    '''
    audio recording object and auxiliary functions to process the recordings
    '''
    def __init__(self, recording_path=None, args=None, frequency_cutoff=45000):
        self.args               = args
        self.recording_path     = recording_path
        self.recording_dir      = None
        self.recording_name     = None
        self.spectrogram_dir    = None
        self.masks_dir          = None
        self.overlay_dir        = None
        self.output_dir         = None

        self.create_paths(recording_path)

        self.sample_rate        = None
        self.samples            = None
        # self.samples_normalized = None
        self.recording_duration = None

        self.read_audio()

        self.frequency_cutoff   = frequency_cutoff
        self.bin_size           = self.args.bin_size
        self.bins               = ceil(self.recording_duration / self.bin_size)
        self.chunks             = self.create_chunks()
        self.list_of_vocals     = None

    def create_paths(self, recording_path):
        '''
        create directory structure for output files
        '''
        basepath, filename      = os.path.split(recording_path)
        self.recording_dir      = basepath
        self.recording_name     = filename
        self.output_dir         = os.path.join(self.recording_dir, 'outputs')
        self.spectrogram_dir    = os.path.join(self.output_dir, 'spectrogram')
        self.mask_dir           = os.path.join(self.output_dir, 'segmentation')
        self.overlay_dir        = os.path.join(self.output_dir, 'overlay')

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        if not os.path.exists(self.spectrogram_dir):
            os.makedirs(self.spectrogram_dir, exist_ok=True)

        if not os.path.exists(self.mask_dir):
            os.makedirs(self.mask_dir, exist_ok=True)

        if not os.path.exists(self.overlay_dir):
            os.makedirs(self.overlay_dir, exist_ok=True)

    def read_audio(self):
        '''
        read audio and metadata
        '''
        sample_rate, samples    = wavfile.read(self.recording_path)
        self.sample_rate        = sample_rate
        self.samples            = samples
        self.recording_duration = self.samples.shape[0] / self.sample_rate
        # self.samples_normalized = self.samples / np.iinfo(np.int16).max # -- recordings are 16bit, might need to change

    def create_chunks(self):
        '''
        separate audio into chunks for parallel or sequential processing
        '''
        chunks = []
        for this_bin in range(1, self.bins+1):
            if this_bin == 1: # -- first bin, remove first 0.3 seconds of recording (noisy)
                start_range       = ceil(0.3 * self.sample_rate)
                end_range         = self.bin_size * self.sample_rate
                sample_range      = self.samples[start_range:end_range]
                # sample_range_norm = self.samples_normalized[start_range:end_range]
                chunks.append((self.output_dir,
                               self.spectrogram_dir,
                               self.mask_dir,
                               self.overlay_dir,
                               self.sample_rate,
                               sample_range,
                               # sample_range_norm,
                               this_bin,
                               start_range,
                               end_range,
                               self.bin_size,
                               self.frequency_cutoff,
                               self.args))

            elif this_bin == self.bins: # -- last bin
                start_range       = (this_bin - 1) * self.bin_size * self.sample_rate
                end_range         = self.recording_duration * self.sample_rate
                sample_range      = self.samples[start_range:]
                # sample_range_norm = self.samples_normalized[start_range:]
                chunks.append((self.output_dir,
                               self.spectrogram_dir,
                               self.mask_dir,
                               self.overlay_dir,
                               self.sample_rate,
                               sample_range,
                               # sample_range_norm,
                               this_bin,
                               start_range,
                               end_range,
                               self.bin_size,
                               self.frequency_cutoff,
                               self.args))

            else: # -- all other bins
                start_range       = (this_bin - 1) * self.bin_size * self.sample_rate
                end_range         = this_bin * self.bin_size * self.sample_rate
                sample_range      = self.samples[start_range:end_range]
                # sample_range_norm = self.samples_normalized[start_range:end_range]
                chunks.append((self.output_dir,
                               self.spectrogram_dir,
                               self.mask_dir,
                               self.overlay_dir,
                               self.sample_rate,
                               sample_range,
                               # sample_range_norm,
                               this_bin,
                               start_range,
                               end_range,
                               self.bin_size,
                               self.frequency_cutoff,
                               self.args))

        # -- samples are now in chuncks, remove from object
        self.samples  = None
        # self.samples_normalized = None

        return chunks