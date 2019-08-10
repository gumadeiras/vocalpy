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
from   utils    import save_file, load_file

class Recording(object):
    '''
    audio recording object and auxiliary functions to process the recordings
    '''
    def __init__(self, recording_path=None, args=None, frequency_cutoff=45000):
        self.args                 = args
        self.recording_path       = recording_path
        self.recording_dir        = None
        self.recording_name       = None
        self.spectrogram_dir      = None
        self.masks_dir            = None
        self.output_dir           = None

        self.create_paths(recording_path)

        self.sample_rate          = None
        self.samples              = None
        # self.samples_normalized = None
        self.recording_duration   = None

        self.read_audio()

        self.frequency_cutoff     = frequency_cutoff
        self.bin_size             = self.args.bin_size
        self.bins                 = ceil(self.recording_duration / self.bin_size)
        self.chunks               = self.create_chunks()
        self._has_list_of_vocals  = None
    
    @property
    def has_list_of_vocals(self):
        return self._has_list_of_vocals
    
    @has_list_of_vocals.setter
    def has_list_of_vocals(self, new_has_list_of_vocals):
        self._has_list_of_vocals = new_has_list_of_vocals

    def save_recording_object(self, path):
        save_file(self, 'recording', path)

    def create_paths(self, recording_path):
        '''
        create directory structure for output files
        '''
        basepath, filename      = os.path.split(recording_path)
        self.recording_dir      = basepath
        self.recording_name     = filename
        self.output_dir         = os.path.join(self.recording_dir, 'outputs')
        self.spectrogram_dir    = os.path.join(self.output_dir, 'spectrogram')
        self.mask_dir           = os.path.join(self.output_dir, 'mask')

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        if not os.path.exists(self.spectrogram_dir):
            os.makedirs(self.spectrogram_dir, exist_ok=True)

        if not os.path.exists(self.mask_dir):
            os.makedirs(self.mask_dir, exist_ok=True)


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
        ############## CREATE A BASELINE CHUNK WITH REPETITIVE INFO AND CONCATENATE
        for this_bin in range(1, self.bins+1):
            if this_bin == 1: # -- first bin, remove first 0.3 seconds of recording (noisy)
                start_range       = ceil(0.3 * self.sample_rate)
                end_range         = self.bin_size * self.sample_rate
                sample_range      = self.samples[start_range:end_range]
                # sample_range_norm = self.samples_normalized[start_range:end_range]
                chunks.append((self.output_dir,
                               self.spectrogram_dir,
                               self.mask_dir,
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

    def recording_processing_finished(self):
        '''
        recording has already been processed, clear chunks 
        '''
        self.chunks = None
        return 0

    def load_list_of_vocals(self):
        return load_file('list_of_vocals', self.output_dir)

    def save_recording_data_to_excel(self, list_of_vocals=None):
        # -- save metadata to an excel file
        if self.has_list_of_vocals != True:
            return -1
        list_of_vocals = list_of_vocals if list_of_vocals is not None else self.load_list_of_vocals()

        if list_of_vocals.intervals_fixed == False:
            list_of_vocals.update_intervals()

        recording_df = pd.DataFrame(columns=['bin_number',
                                             'start',
                                             'end',
                                             'duration',
                                             'interval',
                                             # 'min_freq_main',
                                             # 'max_freq_main',
                                             # 'avg_freq_main',
                                             'min_freq',
                                             'max_freq',
                                             'avg_freq',
                                             'median_freq',
                                             'bandwidth',
                                             'min_intensity',
                                             'max_intensity',
                                             'avg_intensity',
                                             'bg_intensity',
                                             'area',
                                             # 'points',
                                             'centroid',
                                             'orientation',
                                             ])
        
        for this_vocal in list_of_vocals.vocals_in_recording:
            recording_df = recording_df.append({'bin_number'      : this_vocal.bin_number,
                                                'start'           : this_vocal.start,
                                                'end'             : this_vocal.end,
                                                'duration'        : this_vocal.duration,
                                                'interval'        : this_vocal.interval,
                                                'min_freq'        : this_vocal.min_freq,
                                                'max_freq'        : this_vocal.max_freq,
                                                'avg_freq'        : this_vocal.avg_freq,
                                                'median_freq'     : this_vocal.median_freq,
                                                'bandwidth'       : this_vocal.bandwidth,
                                                'min_intensity'   : this_vocal.min_intensity,
                                                'max_intensity'   : this_vocal.max_intensity,
                                                'avg_intensity'   : this_vocal.avg_intensity,
                                                'bg_intensity'    : this_vocal.bg_intensity,
                                                'area'            : this_vocal.area,
                                                # 'points'        : this_vocal.points,
                                                'centroid'        : this_vocal.centroid,
                                                'orientation'     : this_vocal.orientation,
                                                }, ignore_index=True)

        # -- sort vocalizations by start time and save to excel
        recording_df.sort_values(by='start', ascending=True, inplace=True, kind='quicksort', na_position='last')
        recording_df.to_excel(os.path.join(self.output_dir, 'recording_stats.xlsx'))
        return 0

    def save_spectrograms_and_masks(self, list_of_vocals=None, path=None):
        if self.has_list_of_vocals != True and list_of_vocals is None:
            return -1
        list_of_vocals = list_of_vocals if list_of_vocals is not None else self.load_list_of_vocals()
        list_of_vocals.save_spectrograms(output_dir=path)
        list_of_vocals.save_masks(output_dir=path)
        return 0

    def create_dataset(self, list_of_vocals=None):
        # -- create dataset for the CNN and FCN
        # -- create from list of vocals or save spectrograms, create filename list etc and create from folder path?
        if self.has_list_of_vocals != True and list_of_vocals is None:
            return -1
        list_of_vocals = list_of_vocals if list_of_vocals is not None else self.load_list_of_vocals()
        
        print("create_dataset not implemented")
        return 0
