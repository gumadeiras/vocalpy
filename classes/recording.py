# -*- coding: utf-8 -*-
'''VocalPy - A python version based on VocalMat'''

__email__ = 'gustavo.santana@yale.edu'
__license__ = 'Apache License, Version 2.0'
__copyright__ = '2020 Dietrich Lab - Yale University School of Medicine'

import numpy as np
import pandas as pd
import soundfile as sf

from math import ceil
from os import makedirs
from os.path import join, split, splitext, exists

from utils.io import save_file, load_file, create_directory, remove_directory


class Recording(object):
    '''
    audio recording object and auxiliary functions to process the recording
    '''

    def __init__(self, recording_path, args):
        self.args = args
        self.recording_path = recording_path
        self.recording_dir = None
        self.recording_name = None
        self.spectrogram_dir = None
        self.masks_dir = None
        self.output_dir = None

        self.create_paths(recording_path)

        self.sample_rate = None
        self.samples = None
        self.samples_min = None
        self.samples_max = None
        self.recording_duration = None

        self.read_audio()

        low_freq, high_freq = [int(f) for f in args.frequency.split(',')]
        self.low_frequency_cutoff = low_freq
        self.high_frequency_cutoff = high_freq
        self.bin_size = self.args.bin_size if (self.args.bin_size < self.recording_duration) else self.recording_duration
        self.bins = ceil(self.recording_duration / self.bin_size)
        self.chunks = self.create_chunks()
        self._group_name = 'not set'
        self._list_of_vocals = None
        self._has_list_of_vocals = None

    def __str__(self):
        return '{}:\n duration: {} \n sampling rate: {}'.format(self.__class__.__name__,
                                                                self.recording_duration,
                                                                self.sample_rate)

    @property
    def has_list_of_vocals(self):
        return self._has_list_of_vocals

    @has_list_of_vocals.setter
    def has_list_of_vocals(self, new_has_list_of_vocals):
        self._has_list_of_vocals = new_has_list_of_vocals

    @property
    def list_of_vocals(self):
        return self._list_of_vocals

    @list_of_vocals.setter
    def list_of_vocals(self, new_list_of_vocals):
        self._list_of_vocals = new_list_of_vocals

    @property
    def group_name(self):
        return self._group_name

    @group_name.setter
    def group_name(self, new_group_name):
        self._group_name = new_group_name

    def save_recording_object(self, path, filename='recording'):
        save_file(self, filename, path)

    def create_paths(self, recording_path):
        '''
        create directory structure for output files
        '''
        basepath, filename = split(recording_path)
        self.recording_dir = basepath
        self.recording_name = filename
        filename = splitext(filename)[0]
        self.output_dir = join(self.recording_dir, filename + '_outputs')
        self.spectrogram_dir = join(self.output_dir, 'spectrogram')
        self.mask_dir = join(self.output_dir, 'mask')

        if not exists(self.output_dir):
            makedirs(self.output_dir, exist_ok=True)

        if not exists(self.spectrogram_dir):
            makedirs(self.spectrogram_dir, exist_ok=True)

        if not exists(self.mask_dir):
            makedirs(self.mask_dir, exist_ok=True)

    def read_audio(self):
        '''
        read audio and metadata
        '''
        samples, sample_rate = sf.read(self.recording_path)
        self.sample_rate = sample_rate
        self.samples = samples
        self.samples_min = np.min(samples)
        self.samples_max = np.max(samples)
        self.recording_duration = self.samples.shape[0] / self.sample_rate

    def create_chunks(self, overlap=0.1):
        '''
        separate audio into chunks for parallel or sequential processing
        Args:
            overlap: (float) chunks overlap in seconds
        '''
        chunks = []
        # CREATE A BASELINE CHUNK WITH REPETITIVE INFO AND CONCATENATE
        for this_bin in range(1, self.bins + 1):
            # -- first bin, remove first 0.5 second of recording (usually noisy)
            if this_bin == 1:
                start_range = ceil(0.5 * self.sample_rate)
                end_range = ceil((self.bin_size * self.sample_rate) + (overlap * self.sample_rate))
                sample_range = self.samples[start_range:end_range]
                chunks.append((self.output_dir,
                               self.spectrogram_dir,
                               self.mask_dir,
                               self.sample_rate,
                               sample_range,
                               this_bin,
                               start_range,
                               end_range,
                               self.bin_size,
                               self.low_frequency_cutoff,
                               self.high_frequency_cutoff,
                               self.args))

            elif this_bin == self.bins:  # -- last bin
                start_range = ceil((this_bin - 1) * self.bin_size * self.sample_rate)
                end_range = self.recording_duration * self.sample_rate
                sample_range = self.samples[start_range:]
                chunks.append((self.output_dir,
                               self.spectrogram_dir,
                               self.mask_dir,
                               self.sample_rate,
                               sample_range,
                               this_bin,
                               start_range,
                               end_range,
                               self.bin_size,
                               self.low_frequency_cutoff,
                               self.high_frequency_cutoff,
                               self.args))

            else:  # -- all other bins
                start_range = ceil((this_bin - 1) * self.bin_size * self.sample_rate)
                end_range = ceil((this_bin * self.bin_size * self.sample_rate) + (overlap * self.sample_rate))
                sample_range = self.samples[start_range:end_range]
                chunks.append((self.output_dir,
                               self.spectrogram_dir,
                               self.mask_dir,
                               self.sample_rate,
                               sample_range,
                               this_bin,
                               start_range,
                               end_range,
                               self.bin_size,
                               self.low_frequency_cutoff,
                               self.high_frequency_cutoff,
                               self.args))

        # -- samples are now in chunks, remove from object
        self.samples = None

        return chunks

    def recording_processing_finished(self):
        '''
        recording has already been processed, clear chunks
        '''
        self.chunks = None
        return 0

    def load_list_of_vocals(self):
        return load_file('list_of_vocals', self.output_dir)

    def save_recording_data_to_csv(self, list_of_vocals=None, path=None):
        # -- save metadata to a csv file
        if list_of_vocals is None and self._has_list_of_vocals is not True:
            return -1
        list_of_vocals = list_of_vocals if list_of_vocals is not None else self._list_of_vocals

        if path is None:
            path = self.output_dir

        if list_of_vocals.intervals_fixed is False:
            list_of_vocals.update_intervals()

        recording_df = pd.DataFrame(columns=['bin_number',
                                             'start(s)',
                                             'end(s)',
                                             'duration(ms)',
                                             'interval(s)',
                                             'min_freq',
                                             'max_freq',
                                             'avg_freq',
                                             'bandwidth',
                                             'min_intensity',
                                             'max_intensity',
                                             'avg_intensity',
                                             'bg_intensity',
                                             'area(pixels)',
                                             'centroid_y',
                                             'class_top1',
                                             'class_top2'
                                             ])

        for this_vocal in list_of_vocals.vocals_in_recording:
            recording_df = recording_df.append({'bin_number': this_vocal.bin_number,
                                                'start(s)': this_vocal.start,
                                                'end(s)': this_vocal.end,
                                                'duration(ms)': this_vocal.duration,
                                                'interval(s)': this_vocal.interval,
                                                'min_freq': this_vocal.min_freq,
                                                'max_freq': this_vocal.max_freq,
                                                'avg_freq': this_vocal.avg_freq,
                                                'bandwidth': this_vocal.bandwidth,
                                                'min_intensity': this_vocal.min_intensity,
                                                'max_intensity': this_vocal.max_intensity,
                                                'avg_intensity': this_vocal.avg_intensity,
                                                'bg_intensity': this_vocal.bg_intensity,
                                                'area(pixels)': this_vocal.area,
                                                'centroid_y': this_vocal.centroid[0],
                                                'class_top1': this_vocal.top1,
                                                'class_top2': this_vocal.top2,
                                                }, ignore_index=True)

        # -- sort vocalizations by start time and save csv
        recording_df.sort_values(by='start(s)',
                                 ascending=True,
                                 inplace=True,
                                 kind='quicksort',
                                 na_position='last')

        # -- start index from 1 instead of 0
        recording_df.index = np.arange(1, len(recording_df) + 1)
        recording_df.to_csv(join(self.output_dir, splitext(self.recording_name)[0] + '_stats.csv'), float_format='%.6f')
        return 0

    def save_spectrograms(self, list_of_vocals=None, path=None):
        if self._has_list_of_vocals is not True and list_of_vocals is None:
            return -1
        list_of_vocals = list_of_vocals if list_of_vocals is not None else self._list_of_vocals
        path = path if path is not None else self.output_dir
        remove_directory(join(path, 'spectrogram'))
        create_directory(join(path, 'spectrogram'))
        list_of_vocals.save_spectrograms(output_dir=path)
        return 0

    def save_spectrograms_and_masks(self, list_of_vocals=None, path=None):
        if self._has_list_of_vocals is not True and list_of_vocals is None:
            return -1
        list_of_vocals = list_of_vocals if list_of_vocals is not None else self._list_of_vocals
        path = path if path is not None else self.output_dir
        remove_directory(join(path, 'spectrogram'))
        create_directory(join(path, 'spectrogram'))
        list_of_vocals.save_spectrograms(output_dir=path)
        remove_directory(join(path, 'mask'))
        create_directory(join(path, 'mask'))
        list_of_vocals.save_masks(output_dir=path)
        return 0

    def remove_spectrograms_and_masks_from_object(self, list_of_vocals=None):
        if self._has_list_of_vocals is not True and list_of_vocals is None:
            return -1
        list_of_vocals = list_of_vocals if list_of_vocals is not None else self._list_of_vocals
        list_of_vocals.remove_spectrograms()
        list_of_vocals.remove_masks()
        return 0

    def create_dataset(self, list_of_vocals=None):
        # -- create dataset for the CNN and FCN
        # -- create from list of vocals or save spectrograms
        # -- create filename list etc and create from folder path
        if self.has_list_of_vocals is not True and list_of_vocals is None:
            return -1
        list_of_vocals = list_of_vocals if list_of_vocals is not None else self.load_list_of_vocals()

        print('create_dataset not implemented')
        return 0

    def remove_vocals_classified_as_noise_from_list_of_vocals(self, predictions):
        # -- if list of vocals is empty, there are no predictions
        if isinstance(predictions, int) and predictions == -1:
            return predictions

        # -- remove vocals that were classifier as noise
        self._list_of_vocals.remove_vocals_classified_as_noise(predictions)
        # # -- update inter-vocal intervals after removing noise
        # self._list_of_vocals.update_intervals()
        return 0

    def update_vocals_with_class_classification(self, predictions, classes):
        # -- if list of vocals is empty, there are no predictions
        if isinstance(predictions, int) and predictions == -1:
            return predictions

        # -- make sure number of predictions is the same as number of vocals
        try:
            assert self._list_of_vocals.number_of_vocals == predictions.shape[0]
        except AssertionError:
            print("[error] number of vocals: {}; number of predictions: {}".format(self._list_of_vocals.number_of_vocals, predictions.shape[0]))
            exit()

        # -- add probability distribution for each vocal, top1 and top2 classes
        self._list_of_vocals.add_classification_to_vocals(predictions, classes)
        return 0
