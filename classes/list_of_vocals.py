# -*- coding: utf-8 -*-
'''VocalPy - A python version based on VocalMat'''

__email__ = 'gustavo.santana@yale.edu'
__license__ = 'Apache License, Version 2.0'
__copyright__ = '2020 Dietrich Lab - Yale University School of Medicine'

import numpy as np

from classes.vocal import Vocal


class ListOfVocals(object):
    '''
    list of vocalizations (Vocal class)
    '''

    def __init__(self, vocals_in_recording=None):
        if vocals_in_recording is not None:
            self.vocals_in_recording = np.hstack(vocals_in_recording)
            self.number_of_vocals = len(self.vocals_in_recording)
        else:
            self.vocals_in_recording = None
            self.number_of_vocals = None

        self.vocals_combined = False
        self.intervals_fixed = False
        self.centroid_spectro_fixed = False

    def __str__(self):
        # return '{}: vocals_in_recording: {} \n number_of_vocals: {} \n vocals_processed: {}'.format(self.__class__.__name__, self.vocals_in_recording, self.number_of_vocals, self.vocals_processed)
        return '{}:\n number_of_vocals: {} \n vocals_combined: {} \n intervals_fixed: {} \n centroid_spectro_fixed: {}'.format(self.__class__.__name__,
                                                                                                                               self.number_of_vocals,
                                                                                                                               self.vocals_combined,
                                                                                                                               self.intervals_fixed,
                                                                                                                               self.centroid_spectro_fixed)

    def save_list_of_vocals_object(self, path):
        from utils import save_file
        save_file(self, 'list_of_vocals', path)

    def update_intervals(self):
            # -- go through vocals and update inter vocal times
        self.vocals_in_recording[0].interval = 0

        for idx in range(1, len(self.vocals_in_recording), 1):
            self.vocals_in_recording[idx].interval = np.abs((self.vocals_in_recording[idx - 1].end - self.vocals_in_recording[idx].start))

        self.intervals_fixed = True
        return 0

    def connect_vocals(self):
        # -- connects vocals that are at most 12ms apart
        def combine_vocals(first_vocal, second_vocal):
            # -- combines two vocals into one
            start_difference = first_vocal.start - second_vocal.start
            end_difference = first_vocal.end - second_vocal.end

            # -- new centroid will be updated from vocal start/end and min/max frequency
            combined_vocal = Vocal(bin_number=first_vocal.bin_number if (first_vocal.bin_number < second_vocal.bin_number) else second_vocal.bin_number,
                                   start=first_vocal.start if (start_difference < 0) else second_vocal.start,
                                   start_coord=first_vocal.start_coord if (start_difference < 0) else second_vocal.start_coord,
                                   end=first_vocal.end if (end_difference > 0) else second_vocal.end,
                                   end_coord=first_vocal.end_coord if (end_difference > 0) else second_vocal.end_coord,
                                   interval=-1,
                                   min_freq=first_vocal.min_freq if (first_vocal.min_freq < second_vocal.min_freq) else second_vocal.min_freq,
                                   max_freq=first_vocal.max_freq if (first_vocal.max_freq > second_vocal.max_freq) else second_vocal.max_freq,
                                   min_freq_coord=first_vocal.min_freq_coord if (first_vocal.min_freq_coord < second_vocal.min_freq_coord) else second_vocal.min_freq_coord,
                                   max_freq_coord=first_vocal.max_freq_coord if (first_vocal.max_freq_coord > second_vocal.max_freq_coord) else second_vocal.max_freq_coord,
                                   avg_freq=np.mean((first_vocal.avg_freq, second_vocal.avg_freq)),
                                   min_intensity=first_vocal.min_intensity if (first_vocal.min_intensity < second_vocal.min_intensity) else second_vocal.min_intensity,
                                   max_intensity=first_vocal.max_intensity if (first_vocal.max_intensity < second_vocal.max_intensity) else second_vocal.max_intensity,
                                   avg_intensity=np.mean((first_vocal.avg_intensity, second_vocal.avg_intensity)),
                                   bg_intensity=np.mean((first_vocal.bg_intensity, second_vocal.bg_intensity)),
                                   area=first_vocal.area + second_vocal.area,
                                   centroid=first_vocal.centroid)
            combined_vocal.duration = combined_vocal.end - combined_vocal.start
            combined_vocal.bandwidth = combined_vocal.max_freq - combined_vocal.min_freq

            return combined_vocal

        new_list_of_vocals = []

        idx = 0
        there_are_vocals = True
        while there_are_vocals is True:
            # merge this vocals with vocals that are within 12ms
            try:
                base_vocal = self.vocals_in_recording[idx]
                new_vocal = base_vocal
                next_vocal = self.vocals_in_recording[idx + 1]
            except:
                # -- there are no more vocals to connect
                new_list_of_vocals.append(new_vocal)
                there_are_vocals = False
                break

            # -- conditions to check:
            # -- 1) next vocal starts within 12ms from base vocal start time
            # -- 2) next vocal starts within 12ms from base vocal end time
            # -- 3) next vocal starts within base vocal start/end (harmonic)
            max_interval = 0.011  # 12ms - 1ms error because morph ops increase area
            next_vocal_is_close = True if (np.abs(base_vocal.end - next_vocal.start) < max_interval or np.abs(base_vocal.start - next_vocal.start) < max_interval or (next_vocal.start >= base_vocal.start and next_vocal.start <= base_vocal.end)) else False

            idx = idx + 1
            while next_vocal_is_close is True:
                new_vocal = combine_vocals(new_vocal, next_vocal)
                try:
                    next_vocal = self.vocals_in_recording[idx + 1]
                    next_vocal_is_close = True if (np.abs(new_vocal.end - next_vocal.start) < max_interval or np.abs(new_vocal.start - next_vocal.start) < max_interval or (next_vocal.start >= new_vocal.start and next_vocal.end <= new_vocal.end)) else False
                except:
                    next_vocal_is_close = False
                idx = idx + 1

            # when last vocal is combined, sometimes gets duplicated; check why
            new_list_of_vocals.append(new_vocal)

        try:
            self.vocals_in_recording = np.hstack(new_list_of_vocals)
        except:
            self.vocals_in_recording = new_list_of_vocals
        self.number_of_vocals = len(self.vocals_in_recording)
        self.vocals_combined = True
        return 0

    def update_centroids(self):
        '''
        update centroids for each vocal
        absolute centroid from vocal coordinates start/end, and min/max frequency
        '''
        for vocal in self.vocals_in_recording:
            cx = vocal.start_coord + ((vocal.end_coord - vocal.start_coord) // 2)
            cy = vocal.min_freq_coord + ((vocal.max_freq_coord - vocal.min_freq_coord) // 2)
            vocal.centroid = np.rint([cy, cx]).astype(np.int)

        self.centroid_spectro_fixed = True
        return 0

    def combine_list_of_list_of_vocals(self, list_of_list_of_vocals):
        new_list_of_vocals = []
        for list_of_vocals in list_of_list_of_vocals:
            try:
                new_list_of_vocals.append(np.hstack(list_of_vocals.vocals_in_recording))
            except:
                # empty list
                continue

        self.vocals_in_recording = np.hstack(new_list_of_vocals)
        self.number_of_vocals = len(self.vocals_in_recording)
        self.vocals_combined = True
        self.centroid_spectro_fixed = True
        return 0

    def add_spectrograms_to_vocals(self, full_spectrogram, full_mask, spec_range=200):
        for vocal in self.vocals_in_recording:
            cy, cx = vocal.centroid
            spec_max = full_spectrogram.shape[1]

            lower = cx - spec_range
            lower = lower if lower > 0 else 0
            higher = min(cx + spec_range, spec_max)

            vocal.spectrogram = full_spectrogram[:, lower:higher]
            vocal.mask = full_mask[:, lower:higher]
            vocal.centroid = [vocal.centroid[0], spec_range]
            self.centroid_spectro_fixed = True
        return 0

    def save_spectrograms(self, output_dir=None):
        for filename, vocal in enumerate(self.vocals_in_recording, start=1):
            vocal.save_spectrogram_as_image(path=output_dir, filename=str(filename))
        return 0

    def save_masks(self, output_dir=None):
        for filename, vocal in enumerate(self.vocals_in_recording, start=1):
            vocal.save_mask_as_image(path=output_dir, filename=str(filename))
        return 0

    def remove_spectrograms(self):
        for vocal in self.vocals_in_recording:
            vocal._spectrogram = None
        return 0

    def remove_masks(self):
        for vocal in self.vocals_in_recording:
            vocal._mask = None
        return 0

    def remove_vocals_classified_as_noise(self, predictions):
        self.vocals_in_recording = self.vocals_in_recording[predictions]
        self.number_of_vocals = len(self.vocals_in_recording)
        self.update_intervals()
        return 0
