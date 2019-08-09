# -*- coding: utf-8 -*-
'''VocalPy - A python version based on (VocalMat by Antonio Fonseca)'''

__author__    = 'Gustavo Madeira Santana'
__email__     = 'gustavo.santana@yale.edu'
__copyright__ = '2019 Dietrich Lab - Yale University School of Medicine'

import numpy as     np

from   vocal import Vocal
from   PIL   import Image

class ListOfVocals(object):
    '''
    list of vocalizations
    '''
    def __init__(self, vocals_in_recording=None):
        if vocals_in_recording is not None:
            self.vocals_in_recording = np.hstack(vocals_in_recording)
            self.number_of_vocals    = len(self.vocals_in_recording)
        else:
            self.vocals_in_recording = None
            self.number_of_vocals    = None
        
        self.vocals_combined         = False
        self.intervals_fixed         = False

    def __str__(self):
        # return "{}: vocals_in_recording: {} \n number_of_vocals: {} \n vocals_processed: {}".format(self.__class__.__name__, self.vocals_in_recording, self.number_of_vocals, self.vocals_processed)
        return "{}:\n number_of_vocals: {} \n vocals_combined: {} \n intervals_fixed: {}".format(self.__class__.__name__, self.number_of_vocals, self.vocals_combined, self.intervals_fixed)

    def save_list_of_vocals_object(self, path):
        from utils import save_file
        save_file(self, 'list_of_vocals', path)
    
    def update_intervals(self):
            # -- go through vocals and update inter vocal times
            self.vocals_in_recording[0].interval = 0

            for idx in range(1, len(self.vocals_in_recording),1):
                    self.vocals_in_recording[idx].interval = np.abs((self.vocals_in_recording[idx-1].end - self.vocals_in_recording[idx].start) * 1000)

            self.intervals_fixed = True
            return 0

    def connect_vocals(self):
    # -- connects vocals that are less than 12ms apart
        def combine_vocals(first_vocal, second_vocal):
            # -- combines vocals into one
            combined_vocal = Vocal(bin_number    = first_vocal.bin_number if (first_vocal.bin_number < second_vocal.bin_number) else second_vocal.bin_number,
                                   start         = first_vocal.start if (first_vocal.start < second_vocal.start) else second_vocal.start,
                                   end           = first_vocal.end if (first_vocal.end > second_vocal.end) else second_vocal.end,
                                   duration      = combined.end - combined.start,
                                   interval      = [-1],
                                   min_freq      = first_vocal.min_freq if (first_vocal.min_freq < second_vocal.min_freq) else second_vocal.min_freq,
                                   max_freq      = first_vocal.max_freq if (first_vocal.max_freq > second_vocal.max_freq) else second_vocal.max_freq,
                                   avg_freq      = np.mean((first_vocal.avg_freq, second_vocal.avg_freq)),
                                   bandwidth     = combined_vocal.max_freq - combined_vocal.min_freq,
                                   min_intensity = first_vocal.min_intensity if (first_vocal.min_intensity < second_vocal.min_intensity) else second_vocal.min_intensity,
                                   max_intensity = first_vocal.max_intensity if (first_vocal.max_intensity < second_vocal.max_intensity) else second_vocal.max_intensity,
                                   avg_intensity = np.mean((first_vocal.avg_intensity, second_vocal.avg_intensity)),
                                   bg_intensity  = np.mean((first_vocal.bg_intensity, second_vocal.bg_intensity)),
                                   area          = first_vocal.area + second_vocal.area,
                                   centroid      = np.mean(np.vstack((first_vocal.centroid, second_vocal.centroid)), axis=0),
                                   orientation   = None)
            # combined_vocal.bin_number    = first_vocal.bin_number if (first_vocal.bin_number < second_vocal.bin_number) else second_vocal.bin_number
            # combined_vocal.start         = first_vocal.start if (first_vocal.start < second_vocal.start) else second_vocal.start
            # combined_vocal.end           = first_vocal.end if (first_vocal.end > second_vocal.end) else second_vocal.end
            # combined_vocal.duration      = combined.end - combined.start
            # combined_vocal.interval      = [-1]
            # combined_vocal.min_freq      = first_vocal.min_freq if (first_vocal.min_freq < second_vocal.min_freq) else second_vocal.min_freq
            # combined_vocal.max_freq      = first_vocal.max_freq if (first_vocal.max_freq > second_vocal.max_freq) else second_vocal.max_freq
            # combined_vocal.avg_freq      = np.mean((first_vocal.avg_freq, second_vocal.avg_freq))
            # combined_vocal.bandwidth     = combined_vocal.max_freq - combined_vocal.min_freq
            # combined_vocal.min_intensity = first_vocal.min_intensity if (first_vocal.min_intensity < second_vocal.min_intensity) else second_vocal.min_intensity
            # combined_vocal.max_intensity = first_vocal.max_intensity if (first_vocal.max_intensity < second_vocal.max_intensity) else second_vocal.max_intensity
            # combined_vocal.avg_intensity = np.mean((first_vocal.avg_intensity, second_vocal.avg_intensity))
            # combined_vocal.bg_intensity  = np.mean((first_vocal.bg_intensity, second_vocal.bg_intensity))
            # combined_vocal.area          = first_vocal.area + second_vocal.area
            # combined_vocal.centroid      = np.mean(np.vstack((first_vocal.centroid, second_vocal.centroid)), axis=0)
            # combined_vocal.orientation   = None

            combined_vocal.spectrogram   = -1
            combined_vocal.mask          = -1

            return combined_vocal

        new_list_of_vocals = []

        vocal_idx = 0
        there_are_vocals = True
        while there_are_vocals == True:
            # look at this vocals and next ones until the next doesn't belong with this one
            base_vocal = self.list_of_vocals[vocal_idx]
            next_vocal = self.list_of_vocals[vocal_idx+1]

            # -- conditions to check:
            # -- 1) next vocal starts within 12ms from base vocal start time
            # -- 2) next vocal starts within 12ms from base vocal end time
            # -- 3) next vocal starts within base vocal start/end (harmonic)
            next_vocal_is_close = True if (np.abs(base_vocal.start - next_vocal.start) <= 12 or np.abs(base_vocal.end   - next_vocal.start) <= 12 or next_vocal.start >= base_vocal.start & next_vocal.end <= base_vocal.end) else False

            # while (np.abs(base_vocal.start - next_vocal.start) <= 12 or np.abs(base_vocal.end   - next_vocal.start) <= 12 or next_vocal.start >= base_vocal.start & next_vocal.end <= base_vocal.end):
            while next_vocal_is_close == True:
                new_vocal  = combine_vocals(base_vocal, next_vocal)
                base_vocal = new_vocal
                vocal_idx  = vocal_idx + 1
                next_vocal = self.list_of_vocals[vocal_idx+1]
                next_vocal_is_close = True if (np.abs(base_vocal.start - next_vocal.start) <= 12 or np.abs(base_vocal.end   - next_vocal.start) <= 12 or next_vocal.start >= base_vocal.start & next_vocal.end <= base_vocal.end) else False            

            new_list_of_vocals.append(base_vocal)

            #check if there is a next vocal, else there_are_vocals = False

        # self.vocals_in_recording = np.hstack(new_list_of_vocals)
        print("check if this is correct")

    def combine_list_of_list_of_vocals(self, list_of_list_of_vocals):
        new_list_of_vocals = []
        for list_of_vocals in list_of_list_of_vocals:
            try: 
                new_list_of_vocals.append(np.hstack(list_of_vocals.vocals_in_recording))
            except:
                 continue
        
        self.vocals_in_recording = np.hstack(new_list_of_vocals)
        self.number_of_vocals    = len(self.vocals_in_recording)
        self.vocals_combined     = True
        return 0