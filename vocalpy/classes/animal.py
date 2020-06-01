# -*- coding: utf-8 -*-
"""VocalPy - Vocal analysis framework"""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

from abc import ABC, abstractmethod


class Animal(ABC):
    """
    Animal class calls apropriate pipeline functions
    """

    def __init__(self, has_identifier, has_classifier):
        self._has_identifier = has_identifier
        self._has_classifier = has_classifier

    @abstractmethod
    def identify_vocalizations(self, chunk):
        pass

    @abstractmethod
    def classify_vocalizations(self, network_type, list_of_vocals, path_to_spectrograms):
        pass

    @abstractmethod
    def check_if_vocals_are_close(self, first_vocal, second_vocal):
        pass

    def connect_vocals(self, list_of_vocals):
        """
            Checks segmentation and combines segments that belong to the same vocalization. Uses predefined
            constraints. Can have different contrainsts for different animal pipelines

            Parameters
            ----------
            animal : str
                animal pipeline (constraints) to use
            """
        # -- combine segmentation blobs that are close
        # -- consider them as one vocal
        import numpy as np

        new_list_of_vocals = []

        idx = 0
        there_are_vocals = True
        while there_are_vocals is True:
            # merge this blobs with blobs that are close
            try:
                base_vocal = list_of_vocals.vocals_in_recording[idx]
                new_vocal = base_vocal
                next_vocal = list_of_vocals.vocals_in_recording[idx + 1]
            except:
                # -- there are no more vocals to connect
                new_list_of_vocals.append(new_vocal)
                there_are_vocals = False
                break

            next_vocal_is_close = self.check_if_vocals_are_close(base_vocal, next_vocal)

            # -- get next vocal (idx)
            idx = idx + 1
            while next_vocal_is_close is True:
                """
                    while there are blobs close by, continue combining them
                    """
                new_vocal = self.combine_vocals(new_vocal, next_vocal)
                try:
                    next_vocal = list_of_vocals.vocals_in_recording[idx + 1]
                    next_vocal_is_close = self.check_if_vocals_are_close(new_vocal, next_vocal)
                except:
                    next_vocal_is_close = False
                idx = idx + 1

            # when last vocal is combined, sometimes gets duplicated; check why
            new_list_of_vocals.append(new_vocal)

        try:
            list_of_vocals.vocals_in_recording = np.hstack(new_list_of_vocals)
        except:
            list_of_vocals.vocals_in_recording = new_list_of_vocals

        list_of_vocals.number_of_vocals = len(list_of_vocals.vocals_in_recording)
        list_of_vocals.vocals_combined = True

    def combine_vocals(self, first_vocal, second_vocal):
        import numpy as np
        from vocalpy.classes.vocal import Vocal

        """
        Combines two vocals

        Parameters
        ----------
        first_vocal : :class:`Vocal`
        second_vocal : :class:`Vocal`
            vocals to be combined
        """
        # -- combines two vocals
        start_difference = first_vocal.start - second_vocal.start
        end_difference = first_vocal.end - second_vocal.end

        # -- new centroid will be updated from vocal start/end and min/max frequency
        combined_vocal = Vocal(
            bin_number=first_vocal.bin_number
            if (first_vocal.bin_number < second_vocal.bin_number)
            else second_vocal.bin_number,
            start=first_vocal.start if (start_difference < 0) else second_vocal.start,
            start_coord=first_vocal.start_coord if (start_difference < 0) else second_vocal.start_coord,
            end=first_vocal.end if (end_difference > 0) else second_vocal.end,
            end_coord=first_vocal.end_coord if (end_difference > 0) else second_vocal.end_coord,
            interval=-1,  # -- updated after noise candidates are removed
            min_freq=first_vocal.min_freq if (first_vocal.min_freq < second_vocal.min_freq) else second_vocal.min_freq,
            max_freq=first_vocal.max_freq if (first_vocal.max_freq > second_vocal.max_freq) else second_vocal.max_freq,
            min_freq_coord=first_vocal.min_freq_coord
            if (first_vocal.min_freq_coord < second_vocal.min_freq_coord)
            else second_vocal.min_freq_coord,
            max_freq_coord=first_vocal.max_freq_coord
            if (first_vocal.max_freq_coord > second_vocal.max_freq_coord)
            else second_vocal.max_freq_coord,
            avg_freq=np.mean((first_vocal.avg_freq, second_vocal.avg_freq)),
            min_intensity=first_vocal.min_intensity
            if (first_vocal.min_intensity < second_vocal.min_intensity)
            else second_vocal.min_intensity,
            max_intensity=first_vocal.max_intensity
            if (first_vocal.max_intensity < second_vocal.max_intensity)
            else second_vocal.max_intensity,
            avg_intensity=np.mean((first_vocal.avg_intensity, second_vocal.avg_intensity)),
            bg_intensity=np.mean((first_vocal.bg_intensity, second_vocal.bg_intensity)),
            area=first_vocal.area + second_vocal.area,
            centroid=first_vocal.centroid,
            coords=np.vstack((first_vocal.coords, second_vocal.coords)),
        )
        combined_vocal.duration = combined_vocal.end - combined_vocal.start
        combined_vocal.bandwidth = combined_vocal.max_freq - combined_vocal.min_freq

        return combined_vocal
