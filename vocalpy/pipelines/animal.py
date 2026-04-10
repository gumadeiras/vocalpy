# -*- coding: utf-8 -*-
"""VocalPy - Vocal analysis framework"""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

from abc import ABC, abstractmethod

import numpy as np


class Animal(ABC):
    """
    Animal class calls apropriate pipeline functions
    """

    def __init__(self, animal, params):
        self.params = params
        self._animal = animal

    def parse_chunk(self, chunk):
        """
        Convert the mixed-type chunk payload back into typed values.
        """
        (
            audio_path,
            output_dir,
            spectrogram_dir,
            mask_dir,
            sample_rate,
            bin_size,
            this_bin,
            start_range,
            end_range,
        ) = chunk

        return (
            audio_path,
            output_dir,
            spectrogram_dir,
            mask_dir,
            int(float(sample_rate)),
            int(float(bin_size)),
            int(float(this_bin)),
            int(float(start_range)),
            int(float(end_range)) if end_range is not None else None,
        )

    def get_centered_window_bounds(self, center, window_radius, max_width):
        """
        Clamp a centered window to valid bounds while preserving width when possible.
        """
        target_width = min(max_width, window_radius * 2)
        if max_width <= target_width:
            return 0, max_width

        start = center - window_radius
        end = center + window_radius

        if start < 0:
            end = min(max_width, end - start)
            start = 0

        if end > max_width:
            start = max(0, start - (end - max_width))
            end = max_width

        return int(start), int(end)

    def estimate_background_intensity(self, spectrogram, center, window_radius):
        """
        Estimate local background intensity around a candidate vocalization.
        """
        start, end = self.get_centered_window_bounds(center, window_radius, spectrogram.shape[1])
        return float(np.mean(spectrogram[:, start:end]))

    def has_minimum_contrast(self, signal_intensity, background_intensity):
        """
        Compare signal and background in dB space.
        """
        min_contrast_db = self.params.get("min_contrast_db", 3.0)
        return (signal_intensity - background_intensity) >= min_contrast_db

    def get_region_intensity_stats(self, prop):
        """
        Read intensity statistics across skimage regionprops API versions.
        """
        min_intensity = prop.intensity_min if hasattr(prop, "intensity_min") else prop.min_intensity
        max_intensity = prop.intensity_max if hasattr(prop, "intensity_max") else prop.max_intensity
        mean_intensity = prop.intensity_mean if hasattr(prop, "intensity_mean") else prop.mean_intensity
        return min_intensity, max_intensity, mean_intensity

    def get_duration_limits_in_frames(self, time_resolution_ms):
        """
        Resolve duration thresholds to spectrogram-frame counts.

        Legacy configs used `min_vocal_duration` / `max_vocal_duration` as
        frame counts despite the ambiguous name. Preserve that behavior while
        allowing explicit frame or millisecond keys going forward.
        """
        if "min_vocal_duration_ms" in self.params:
            min_duration = int(np.ceil(self.params["min_vocal_duration_ms"] / time_resolution_ms))
        elif "min_vocal_duration_frames" in self.params:
            min_duration = int(self.params["min_vocal_duration_frames"])
        else:
            min_duration = int(self.params["min_vocal_duration"])

        if "max_vocal_duration_ms" in self.params:
            max_duration = int(np.floor(self.params["max_vocal_duration_ms"] / time_resolution_ms))
        elif "max_vocal_duration_frames" in self.params:
            max_duration = int(self.params["max_vocal_duration_frames"])
        else:
            max_duration = int(self.params["max_vocal_duration"])

        return min_duration, max_duration

    @abstractmethod
    def identify_vocalizations(self, chunk):
        return NotImplemented

    @abstractmethod
    def classify_vocalizations(self, network_type, list_of_vocals, source=None):
        return NotImplemented

    @abstractmethod
    def check_if_vocals_are_close(self, first_vocal, second_vocal):
        return NotImplemented

    def connect_vocals(self, list_of_vocals):
        """
        Checks segmentation and combines segments that belong to the same vocalization. Uses predefined
        constraints. Can have different contrainsts for different animal pipelines

        Parameters
        ----------
        list_of_vocals : :class:`ListOfVocals`
            list of vocal candidates to be connected
        """
        vocals = [] if list_of_vocals.vocals_in_recording is None else list(list_of_vocals.vocals_in_recording)
        if not vocals:
            list_of_vocals.vocals_in_recording = []
            list_of_vocals.number_of_vocals = 0
            list_of_vocals.vocals_combined = True
            return 0

        merged_vocals = []
        current_vocal = vocals[0]

        for next_vocal in vocals[1:]:
            if self.check_if_vocals_are_close(current_vocal, next_vocal):
                current_vocal = self.combine_vocals(current_vocal, next_vocal)
            else:
                merged_vocals.append(current_vocal)
                current_vocal = next_vocal

        merged_vocals.append(current_vocal)
        list_of_vocals.vocals_in_recording = np.asarray(merged_vocals, dtype=object)
        list_of_vocals.number_of_vocals = len(merged_vocals)
        list_of_vocals.vocals_combined = True
        return 0

    def combine_vocals(self, first_vocal, second_vocal):
        """
        Combines two vocals

        Parameters
        ----------
        first_vocal : :class:`Vocal`
        second_vocal : :class:`Vocal`
            vocals to be combined
        """
        import numpy as np
        from vocalpy.modules.vocal import Vocal

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
        combined_vocal.duration = (combined_vocal.end - combined_vocal.start) * 1000
        combined_vocal.bandwidth = combined_vocal.max_freq - combined_vocal.min_freq

        return combined_vocal
