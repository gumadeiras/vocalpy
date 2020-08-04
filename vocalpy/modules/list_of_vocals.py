# -*- coding: utf-8 -*-
"""VocalPy - Vocal analysis framework"""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import torch

import numpy as np


class ListOfVocals(object):
    """
    List of vocalizations identified in the recording. Each vocal if an instance of :class:`Vocals`

    Parameters
    ----------
    vocals_in_recording : List[:class:`Vocal`]
        list of vocals identified in the recording
    """

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
        self.coords_fixed = False

    def __str__(self):
        return f"{self.__class__.__name__}:\n \
            number_of_vocals: {self.number_of_vocals}\n \
            vocals_combined: {self.vocals_combined}\n \
            intervals_fixed: {self.intervals_fixed}\n \
            centroid_spectro_fixed: {self.centroid_spectro_fixed}"

    def save_list_of_vocals_object(self, path):
        """
        Saves a :class:`ListOfVocals` Object to file

        Parameters
        ----------
        path : str
            path to save the object
        """
        from utils import write_pickle_file

        write_pickle_file(self, "list_of_vocals", path)

    def update_intervals(self):
        """
        Updates the interval (silence) between vocals. Usually used after combining or removing vocals
        """
        # -- go through vocals and update inter vocal times
        self.vocals_in_recording[0].interval = 0

        for idx in range(1, len(self.vocals_in_recording), 1):
            self.vocals_in_recording[idx].interval = np.abs(
                (self.vocals_in_recording[idx - 1].end - self.vocals_in_recording[idx].start)
            )

        self.intervals_fixed = True
        return 0

    def update_centroids(self):
        """
        Updates centroid coordinates for each vocal. Usually called after combining vocals. Absolute
        centroid from vocal coordinates start/end, and min/max frequency
        """
        for vocal in self.vocals_in_recording:
            cx = vocal.start_coord + ((vocal.end_coord - vocal.start_coord) // 2)
            cy = vocal.min_freq_coord + ((vocal.max_freq_coord - vocal.min_freq_coord) // 2)
            vocal.centroid = np.rint([cy, cx]).astype(np.int)

        self.centroid_spectro_fixed = True
        return 0

    def update_coords(self, spec_range=200):
        """
        Updates coordinates for each vocal after cropping area around a vocal. Absolute
        coords from vocal coordinates start/end, and min/max frequency

        Parameters
        ----------
        spec_range : int, optional
            range before/after vocal used to crop and generate spectrograms
        """
        for vocal in self.vocals_in_recording:
            col_values = vocal.coords[:, 1]
            # make column values zero-centered by subtracting the mean
            col_values = col_values - np.int(np.mean(col_values))
            # col values will be centered in the spectrogram
            col_values = col_values + spec_range
            vocal.coords[:, 1] = col_values

        self.coords_fixed = True
        return 0

    def combine_list_of_list_of_vocals(self, list_of_list_of_vocals):
        """
        Combines a list of :class:`ListOfVocals` into one :class:`ListOfVocals`. Usually called
        to combine several lists of parallel processing of a recording

        Parameters
        ----------
        list_of_list_of_vocals : List[:class:`ListOfVocals`]
            list of :class:`ListOfVocals`
        """
        new_list_of_vocals = []
        for list_of_vocals in list_of_list_of_vocals:
            try:
                new_list_of_vocals.append(np.hstack(list_of_vocals.vocals_in_recording))
            except:
                # empty list
                continue

        if new_list_of_vocals:
            self.vocals_in_recording = np.hstack(new_list_of_vocals)
            self.number_of_vocals = len(self.vocals_in_recording)
            self.vocals_combined = True
            self.centroid_spectro_fixed = True
            return 0

        print("recording has no vocals")
        exit()

    def add_spectrograms_to_vocals(self, full_spectrogram, full_mask, spec_range=200):
        """
        Stores the spectrogram in each :class:`Vocal` class object in the :class:`ListOfVocals`

        Parameters
        ----------
        full_spectrogram : ndarray
            complete spectrogram ranging the recording segment
        full_mask : ndarray
            complete segmentation mask ranging the recording segment
        spec_range : int, optional
            range to crop the spectrogram/segmentation around the vocal (+-200)
        """
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
        """
        Saves the spectrogram image to the output directory

        Parameters
        ----------
        output_dir : str, optional
            path to output directory to save the files
        """
        for filename, vocal in enumerate(self.vocals_in_recording, start=1):
            vocal.save_spectrogram_as_image(path=output_dir, filename=str(filename))
        return 0

    def save_validation_images(self, output_dir=None):
        """
        Saves the spectrogram overlaidd with the segmentation image to the output directory

        Parameters
        ----------
        output_dir : str, optional
            path to output directory to save the files
        """
        for filename, vocal in enumerate(self.vocals_in_recording, start=1):
            vocal.save_spectrogram_with_segmentation_as_image(path=output_dir, filename=str(filename))
        return 0

    def save_masks(self, output_dir=None):
        """
        Saves the segmentation mask image to the output directory

        Parameters
        ----------
        output_dir : str, optional
            path to output directory to save the files
        """
        for filename, vocal in enumerate(self.vocals_in_recording, start=1):
            vocal.save_mask_as_image(path=output_dir, filename=str(filename))
        return 0

    def remove_spectrograms(self):
        """
        Removes the spectrogram data from each :class:`Vocal` in the :class:`ListOfVocals`
        """
        for vocal in self.vocals_in_recording:
            vocal.spectrogram = None
        return 0

    def remove_masks(self):
        """
        Removes the segmentation data from each :class:`Vocal` in the :class:`ListOfVocals`
        """
        for vocal in self.vocals_in_recording:
            vocal.mask = None
        return 0

    def remove_vocals_classified_as_noise(self, predictions):
        """
        Removes vocals that were classified as noise from the :class:`ListOfVocals` and
        updates the number of vocals

        Parameters
        ----------
        predictions : List[float]
            Neural Network classification predictions for the :class:`ListOfVocals`
        """
        self.vocals_in_recording = self.vocals_in_recording[predictions]
        self.number_of_vocals = len(self.vocals_in_recording)
        if self.number_of_vocals > 0:
            self.update_intervals()
        return 0

    def add_classification_to_vocals(self, predictions, classes):
        """
        Updates :class:`ListOfVocals` with the class and probability distribution obtained
        using the Neural Network

        Parameters
        ----------
        predictions : List[float]
            Neural Network classification predictions for the :class:`ListOfVocals`
        classes : List[str]
            Labels used for classifying vocalizations
        """
        for idx, vocal in enumerate(self.vocals_in_recording):
            vocal.probabilities = predictions[idx]

            # -- convert numpy to torch, to use TopK function
            preds = torch.tensor(predictions[idx])
            # -- get top2 probabilities and their class names
            top1, top2 = preds.topk(2).indices.numpy()
            vocal.top1 = classes[top1]
            vocal.top2 = classes[top2]

        return 0
