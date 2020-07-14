# -*- coding: utf-8 -*-
"""VocalPy - Vocal analysis framework"""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"


import numpy as np

from time import time
from math import ceil
from os import makedirs
from logging import getLogger
from joblib import Parallel, delayed
from os.path import join, split, splitext, exists

from vocalpy.modules.list_of_vocals import ListOfVocals
from vocalpy.utils.misc import check_pipeline_avalability, create_dataframe_from_list_of_vocals
from vocalpy.utils.io import (
    save_file,
    load_file,
    create_directory,
    remove_directory,
    read_audio_information,
    save_dataframe_as_csv,
)


class Recording(object):
    """
    Audio recording Object which contains auxiliary functions to process the recording
    Reads the audio file and breaks it down into segments of one minute to be processed
    in parallel. Stores metadata about the audio recording.

    Parameters
    ----------
    recording_path : str
        Path to audio file to open
    args : args
        Struct with arguments that customizes the execution with parameters such as:
        animal pipeline, bin size, frequency range, threads for parallelization, verbose output

    Returns
    -------
    recording : Object
        The recording object containing the read audio file and its metadata

    Examples
    --------
    >>> audio_path = 'path/to/audio'
    >>> args = p.parse_args()
    >>> recording = Recording(recording_path=audio_path, args=args)
    """

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
        self.number_of_samples = None
        self.recording_duration = None

        self.read_audio_metadata()

        low_freq, high_freq = [int(f) for f in args.frequency.split(",")]
        self.low_frequency_cutoff = low_freq
        self.high_frequency_cutoff = high_freq
        self.bin_size = self.args.bin_size if (self.args.bin_size < self.recording_duration) else self.recording_duration
        self.bins = ceil(self.recording_duration / self.bin_size)
        self.chunks = self.create_chunks()
        self._group_name = "not set"
        self._list_of_vocals = None
        self._has_list_of_vocals = None
        self._animal_name = args.animal
        self._animal = self.create_animal_pipeline()

    def __str__(self):
        return f"{self.__class__.__name__}:\n duration: {self.recording_duration} \n sampling rate: {self.sample_rate}"

    @property
    def has_list_of_vocals(self):
        """
        Checks if recording has a list of vocals

        Returns
        ------
        _has_list_of_vocals : Bool
        """
        return self._has_list_of_vocals

    @has_list_of_vocals.setter
    def has_list_of_vocals(self, new_has_list_of_vocals):
        self._has_list_of_vocals = new_has_list_of_vocals

    @property
    def list_of_vocals(self):
        """
        Returns the list of vocals for this recording

        Returns
        ------
        _list_of_vocals : :class:`ListOfVocals`
        """
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

    def save_recording_object(self, path, filename="recording"):
        save_file(self, filename, path)

    def create_paths(self, recording_path):
        """
        Creates directory structure for output files
        Creates a `spectrogram` and `mask` directories inside the `audio_output` directory

        Parameters
        ----------
        recording_path : str
        """
        basepath, filename = split(recording_path)
        self.recording_dir = basepath
        self.recording_name = filename
        filename = splitext(filename)[0]
        self.output_dir = join(self.recording_dir, filename + "_outputs")
        self.spectrogram_dir = join(self.output_dir, "spectrogram")
        self.mask_dir = join(self.output_dir, "mask")

        if not exists(self.output_dir):
            makedirs(self.output_dir, exist_ok=True)

        if not exists(self.spectrogram_dir):
            makedirs(self.spectrogram_dir, exist_ok=True)

        if not exists(self.mask_dir):
            makedirs(self.mask_dir, exist_ok=True)

    def read_audio_metadata(self):
        """
        Reads audio metadata. Stores information in the Recording Object
        """
        metadata = read_audio_information(self.recording_path)
        self.sample_rate = metadata.samplerate
        self.recording_duration = metadata.duration
        self.number_of_samples = self.recording_duration * self.sample_rate

    def create_chunks(self, overlap=0.15):
        """
        Segments audio for parallel or sequential processing

        Parameters
        ----------
            overlap : float
                segments overlap (in seconds)
        """
        chunks = []
        baseline_chunk = [
            self.recording_path,
            self.output_dir,
            self.spectrogram_dir,
            self.mask_dir,
            self.sample_rate,
            self.bin_size,
            self.low_frequency_cutoff,
            self.high_frequency_cutoff,
            self.args,
        ]
        for this_bin in range(1, self.bins + 1):
            # -- first bin, remove first 0.5 second of recording (usually noisy)
            if this_bin == 1:
                start_range = ceil(0.5 * self.sample_rate)
                end_range = ceil((self.bin_size * self.sample_rate) + (overlap * self.sample_rate))
                end_range = end_range if end_range < self.number_of_samples else self.number_of_samples
                this_chunk = baseline_chunk.copy()
                this_chunk.append((this_bin, start_range, end_range,))
                chunks.append(np.hstack(this_chunk))

            elif this_bin == self.bins:  # -- last bin
                start_range = ceil((this_bin - 1) * self.bin_size * self.sample_rate)
                end_range = self.recording_duration * self.sample_rate
                if end_range - start_range < self.sample_rate:
                    continue  # less than 1s
                # -- None reads until the end of the audio
                # end_range = None
                this_chunk = baseline_chunk.copy()
                this_chunk.append((this_bin, start_range, end_range,))
                chunks.append(np.hstack(this_chunk))

            else:  # -- all other bins
                start_range = ceil((this_bin - 1) * self.bin_size * self.sample_rate)
                end_range = ceil((this_bin * self.bin_size * self.sample_rate) + (overlap * self.sample_rate))
                end_range = end_range if end_range < self.number_of_samples else self.number_of_samples
                if end_range - start_range < self.sample_rate:
                    continue  # less than 1s
                this_chunk = baseline_chunk.copy()
                this_chunk.append((this_bin, start_range, end_range,))
                chunks.append(np.hstack(this_chunk))

        return chunks

    def create_animal_pipeline(self):
        """
        Creates an instance of :class:`Animal` that implements the pipeline for the animal selected by the user

        Parameters
        ----------
        animal : str
            animal pipeline selected by the user

        Returns
        -------
        class instance : :class:`Animal`
            animal pipeline selected by the user
        """
        import importlib

        has_identifier, has_classifier = check_pipeline_avalability(self._animal_name)

        if has_identifier:
            AnimalClass = getattr(importlib.import_module("vocalpy.pipelines." + self._animal_name), self._animal_name.title())
        else:
            print("implement error, animal pipeline not available")

        return AnimalClass(self._animal_name.lower(), has_identifier, has_classifier)

    def identify_vocalizations(self):
        """
        Process recording by calling appropriate animal pipeline functions
        """

        logger = getLogger()
        timeAParallel = time()

        # -- distribute Recording chunks to available cores
        # -- process each chunk and find candidate vocalizations
        results = Parallel(n_jobs=self.args.threads, require="sharedmem")(
            delayed(self._animal.identify_vocalizations)(chunk=i) for i in self.chunks
        )

        # -- create list of vocals found in the recording
        logger.info("combining list of vocals from each bin")
        timeAcombining = time()
        list_of_vocals = ListOfVocals()
        list_of_vocals.combine_list_of_list_of_vocals(list_of_list_of_vocals=results)

        # -- only needs to fix first vocal in each segment
        list_of_vocals.update_intervals()
        self._has_list_of_vocals = True
        self.list_of_vocals = list_of_vocals
        logger.info(f"done combining ({time() - timeAcombining:.0f}s)")
        logger.info(list_of_vocals)

        self.identify_vocalizations_finished()
        logger.info(
            f"recording parallel processing ({(time() - timeAParallel) // 60:.0f}m {(time() - timeAParallel) % 60:.0f}s)"
        )

    def identify_vocalizations_finished(self):
        """
        Recording has already been processed -> clear segments
        """
        self.chunks = None
        return 0

    def classify_vocalizations(self):
        """
        Classify identified vocalizations using the appropiate animal pipeline
        """
        logger = getLogger()

        if self._animal._has_classifier is True:
            # -- save spectrograms (used in noise classifier)
            logger.info("saving spectrograms of candidate vocalizations")
            timeAsaving = time()
            self.save_spectrograms(path=self.output_dir)
            logger.info(f"done saving ({time() - timeAsaving:.0f}s)")

            # -- classify candidate vocalizations as Vocal or Noise; remove Noise
            logger.info("classifying candidate vocalizations as vocal or noise")
            timeAclassification = time()
            predictions, classes = self._animal.classify_vocalizations(
                network_type="noise", list_of_vocals=self.list_of_vocals, path_to_spectrograms=self.spectrogram_dir
            )
            logger.info("removing candidates classified as noise")
            self.remove_vocals_classified_as_noise_from_list_of_vocals(predictions)
            self.save_spectrograms_and_masks(path=self.output_dir)
            logger.info(f"done classifying and removing ({time() - timeAclassification:.0f}s)")
            logger.info(self._list_of_vocals)

            # -- classify vocalizations into vocal types
            logger.info("classifying vocalizations")
            timeAclassification = time()
            predictions, classes = self._animal.classify_vocalizations(
                network_type="class", list_of_vocals=self.list_of_vocals, path_to_spectrograms=self.spectrogram_dir
            )
            logger.info("adding classification to vocals")
            self.update_vocals_with_class_classification(predictions, classes)
            logger.info(f"done classifying and updating vocals ({time() - timeAclassification:.0f}s)")
        else:
            logger.info(f"no classifier available for animal type: {self._animal._animal}")

    def load_list_of_vocals(self):
        """
        Loads :class:`ListOfVocals`  from a python object file
        """
        return load_file("list_of_vocals", self.output_dir)

    def save_recording_data_to_csv(self, list_of_vocals=None, path=None):
        """
        Saves recording metadata to a CSV file. The file will contain information
        regarding each vocalization identified in the recording.

        Parameters
        ----------
        list_of_vocals : :class:`ListOfVocals`, optional
        path : str, optional
        """
        # -- save metadata to a csv file
        if list_of_vocals is None and self._has_list_of_vocals is not True:
            return -1
        list_of_vocals = list_of_vocals if list_of_vocals is not None else self._list_of_vocals

        if path is None:
            path = self.output_dir

        if list_of_vocals.intervals_fixed is False:
            list_of_vocals.update_intervals()

        recording_df = create_dataframe_from_list_of_vocals(self.list_of_vocals)

        save_dataframe_as_csv(dataframe=recording_df, path=self.output_dir, filename=self.recording_name)

    def save_spectrograms(self, list_of_vocals=None, path=None):
        """
        Saves the spectrogram images to the output directory for this Recording Object

        Parameters
        ----------
        list_of_vocals : :class:`ListOfVocals`, optional
        path : str, optional
        """
        if self._has_list_of_vocals is not True and list_of_vocals is None:
            return -1
        list_of_vocals = list_of_vocals if list_of_vocals is not None else self._list_of_vocals
        path = path if path is not None else self.output_dir
        remove_directory(join(path, "spectrogram"))
        create_directory(join(path, "spectrogram"))
        list_of_vocals.save_spectrograms(output_dir=path)
        return 0

    def save_validation_images(self, list_of_vocals=None, path=None):
        """
        Saves the spectrogram overlaid with the segmentation images to the output directory for this Recording Object

        Parameters
        ----------
        list_of_vocals : :class:`ListOfVocals`, optional
        path : str, optional
        """
        if self._has_list_of_vocals is not True and list_of_vocals is None:
            return -1
        list_of_vocals = list_of_vocals if list_of_vocals is not None else self._list_of_vocals
        path = path if path is not None else self.output_dir
        remove_directory(join(path, "spectrogram_validation"))
        create_directory(join(path, "spectrogram_validation"))
        list_of_vocals.save_validation_images(output_dir=path)
        return 0

    def save_spectrograms_and_masks(self, list_of_vocals=None, path=None):
        """
        Saves the spectrograms and segmentation images to the output directory for this Recording Object

        Parameters
        ----------
        list_of_vocals : :class:`ListOfVocals`, optional
        path : str, optional
        """
        if self._has_list_of_vocals is not True and list_of_vocals is None:
            return -1
        list_of_vocals = list_of_vocals if list_of_vocals is not None else self._list_of_vocals
        path = path if path is not None else self.output_dir
        remove_directory(join(path, "spectrogram"))
        create_directory(join(path, "spectrogram"))
        list_of_vocals.save_spectrograms(output_dir=path)
        remove_directory(join(path, "mask"))
        create_directory(join(path, "mask"))
        list_of_vocals.save_masks(output_dir=path)
        return 0

    def remove_spectrograms_and_masks_from_object(self, list_of_vocals=None):
        """
        Removes the directories containing the spectrograms and segmentation images in
        the Recording Object output directory

        Parameters
        ----------
        list_of_vocals : :class:`ListOfVocals`, optional
        """
        if self._has_list_of_vocals is not True and list_of_vocals is None:
            return -1
        list_of_vocals = list_of_vocals if list_of_vocals is not None else self._list_of_vocals
        list_of_vocals.remove_spectrograms()
        list_of_vocals.remove_masks()
        return 0

    def create_dataset(self, list_of_vocals=None):
        """
        Creates an image dataset to be used by the Neural Networks
        """
        # ToDo
        # create from list of vocals or save spectrograms
        # create filename list etc and create from folder path
        if self.has_list_of_vocals is not True and list_of_vocals is None:
            return -1
        list_of_vocals = list_of_vocals if list_of_vocals is not None else self.load_list_of_vocals()

        print("create_dataset not implemented")
        return NotImplemented

    def remove_vocals_classified_as_noise_from_list_of_vocals(self, predictions):
        """
        Removes vocals that were classified as noise from the :class:`ListOfVocals`

        Parameters
        ----------
        predictions : List[float]
            Neural Network classification predictions for the :class:`ListOfVocals`
        """

        # -- if list of vocals is empty, there are no predictions
        if isinstance(predictions, int) and predictions == -1:
            return predictions

        # -- remove vocals that were classifier as noise
        self._list_of_vocals.remove_vocals_classified_as_noise(predictions)
        # # -- update inter-vocal intervals after removing noise
        # self._list_of_vocals.update_intervals()
        return 0

    def update_vocals_with_class_classification(self, predictions, classes):
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

        # -- if list of vocals is empty, there are no predictions
        if isinstance(predictions, int) and predictions == -1:
            return predictions

        # -- make sure number of predictions is the same as number of vocals
        try:
            assert self._list_of_vocals.number_of_vocals == predictions.shape[0]
        except AssertionError:
            print(
                f"[error] number of vocals: {self._list_of_vocals.number_of_vocals}; \
                number of predictions: {predictions.shape[0]}"
            )
            exit()

        # -- add probability distribution for each vocal, top1 and top2 classes
        self._list_of_vocals.add_classification_to_vocals(predictions, classes)
        return 0

    def save_outputs(self, validation_flag):
        logger = getLogger()

        logger.info("saving recording object, vocalizations, and csv file")
        timeAsaving = time()
        if validation_flag is True:
            self.save_validation_images(path=self.output_dir)
        # self.save_recording_object(path=self.output_dir)
        self.remove_spectrograms_and_masks_from_object()
        self.save_recording_object(filename="recording_without_spectrograms", path=self.output_dir)
        self.save_recording_data_to_csv(path=self.output_dir)
        logger.info(f"done saving ({time() - timeAsaving:.0f}s)")
