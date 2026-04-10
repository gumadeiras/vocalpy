# -*- coding: utf-8 -*-
"""VocalPy - Vocal analysis framework"""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import cv2

import numpy as np

from time import time
from math import ceil
from scipy import ndimage
from logging import getLogger
from skimage import exposure, measure

from vocalpy.utils.io import read_audio
from vocalpy.pipelines.animal import Animal
from vocalpy.nn.classifier import VocalClassifier
from vocalpy.nn.datasets import create_array_from_list_of_vocals
from vocalpy.utils.signal_processing import compute_spectrogram
from vocalpy.utils.image_processing import normalize, contrast_adjustment, bradley_roth


class Mouse(Animal):
    def identify_vocalizations(self, chunk):
        return self.identifier(chunk)

    def classify_vocalizations(self, network_type, list_of_vocals, source=None):
        source = create_array_from_list_of_vocals(list_of_vocals) if source is None else source
        Classifier = VocalClassifier(network_type=network_type, source=source)
        predictions = Classifier.classify_list_of_vocals(list_of_vocals)
        return predictions, Classifier.classes

    def check_if_vocals_are_close(self, first_vocal, second_vocal):
        # -- conditions to check:
        # -- 1) next vocal starts within 12ms from base vocal start time
        # -- 2) next vocal starts within 12ms from base vocal end time
        # -- 3) next vocal starts within base vocal start/end (harmonic)
        max_interval = 0.011  # 12ms - 1ms error because morph ops increase area
        condition_1 = np.abs(first_vocal.end - second_vocal.start) < max_interval
        condition_2 = np.abs(first_vocal.start - second_vocal.start) < max_interval
        condition_3 = (second_vocal.start >= first_vocal.start) and (second_vocal.start <= first_vocal.end)

        return True if (condition_1 or condition_2 or condition_3) else False

    def identifier(self, chunk):
        from vocalpy.modules.vocal import Vocal
        from vocalpy.modules.list_of_vocals import ListOfVocals

        logger = getLogger()

        timeBinA = time()

        # -- unwrap chunk
        (
            audio_path,
            _output_dir,
            _spectrogram_dir,
            _mask_dir,
            sample_rate,
            bin_size,
            this_bin,
            start_range,
            end_range,
        ) = self.parse_chunk(chunk)

        timeAudioRead = time()
        logger.info(f"[bin {this_bin}]: reading audio;")
        sample_range, __ = read_audio(audio_path, start=start_range, stop=end_range)
        logger.info(f"[bin {this_bin}]: read audio runtime: {time() - timeAudioRead:.2f}s;")

        timeASpectrogram = time()
        if end_range is None:
            logger.info(
                f"[bin {this_bin}]: computing spectrogram; \
                time range: {sample_range.shape[0] / sample_rate:.2f}s; \
                audio range: {start_range / sample_rate:.2f}s-end of audio"
            )
        else:
            logger.info(
                f"[bin {this_bin}]: computing spectrogram; \
                time range: {sample_range.shape[0] / sample_rate:.2f}s; \
                audio range: {start_range / sample_rate:.2f}-{end_range / sample_rate:.2f}s"
            )

        # -- compute spectrogram
        f, t, Pxx = compute_spectrogram(
            samples=sample_range,
            fs=sample_rate,
            window_type=self.params["window_type"],
            window_size=self.params["window_size"],
            noverlap=self.params["noverlap"],
            nfft=self.params["nfft"],
            lower_frequency_cutoff=self.params["lower_frequency_cutoff"],
            higher_frequency_cutoff=self.params["higher_frequency_cutoff"],
        )

        time_res = (sample_range.shape[0] / sample_rate) / t.shape[0]
        freq_res = (np.max(f) - self.params["lower_frequency_cutoff"]) / f.shape[0]
        logger.info(f"[bin {this_bin}]: spectrogram runtime: {time() - timeASpectrogram:.2f}s")
        logger.info(f"[bin {this_bin}]: time resolution: {time_res * 1000:.2f}ms")
        logger.info(f"[bin {this_bin}]: freq resolution: {freq_res:.2f}Hz")

        # -- rescale data to (0,1)
        B = normalize(data=Pxx)

        # -- saturate extreme values
        B = contrast_adjustment(data=B, lower_percentile=1, upper_percentile=99)

        # -- binarize image
        B = bradley_roth(B, t=20)

        # -- median filter
        B = ndimage.median_filter(B, size=(3, 3))

        # -- kernels for morphological operations
        kernel_rect = np.ones((4, 2), np.uint8)
        kernel_line1 = np.ones((4, 1), np.uint8)
        kernel_line2 = np.ones((5, 1), np.uint8)

        # -- morphological operations
        B = cv2.erode(B, kernel_line1, iterations=1)
        B = cv2.dilate(B, kernel_rect, iterations=1)
        B = cv2.dilate(B, kernel_line2, iterations=1)
        B = cv2.erode(B, kernel_line1, iterations=2)
        timeAConnectedComponents = time()
        connectivity = 4
        num_cc, output, stats, centroids = cv2.connectedComponentsWithStats(B, connectivity, cv2.CV_32S)
        del B

        # -- remove background stats
        num_cc = num_cc - 1
        areas = stats[1:, 4]

        # -- filtered connected components placeholder
        grain = np.zeros((output.shape))

        # -- threshold connected components by minimum area
        min_area = 20
        for i in range(0, num_cc):
            if areas[i] >= min_area:
                grain[output == i + 1] = 255

        logger.info(f"[bin {this_bin}]: connected components runtime: {time() - timeAConnectedComponents:.2f}s")

        # -- one more opening to make sure
        # -- segmentation covers *at least* the real area
        grain = grain.astype(np.uint8)
        # kernel_cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        kernel_line3 = np.ones((1, 3), dtype=np.uint8)
        grain = cv2.dilate(grain, kernel_line3, iterations=1)

        # -- get connected components stats
        timeARegionProps = time()
        labels = measure.label(grain, background=0)

        props = measure.regionprops(labels, intensity_image=Pxx, cache=True)

        # -- sort segments by time
        props = sorted(props, key=lambda p: np.min(p.coords[:, 1]), reverse=False)

        logger.info(f"[bin {this_bin}]: region props runtime: {time() - timeARegionProps:.2f}s")
        del labels

        timeAVocal = time()
        vocal_id = 0
        vocal_list = []

        index = 0
        end = 0
        for prop in props:
            start = np.min(prop.coords[:, 1])
            if index > 0:
                interval = np.abs(end - start)
            else:
                interval = 0
                index = index + 1

            end = np.max(prop.coords[:, 1])
            duration = end - start

            if (duration < self.params["min_vocal_duration"]) | (duration > self.params["max_vocal_duration"]):
                continue

            # -- get spectrogram and mask around
            # -- each vocalization to compute intensity
            spectro_range = 200  # 2*200 * 0.51 = 205ms
            centroid_time = ceil(prop.centroid[1])

            # -- edge conditions:
            # -- spectro_range goes over the spectrom vector limit (for this bin)
            # -- left edge: -200 is before vector start index
            # -- right edge: +200 is after vector end index
            bg_intensity = self.estimate_background_intensity(Pxx, centroid_time, spectro_range)

            # -- spectrogram intensities are in dB, so contrast must also be
            # -- compared in dB instead of with a linear-space ratio
            if not self.has_minimum_contrast(prop.mean_intensity, bg_intensity):
                continue

            if this_bin == 1:
                # first 0.5 were removed from recording as they are noisy
                # make this better
                start_time = (start * time_res) + ((this_bin - 1) * bin_size) + 0.5
                end_time = (end * time_res) + ((this_bin - 1) * bin_size) + 0.5
            else:
                start_time = (start * time_res) + ((this_bin - 1) * bin_size)
                end_time = (end * time_res) + ((this_bin - 1) * bin_size)

            min_freq_coord = np.min(prop.coords[:, 0])
            max_freq_coord = np.max(prop.coords[:, 0])
            min_freq = (min_freq_coord * freq_res) + self.params["lower_frequency_cutoff"]
            max_freq = (max_freq_coord * freq_res) + self.params["lower_frequency_cutoff"]
            avg_freq = (np.mean(prop.coords[:, 0]) * freq_res) + self.params["lower_frequency_cutoff"]
            bandwidth = max_freq - min_freq

            new_vocal = Vocal(
                bin_number=this_bin,
                start=start_time,
                end=end_time,
                start_coord=start,
                end_coord=end,
                duration=duration * time_res * 1000,
                interval=interval * time_res,
                min_freq=min_freq,
                max_freq=max_freq,
                min_freq_coord=min_freq_coord,
                max_freq_coord=max_freq_coord,
                avg_freq=avg_freq,
                bandwidth=bandwidth,
                min_intensity=prop.min_intensity,
                max_intensity=prop.max_intensity,
                avg_intensity=prop.mean_intensity,
                bg_intensity=bg_intensity,
                area=prop.area,
                centroid=np.rint(prop.centroid).astype(int),
                coords=prop.coords,
            )

            vocal_list.append(new_vocal)
            vocal_id = vocal_id + 1

        del props

        # -- if list is not empty, create a list of vocals
        if vocal_list:
            vocal_list = ListOfVocals(vocals_in_recording=np.asarray(vocal_list))
            timeAConnectVocals = time()
            self.connect_vocals(vocal_list)
            vocal_list.update_centroids()

            # -- 206*2 ~ 210ms @ 0.51ms resolution
            spectrogram_range = 206
            vocal_list.update_coords(spectrogram_range)

            # -- rescale pixel values to save spectrograms in 8bits
            dtype = np.uint8
            Pxx = exposure.rescale_intensity(Pxx, in_range="image", out_range=dtype)
            vocal_list.add_spectrograms_to_vocals(
                full_spectrogram=np.flipud(Pxx), full_mask=np.flipud(grain), spec_range=spectrogram_range,
            )

            logger.info(f"[bin {this_bin}]: connecting vocals runtime: {time() - timeAConnectVocals:.2f}s")

        logger.info(f"[bin {this_bin}]: list of vocals runtime: {time() - timeAVocal:.2f}s")
        logger.info(f"[bin {this_bin}]: raw number of vocals: {vocal_id}")
        logger.info(f"[bin {this_bin}]: {vocal_list}")
        logger.info(f"[bin {this_bin}]: bin runtime: {time() - timeBinA:.2f}s")

        return vocal_list
