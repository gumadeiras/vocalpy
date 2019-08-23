# -*- coding: utf-8 -*-
'''VocalPy - A python version based on (VocalMat by Antonio Fonseca)'''

__author__    = 'Gustavo Madeira Santana'
__email__     = 'gustavo.santana@yale.edu'
__copyright__ = '2019 Dietrich Lab - Yale University School of Medicine'

import os

from   PIL import Image

class Vocal(object):
    '''
    vocalization object
    '''
    def __init__(self, bin_number    = None,
                       start         = None,
                       end           = None,
                       duration      = None,
                       interval      = None,
                       min_freq      = None,
                       max_freq      = None,
                       avg_freq      = None,
                       median_freq   = None,
                       bandwidth     = None,
                       min_intensity = None,
                       max_intensity = None,
                       avg_intensity = None,
                       bg_intensity  = None,
                       area          = None,
                       # points        = None,
                       centroid      = None,
                       orientation   = None,
                       spectrogram   = None,
                       mask          = None,
                       cnn_mask      = None,
                       label         = None,
                       probabilities = None):

        self._bin_number    = bin_number
        self._start         = start
        self._end           = end
        self._duration      = duration
        self._interval      = interval
        self._min_freq      = min_freq
        self._max_freq      = max_freq
        self._avg_freq      = avg_freq
        self._median_freq   = median_freq
        self._bandwidth     = bandwidth
        self._min_intensity = min_intensity
        self._max_intensity = max_intensity
        self._avg_intensity = avg_intensity
        self._bg_intensity  = bg_intensity
        self._area          = area
        # self._points        = points
        self._centroid      = centroid
        self._orientation   = orientation
        self._spectrogram   = spectrogram
        self._mask          = mask
        self._cnn_mask      = cnn_mask
        self._label         = label
        self._probabilities = probabilities

    def __str__(self):
        return "{}:\n bin_number: {} \n start: {} \n end: {} \n duration: {} \n interval: {} \n min_freq: {} \n max_freq: {} \n avg_freq: {} \n bandwidth: {} \n min_intensity: {} \n max_intensity: {} \n avg_intensity: {} \n bg_intensity: {} \n area: {} \n centroid: {} \n orientation: {} \n".format(self.__class__.__name__, self.bin_number, self.start, self.end, self.duration, self.interval, self.min_freq, self.max_freq, self.avg_freq, self.bandwidth, self.min_intensity, self.max_intensity, self.avg_intensity, self.bg_intensity, self.area, self.centroid, self.orientation)

    @property
    def bin_number(self):
        return self._bin_number

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def duration(self):
        return self._duration

    @property
    def interval(self):
        return self._interval

    @property
    def min_freq(self):
        return self._min_freq

    @property
    def max_freq(self):
        return self._max_freq

    @property
    def avg_freq(self):
        return self._avg_freq

    @property
    def median_freq(self):
        return self._median_freq

    @property
    def bandwidth(self):
        return self._bandwidth

    @property
    def min_intensity(self):
        return self._min_intensity

    @property
    def max_intensity(self):
        return self._max_intensity

    @property
    def avg_intensity(self):
        return self._avg_intensity

    @property
    def bg_intensity(self):
        return self._bg_intensity

    @property
    def area(self):
        return self._area

    # @property
    # def points(self):
    #     return self._points

    @property
    def centroid(self):
        return self._centroid

    @property
    def orientation(self):
        return self._orientation

    @property
    def spectrogram(self):
        return self._spectrogram

    @property
    def mask(self):
        return self._mask

    @property
    def cnn_mask(self):
        return self._cnn_mask

    @property
    def label(self):
        return self._label

    @property
    def probabilities(self):
        return self._probabilities

    @bin_number.setter
    def bin_number(self, new_bin_number):
        self._bin_number    = new_bin_number

    @start.setter
    def start(self, new_start):
        self._start         = new_start

    @end.setter
    def end(self, new_end):
        self._end           = new_end

    @duration.setter
    def duration(self, new_duration):
        self._duration      = new_duration

    @interval.setter
    def interval(self, new_interval):
        self._interval      = new_interval

    @min_freq.setter
    def min_freq(self, new_min_freq):
        self._min_freq      = new_min_freq

    @max_freq.setter
    def max_freq(self, new_max_freq):
        self._max_freq      = new_max_freq

    @avg_freq.setter
    def avg_freq(self, new_avg_freq):
        self._avg_freq      = new_avg_freq

    @median_freq.setter
    def median_freq(self, new_median_freq):
        self._median_freq   = new_median_freq

    @bandwidth.setter
    def bandwidth(self, new_bandwidth):
        self._bandwidth     = new_bandwidth

    @min_intensity.setter
    def min_intensity(self, new_min_intensity):
        self._min_intensity = new_min_intensity

    @max_intensity.setter
    def max_intensity(self, new_max_intensity):
        self._max_intensity = new_max_intensity

    @avg_intensity.setter
    def avg_intensity(self, new_avg_intensity):
        self._avg_intensity = new_avg_intensity

    @bg_intensity.setter
    def bg_intensity(self, new_bg_intensity):
        self._bg_intensity  = new_bg_intensity

    @area.setter
    def area(self, new_area):
        self._area          = new_area

    # @points.setter
    # def points(self, new_points):
        #     self._points  = new_points

    @centroid.setter
    def centroid(self, new_centroid):
        self._centroid      = new_centroid

    @orientation.setter
    def orientation(self, new_orientation):
        self._orientation   = new_orientation

    @spectrogram.setter
    def spectrogram(self, new_spectrogram):
        self._spectrogram   = new_spectrogram

    @mask.setter
    def mask(self, new_mask):
        self._mask          = new_mask

    @cnn_mask.setter
    def cnn_mask(self, new_cnn_mask):
        self._cnn_mask      = new_cnn_mask

    @label.setter
    def label(self, new_label):
        self._label         = new_label

    @probabilities.setter
    def probabilities(self, new_probabilities):
        self._probabilities = new_probabilities

    def save_spectrogram_as_image(self, path=None, filename='vocal'):
        if os.path.exists(path)==False:
            raise ValueError("path does not existe: {}".format(path))

        img = Image.fromarray(self.spectrogram)
        img = img.convert('L')
        img.save(os.path.join(path, 'spectrogram', str(self.bin_number) + '_' + filename + '.png'))

    def save_mask_as_image(self, path=None, filename='vocal'):
        if os.path.exists(path)==False:
            raise ValueError("path does not existe: {}".format(path))

        img = Image.fromarray(self.mask)
        img = img.convert('L')
        img.save(os.path.join(path, 'mask', str(self.bin_number) + '_' + filename + '.png'))

    def save_cnn_mask_as_image(self, path=None, filename='vocal'):
        if os.path.exists(path)==False:
            raise ValueError("path does not existe: {}".format(path))

        img = Image.fromarray(self.cnn_mask)
        img = img.convert('L')
        img.save(os.path.join(path, 'cnn_mask', str(self.bin_number) + '_' + filename + '.png'))