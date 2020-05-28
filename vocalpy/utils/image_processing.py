# -*- coding: utf-8 -*-
"""VocalPy - Vocal analysis framework"""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import numpy as np


def normalize(data):
    """
    Rescales data to be in the range (0,1)

    Parameters
    ----------
    data : numpy.array
        data to be rescaled

    Returns
    -------
    data : numpy.array
        rescaled data
    """
    data = np.abs(data)
    return data / np.max(data)


def contrast_adjustment(data, lower_percentile, upper_percentile):
    """
    Contrast adjustment by saturating extreme values

    Parameters
    ----------
    data : numpy.array
        input data
    lower_percentile : int
        values bellow this percentile will be set to 0
    upper_percentile : int
        values above this percentile will be set to 1

    Returns
    -------
    data : numpy.array
        original data with extreme values saturated
    """
    lower, upper = np.percentile(data, (lower_percentile, upper_percentile))
    data[data < lower] = 0
    data[data > upper] = 1
    return data


def bradley_roth(image, s=None, t=None):
    """
    Implements the Bradley-Roth adaptive thresholding algorithm
    'Adaptive Thresholding Using the Integral Image'
    paper:
    https://people.scs.carleton.ca/~roth/iit-publications-iti/docs/gerh-50002.pdf

    Parameters
    ----------
    image : numpy.array
        image the be tresholded
    s : int
        window size
    t : int
        thresholding sensitivity

    Returns
    -------
    image : numpy.array
        returns trehsholded image in binary values
    """
    # -- from somewhere
    img = np.array(image).astype(np.float)

    # -- default window size is round(width/8)
    if s is None:
        s = np.round(img.shape[1] / 8)

    # -- default threshold is 15% of the total area in the window
    if t is None:
        t = 15.0

    # -- integral image
    intImage = np.cumsum(np.cumsum(img, axis=1), axis=0)

    # -- define grid of points
    (rows, cols) = img.shape[:2]
    (X, Y) = np.meshgrid(np.arange(cols), np.arange(rows))

    # -- make into 1D grid of coordinates for easier access
    X = X.ravel()
    Y = Y.ravel()

    # -- ensures is even so that we are able to index the image properly
    s = s + np.mod(s, 2)

    # -- access the four corners of each neighborhood area
    x1 = X - s / 2
    x2 = X + s / 2
    y1 = Y - s / 2
    y2 = Y + s / 2

    # -- assert no coordinates are out of bounds
    x1[x1 < 0] = 0
    x2[x2 >= cols] = cols - 1
    y1[y1 < 0] = 0
    y2[y2 >= rows] = rows - 1

    # -- assert coordinates are integers
    x1 = x1.astype(np.int)
    x2 = x2.astype(np.int)
    y1 = y1.astype(np.int)
    y2 = y2.astype(np.int)

    # -- count how many pixels are in each neighborhood
    count = (x2 - x1) * (y2 - y1)

    # -- compute the row and column coordinates to access each
    # -- corner of the neighborhood for the integral image
    f1_x = x2
    f1_y = y2
    f2_x = x2
    f2_y = y1 - 1
    f2_y[f2_y < 0] = 0
    f3_x = x1 - 1
    f3_x[f3_x < 0] = 0
    f3_y = y2
    f4_x = f3_x
    f4_y = f2_y

    # -- compute areas of each window
    sums = intImage[f1_y, f1_x] - intImage[f2_y, f2_x] - intImage[f3_y, f3_x] + intImage[f4_y, f4_x]

    # -- compute thresholded image and reshape into a 2D grid
    out = np.zeros(rows * cols, dtype=np.bool)
    out[img.ravel() * count <= sums * (100.0 - t) / 100.0] = True

    # -- convert back to uint8
    out = np.reshape(out, (rows, cols)).astype(np.uint8)

    return out
