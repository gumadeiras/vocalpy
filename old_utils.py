# -*- coding: utf-8 -*-
'''VocalPy - A python version based on VocalMat'''

__email__ = 'gustavo.santana@yale.edu'
__license__ = 'Apache License, Version 2.0'
__copyright__ = '2020 Dietrich Lab - Yale University School of Medicine'

import os
import cv2
import pickle
import logging
import warnings


import numpy as np
import matplotlib.pyplot as plt

from time import time
from math import ceil
from scipy import signal, ndimage
from skimage import exposure, measure

from classes.vocal import Vocal
from list_of_vocals import ListOfVocals
