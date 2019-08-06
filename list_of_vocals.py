# -*- coding: utf-8 -*-
'''VocalPy - A python version based on (VocalMat by Antonio Fonseca)'''

__author__    = 'Gustavo Madeira Santana'
__email__     = 'gustavo.santana@yale.edu'
__copyright__ = '2019 Dietrich Lab - Yale University School of Medicine'

import numpy as np

class ListOfVocals(object):
    '''
    list of vocalizations
    '''
    def __init__(self, vocals_in_recording=None, number_of_vocals=None):
        self.vocals_in_recording = np.hstack(vocals_in_recording)
        self.number_of_vocals    = len(self.vocals_in_recording)
        self.vocals_processed    = False

    def __repr__(self):
        # return "{}: vocals_in_recording: {}, \n number_of_vocals: {}, \n vocals_processed: {}".format(self.__class__.__name__, self.vocals_in_recording, self.number_of_vocals, self.vocals_processed)
        return "{}: number_of_vocals: {}, \n vocals_processed: {}".format(self.__class__.__name__, self.number_of_vocals, self.vocals_processed)
