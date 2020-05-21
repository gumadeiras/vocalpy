# -*- coding: utf-8 -*-
'''VocalPy - Vocal analysis framework'''

__license__ = 'Apache License, Version 2.0'
__copyright__ = '2020 Dietrich Lab - Yale University School of Medicine'

from vocalpy.pipelines import mouse, rat, guineapig


class Animal(object):
    """
    Animal class calls apropriate pipeline functions
    """

    def __init__(self, animal):
        self._available_pipelines = ['mouse', 'rat', 'guineapig']
        self._default_animal = 'mouse'
        self._animal = animal if (
            animal in self._available_pipelines) else self._default_animal

    def identify_vocalizations():
        pass

    def classify_vocalizations():
        pass

    def check_if_vocals_are_close(self, first_vocal, second_vocal):
        if self._animal == 'mouse':
            next_vocal_is_close = mouse.check_if_vocals_are_close(first_vocal, second_vocal)
        elif self._animal == 'rat':
            next_vocal_is_close = rat.check_if_vocals_are_close(first_vocal, second_vocal)
        elif self._animal == 'guineapig':
            next_vocal_is_close = guineapig.check_if_vocals_are_close(first_vocal, second_vocal)

        return next_vocal_is_close
