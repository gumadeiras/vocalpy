# -*- coding: utf-8 -*-
"""VocalPy - Vocal analysis framework"""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

from vocalpy.pipelines import mouse, rat, guineapig


class Animal(object):
    """
    Animal class calls apropriate pipeline functions
    """

    def __init__(self, animal):
        self._available_identification_pipelines = ["mouse", "rat", "guineapig"]
        self._available_classification_pipelines = ["mouse"]
        self._animal = animal if (animal in self._available_identification_pipelines) else self._available_pipelines[0]
        self._has_classifier = True if (animal in self._available_classification_pipelines) else None

    def identify_vocalizations(self, chunk):
        # -- import animal pipelines here
        import vocalpy.pipelines.rat as rat
        import vocalpy.pipelines.mouse as mouse
        import vocalpy.pipelines.guineapig as guineapig

        if self._animal == "mouse":
            return mouse.identifier(chunk)
        elif self._animal == "rat":
            return rat.identifier(chunk)
        elif self._animal == "guineapig":
            return guineapig.identifier(chunk)

    def classify_vocalizations(self, network_type, list_of_vocals, path_to_spectrograms):
        if self._animal == "mouse":
            return mouse.classifier(network_type, list_of_vocals, path_to_spectrograms)
        elif self._animal == "rat":
            pass
        elif self._animal == "guineapig":
            pass

    def check_if_vocals_are_close(self, first_vocal, second_vocal):
        if self._animal == "mouse":
            next_vocal_is_close = mouse.check_if_vocals_are_close(first_vocal, second_vocal)
        elif self._animal == "rat":
            next_vocal_is_close = rat.check_if_vocals_are_close(first_vocal, second_vocal)
        elif self._animal == "guineapig":
            next_vocal_is_close = guineapig.check_if_vocals_are_close(first_vocal, second_vocal)

        return next_vocal_is_close
