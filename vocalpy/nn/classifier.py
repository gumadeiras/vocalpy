# -*- coding: utf-8 -*-
"""VocalPy - Vocal analysis framework"""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import os
import torch

import numpy as np
import torch.nn as nn
import torchvision.models as models

from torch.autograd import Variable
from torch.nn.functional import softmax
from os.path import join, dirname, pardir

from vocalpy.utils.io import load_checkpoint
from vocalpy.nn import datasets


class VocalClassifier(object):
    """
    Vocalization classifier

    Parameters
    ----------
    network_type : str
        vocal classifier network_type ('noise', or 'class')
    source : str or numpy.ndarray
        path to directory with spectrograms or array with data to be classified
    batch_size : str, optional
        batch size to use with the neural network
    path_to_checkpoint : str, optional
        path to checkpoint to laod pretrained neural network model
    """

    def __init__(self, network_type, source, batch_size=32, path_to_checkpoint=None):
        if network_type in ["noise", "class"]:
            self.network_type = network_type
        else:
            print(f"VocalClassifiier network_type must be 'noise' or 'class'")
            print(f"provided value {network_type}")

        # self.source = source
        self.batch_size = batch_size
        self.path_to_checkpoint = path_to_checkpoint

        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_available else "cpu")

        if self.network_type == "noise":
            self.model = self.load_pretrained_noise_model(self.device, self.path_to_checkpoint)
        else:
            self.model = self.load_pretrained_class_model(self.device, self.path_to_checkpoint)

        self.dataset = self.create_dataset(source)
        self.dataloader = datasets.create_dataloader(self.dataset, self.batch_size)

    def load_pretrained_noise_model(self, device, model=None):
        """
        Loads pretrained Class CNN model by default, trained to classify spectrograms
        as Vocal or Noise; or model at path provided by the user

        Parameters
        ----------
        device : torch.device
            device to run (CPU or GPU)
        model : str, optional
            path to checkpoint for a neural network model
        """
        if model is None:
            model = models.mobilenet_v2()
            model.classifier = nn.Sequential(
                nn.Dropout(0.2), nn.Linear(1280, 1024), nn.ReLU(inplace=True), nn.Linear(1024, 2),
            )

            model_path = join(dirname(__file__), pardir, "nn", "pretrained", "noise_model.pth.tar")
            classifier_dir_path = os.path.dirname(os.path.abspath(__file__))
            model_path = join(classifier_dir_path, model_path)
            load_checkpoint(model_path, model, device)
            model.eval()

        else:
            load_checkpoint(model, device)

        self.classes = ["noise", "vocal"]
        return model

    def load_pretrained_class_model(self, device, model=None):
        """
        Loads pretrained Class CNN model by default, trained to classify spectrograms
        as one of eleven classes:
            chevron, complex, down_fm, flat, mult_steps, rev_chevron,
            short, step_down, step_up, two_steps, up_fm
        or model at path provided by the user

        Parameters
        ----------
        device : torch.device
            device to run (CPU or GPU)
        model : str, optional
            path to checkpoint for a neural network model
        """
        if model is None:
            model = models.mobilenet_v2()
            # -- add extra layers after the 'classifier' sequence
            model.classifier = nn.Sequential(
                nn.Dropout(0.2), nn.Linear(1280, 1024), nn.ReLU(inplace=True), nn.Linear(1024, 11),
            )

            model_path = join(dirname(__file__), pardir, "nn", "pretrained", "class_model.pth.tar")
            classifier_dir_path = os.path.dirname(os.path.abspath(__file__))
            model_path = join(classifier_dir_path, model_path)
            load_checkpoint(model_path, model, device)
            model.eval()

        else:
            load_checkpoint(model, device)

        self.classes = [
            "chevron",
            "complex",
            "down_fm",
            "flat",
            "mult_steps",
            "rev_chevron",
            "short",
            "step_down",
            "step_up",
            "two_steps",
            "up_fm",
        ]
        return model

    def create_dataset(self, source):
        """
        Creates a dataset by instantiating the VocalDatasetFromFolder class

        Parameters
        ----------
        source : str or numpy.ndarray
            if path -> directory that contains the spectrogram images used to create the dataset
            if ndarray -> return dataset from array
        """
        if isinstance(source, np.ndarray):
            return datasets.VocalDatasetFromArray(source)

        return datasets.VocalDatasetFromFolder(source)

    def classify_list_of_vocals(self, list_of_vocals):
        """
        Classify a :class:`ListOfVocals` using a Neural Network

        Parameters
        ----------
        list_of_vocals : :class:`ListOfVocals`
            list of vocals to be classified
        """
        # -- is list of vocals is empty, just return
        if list_of_vocals.number_of_vocals < 1:
            print("[classify vocals as noise]: list of vocals is empty")
            return -1

        if self.network_type == "noise":
            return self.classify_list_of_vocals_noise(list_of_vocals)

        return self.classify_list_of_vocals_class(list_of_vocals)

    def classify_list_of_vocals_class(self, list_of_vocals):
        """
        Classify a :class:`ListOfVocals` into vocal classes using a Neural Network

        Parameters
        ----------
        list_of_vocals : :class:`ListOfVocals`
            list of vocals to be classified
        """
        predictions = []

        # compute metrics over the dataset
        for itr, image in enumerate(self.dataloader):
            image = image.to(self.device)
            image = Variable(image)

            score = self.model(image)
            predicted = softmax(score.data, dim=1)
            predictions.append(predicted.numpy())

        return np.vstack(predictions)

    def classify_list_of_vocals_noise(self, list_of_vocals):
        """
        Classify a :class:`ListOfVocals` as Vocal or Noise using a Neural Network

        Parameters
        ----------
        list_of_vocals : :class:`ListOfVocals`
            list of vocals to be classified
        """
        predictions = []

        # compute metrics over the dataset
        for itr, image in enumerate(self.dataloader):
            image = image.to(self.device)
            image = Variable(image)

            score = self.model(image)
            _, predicted = torch.max(score.data, 1)
            predictions.append(predicted.numpy())

        return np.hstack(predictions).astype("bool")

    def remove_candidates_classified_as_noise(self, classifications, list_of_vocals):
        print("remove_candidates_classified_as_noise() not implemented")
        return 0
