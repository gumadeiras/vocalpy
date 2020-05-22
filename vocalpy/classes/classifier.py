# -*- coding: utf-8 -*-
"""VocalPy - Vocal analysis framework"""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import os
import torch

import numpy as np
import torch.nn as nn
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from glob import glob
from torch.autograd import Variable
from torch.nn.functional import softmax
from os.path import join, basename, splitext

from vocalpy.utils.io import load_checkpoint


class VocalClassifier(object):
    """
    Vocalization classifier

    Parameters
    ----------
    network_type : str
        vocal classifier network_type ('noise', or 'class')
    path_to_spectrograms : str
        path to directory with spectrograms to be classified
    batch_size : str, optional
        batch size to use with the neural network
    path_to_checkpoint : str, optional
        path to checkpoint to laod pretrained neural network model
    """

    def __init__(self, network_type, path_to_spectrograms, batch_size=32, path_to_checkpoint=None):
        if network_type in ["noise", "class"]:
            self.network_type = network_type
        else:
            print(f"VocalClassifiier network_type must be 'noise' or 'class'")
            print(f"provided value {network_type}")

        self.path_to_spectrograms = path_to_spectrograms
        self.batch_size = batch_size
        self.path_to_checkpoint = path_to_checkpoint

        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_available else "cpu")

        if self.network_type == "noise":
            self.model = self.load_pretrained_noise_model(self.device, self.path_to_checkpoint)
        else:
            self.model = self.load_pretrained_class_model(self.device, self.path_to_checkpoint)

        self.dataset = self.create_dataset(self.path_to_spectrograms)
        self.dataloader = self.create_dataloader(self.dataset, self.batch_size)

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

            model_path = "../models/noise_model.pth.tar"
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

            model_path = "../models/class_model.pth.tar"
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

    def create_dataset(self, path_to_spectrograms):
        """
        Creates a dataset by instantiating the VocalDatasetFromFolder class

        Parameters
        ----------
        path_to_spectrograms : str
            path to directory that contains the spectrogram images used to create the dataset
        """
        return VocalDatasetFromFolder(path_to_spectrograms)

    def create_dataloader(self, dataset, batch_size):
        """
        Creates a DataLoader to load data from the dataset

        Parameters
        ----------
        dataset : :class:`VocalDatasetFromFolder`
        batch_size : int
        """
        return data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

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
        else:
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


class VocalDatasetFromFolder(data.Dataset):
    """
    Creates a vocalization dataset from a directory containing spectrograms

    Parameters
    ----------
    dataset_path : str
        path to the directory containing the spectrograms
    """

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        # -- get file names and sort ascending
        self.filenames = sorted([basename(splitext(f)[0]) for f in glob(join(self.dataset_path, "*.png"))], key=int,)
        # -- build back full path to images
        self.images = [join(self.dataset_path, f + ".png") for f in self.filenames]

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        transToTensor = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        img = transToTensor(img)
        return img

    def __len__(self):
        return len(self.images)
