# -*- coding: utf-8 -*-
"""VocalPy - Vocal analysis framework"""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"


import numpy as np
import torch.utils.data as data

from PIL import Image
from glob import glob
from os.path import join, basename, splitext

import torchvision.transforms as transforms


class VocalDatasetFromFolder(data.Dataset):
    """
    Creates a vocalization dataset from a directory containing spectrograms

    Parameters
    ----------
    dataset_path : str
        path to the directory containing the spectrograms
    transform : :class:`torchvision.trasnforms`
        transformations composition
    """

    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        # -- get file names and sort ascending
        self.filenames = sorted([basename(splitext(f)[0]) for f in glob(join(self.dataset_path, "*.png"))], key=int,)
        # -- build back full path to images
        self.images = [join(self.dataset_path, f + ".png") for f in self.filenames]
        self.transform = transform

    def __getitem__(self, index):
        x = Image.open(self.images[index]).convert("RGB")
        if self.transform is None:
            self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        return self.transform(x)

    def __len__(self):
        return len(self.images)


class VocalDatasetFromArray(data.Dataset):
    """
    Creates a vocalization dataset from a numpy ndarray

    Parameters
    ----------
    data : numpy.ndarray
        numpy array with images to construct the dataset
        expects (N,H,W) grayscale images
        N number of images
        H is height
        W is width
    transform : :class:`torchvision.trasnforms`
        transformations composition
    """

    def __init__(self, data, transform=None):
        self.data = np.asarray(data, dtype=np.uint8)
        self.len, self.height, self.width = data.shape
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                ]
            )
        return self.transform(x)

    def __len__(self):
        return len(self.data)


def create_dataloader(dataset, batch_size):
    """
    Creates a :class:`data.DataLoader` to load data from the dataset

    Parameters
    ----------
    dataset : :class:`VocalDatasetFromFolder` or :class:`VocalDatasetFromArray`
        actual dataset
    batch_size : int
        number of items to fetch in each iteration
    """
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def create_array_from_list_of_vocals(list_of_vocals):
    """
    Creates a numpy.ndarray with all detected vocals from recording

    Parameters
    ----------
    list_of_vocals : :class:`ListOfVocals`
        list of vocals detected in the dataset
    """
    array_of_vocals = []
    for vocal in list_of_vocals.vocals_in_recording:
        array_of_vocals.append(vocal.spectrogram)
    return np.asarray(array_of_vocals)
