# -*- coding: utf-8 -*-
'''VocalPy - A python version based on VocalMat'''

__email__ = 'gustavo.santana@yale.edu'
__license__ = 'Apache License, Version 2.0'
__copyright__ = '2020 Dietrich Lab - Yale University School of Medicine'

import glob

import numpy as np

from PIL import Image

import torch
import torch.utils.data as data

import torchvision.models as models
import torchvision.transforms as transforms

from utils.io import load_checkpoint

class NoiseClassifier(object):
    '''
    CNN noise classifier class
    '''

    def __init__(self):
        self.model = load_pretrained_noise_model()

    def load_pretrained_noise_model(self):
        '''
        load MobileNet-V2 pretrained model
        trained to classify spectrograms as Vocal or Noise
        '''
        print('load_pretrained_noise_model not implemented')
        model = load_checkpoint("../models/noise_classifier.pth")
        return model


class ClassClassifier(object):
    '''
    CNN class classifier class
    '''

    def __init__(self,
                 model='resnet50',
                 pretrained=True):

        if model == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
        elif model == 'alexnet':
            self.model = models.alexnet(pretrained=pretrained)


class VocalDatasetFromFolder(data.Dataset):
    """Create a vocalization dataset

    Arguments:
        dataset_path {string} -- path to the dataset

    Keyword Arguments:
        transforms {optional} -- A function/transform that takes in an PIL image and returns a transformed version. (default: {None})
    """

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.transforms = transforms
        self.file_names = [f for f in glob.glob("*.png")]
        self.images = [os.path.join(self.dataset_path, f + '.png') for f in file_names]

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        transToTensor = transforms.Compose([transforms.ToTensor()])
        img = transToTensor(Image.fromarray(img))

        return img

    def __len__(self):
        return len(self.images)

# class VocalDataset(data.Dataset):
#     """Create a vocalization dataset

#     Arguments:
#         dataset_path {string} -- path to the dataset, expects the following structure:
#                                 ./dataset_path
#                                  |-- filenames.txt (list of file names)
#                                  |-- labels/class/*.png (sample images)

#     Keyword Arguments:
#         transforms {optional} -- A function/transform that takes in an PIL image and returns a transformed version. (default: {None})
#     """
#     def __init__(self,
#                  dataset_path,
#                  transforms=None):
#         self.transforms        = transforms
#         self.dataset_path      = dataset_path
#         self.image_path        = os.path.join(dataset_path, 'images')

#         file_list = os.path.join(dataset_path, 'filenames.txt')

#         with open(file_list, "r") as f:
#             file_names = [x.strip() for x in f.readlines()]

#         self.images = [os.path.join(self.image_path, x + '.png') for x in file_names]

#     def __getitem__(self, index):
#         img    = Image.open(self.images[index])

#         if self.transforms is not None:
#             # new_seed = np.random.randint(1, 2**32-1, 1)
#             ia.seed(int(torch.random.initial_seed()/2**32))
#             img, target   = self.transforms(img, target)

#         transToTensor = transforms.Compose([
#                         transforms.ToTensor(),
#                     ])
#         img           = transToTensor(Image.fromarray(img))

#         # return img # -- use this for mean,std calc
#         return img

#     def __len__(self):
#         return len(self.images)
