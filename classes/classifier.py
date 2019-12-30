# -*- coding: utf-8 -*-
'''VocalPy - A python version based on VocalMat'''

__email__ = 'gustavo.santana@yale.edu'
__license__ = 'Apache License, Version 2.0'
__copyright__ = '2020 Dietrich Lab - Yale University School of Medicine'

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
from os.path import join, basename, splitext

from utils.io import load_checkpoint


class VocalClassifier(object):
    '''
    CNN noise classifier class
    '''

    def __init__(self, type, path_to_spectrograms, batch_size=32, path_to_checkpoint=None):
        if type in ['noise', 'class']:
            self.type = type
        else:
            print('VocalClassifiier type must be \'noise\' or \'class\'')
            print('provided value {}'.format(type))

        self.path_to_spectrograms = path_to_spectrograms
        self.batch_size = batch_size
        self.path_to_checkpoint = path_to_checkpoint

        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda_available else 'cpu')

        if self.type is 'noise':
            self.model = self.load_pretrained_noise_model(self.path_to_checkpoint, self.device)
        else:
            self.model = self.load_pretrained_class_model(self.path_to_checkpoint, self.device)

        self.dataset = self.create_dataset(self.path_to_spectrograms)
        self.dataloader = self.create_dataloader(self.dataset, self.batch_size)

    def load_pretrained_noise_model(self, path_to_checkpoint, device, model=None):
        '''
        load MobileNet-V2 pretrained model,
        trained to classify spectrograms as Vocal or Noise;
        or model at path provided by the user.
        '''
        if path_to_checkpoint is None:
            model = models.mobilenet_v2()
            model.classifier = nn.Sequential(nn.Dropout(0.2),
                                             nn.Linear(1280, 1024),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(1024, 2))

            model_path = '../models/mobilenet_v2_noise_checkpoint.pth.tar'
            classifier_dir_path = os.path.dirname(os.path.abspath(__file__))
            model_path = join(classifier_dir_path, model_path)
            load_checkpoint(model_path, model, device)
            model.eval()

        else:
            load_checkpoint(path_to_checkpoint, model, device)

        return model

    def load_pretrained_class_model(self, path_to_checkpoint, device, model=None):
        '''
        load MobileNet-V2 pretrained model,
        trained to classify spectrograms as one of eleven classes:
        chevron, complex, down_fm, flat, mult_steps, rev_chevron, short, step_down, step_up, two_steps, up_fm
        or model at path provided by the user.
        '''
        if path_to_checkpoint is None:
            model = models.mobilenet_v2()
            model.classifier = nn.Sequential(nn.Dropout(0.2),
                                             nn.Linear(1280, 1024),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(1024, 2))

            model_path = '../models/mobilenet_v2_noise_checkpoint.pth.tar'
            classifier_dir_path = os.path.dirname(os.path.abspath(__file__))
            model_path = join(classifier_dir_path, model_path)
            load_checkpoint(model_path, model, device)
            model.eval()

        else:
            load_checkpoint(path_to_checkpoint, model, device)

        return model

    def create_dataset(self, path_to_spectrograms):
        '''
        create a dataset by instantiating the VocalDatasetFromFolder class

        Args:
            path_to_spectrograms: (string) path to directory that
                contains the spectrogram images used to create the dataset
        '''
        return VocalDatasetFromFolder(path_to_spectrograms)

    def create_dataloader(self, dataset, batch_size):
        '''
        create a DataLoader to load data from the dataset

        Args:
            dataset: (VocalDatasetFromFolder) dataset class instance
            batch_size: (int) batch size number
        '''
        return data.DataLoader(dataset,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=0)

    def classify_list_of_vocals(self, list_of_vocals):
        if self.type is 'noise':
            return self.classify_list_of_vocals_noise(list_of_vocals)
        else:
            return self.classify_list_of_vocals_class(list_of_vocals)

    def classify_list_of_vocals_class(self, list_of_vocals):
        '''
        classify candidate vocalizations found in the recording;
        candidates are classified as vocal or noise;

        Args:
            list_of_vocals: (ListOfVocals) list of candidate vocals

        returns class probabilities for all vocals in the list of vocals
        '''
        predictions = []

        # compute metrics over the dataset
        for itr, image in enumerate(self.dataloader):
            image = image.to(self.device)
            image = Variable(image)

            score = self.model(image)
            _, predicted = torch.max(score.data, 1)
            predictions.append(predicted.numpy())

        return np.hstack(predictions).astype('bool')

    def classify_list_of_vocals_noise(self, list_of_vocals):
        '''
        classify candidate vocalizations found in the recording;
        candidates are classified as vocal or noise;

        Args:
            list_of_vocals: (ListOfVocals) list of candidate vocals

        returns class probabilities for all vocals in the list of vocals
        '''
        predictions = []

        # compute metrics over the dataset
        for itr, image in enumerate(self.dataloader):
            image = image.to(self.device)
            image = Variable(image)

            score = self.model(image)
            _, predicted = torch.max(score.data, 1)
            predictions.append(predicted.numpy())

        return np.hstack(predictions).astype('bool')

    def remove_candidates_classified_as_noise(self, classifications, list_of_vocals):
        print('remove_candidates_classified_as_noise() not implemented')
        return 0


class VocalClassClassifier(object):
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
    '''Create a vocalization dataset

    Arguments:
        dataset_path {string} -- path to the dataset

    Keyword Arguments:
        transforms {optional} -- A function/transform that takes in a PIL
        image and returns a transformed version. (default: {None})
    '''

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        # -- get file names and sort ascending
        self.filenames = sorted([basename(splitext(f)[0]) for f in glob(join(self.dataset_path, '*.png'))], key=int)
        # -- build back full path to images
        self.images = [join(self.dataset_path, f + '.png') for f in self.filenames]

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        transToTensor = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.ToTensor()])
        img = transToTensor(img)
        return img

    def __len__(self):
        return len(self.images)

# class VocalDataset(data.Dataset):
#     '''Create a vocalization dataset

#     Arguments:
#         dataset_path {string} -- path to the dataset, expects the following structure:
#                                 ./dataset_path
#                                  |-- filenames.txt (list of file names)
#                                  |-- labels/class/*.png (sample images)

#     Keyword Arguments:
#         transforms {optional} -- A function/transform that takes in an PIL image and returns a transformed version. (default: {None})
#     '''
#     def __init__(self,
#                  dataset_path,
#                  transforms=None):
#         self.transforms        = transforms
#         self.dataset_path      = dataset_path
#         self.image_path        = os.path.join(dataset_path, 'images')

#         file_list = os.path.join(dataset_path, 'filenames.txt')

#         with open(file_list, 'r') as f:
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
