# -*- coding: utf-8 -*-
'''VocalPy - A python version based on VocalMat'''

__email__ = 'gustavo.santana@yale.edu'
__license__ = 'Apache License, Version 2.0'
__copyright__ = '2020 Dietrich Lab - Yale University School of Medicine'

import glob
import torch
import shutil
import pickle

from sys import exit
from os import makedirs
from os.path import basename, exists, isdir, isfile, join, splitext


def save_file(file, filename, path):
    if exists(path) is False:
        raise ValueError("path does not existe: {}".format(path))

    pickle.dump(file, open(join(path, filename + '.vocalpy'), 'wb'))


def load_file(filename, path):
    if exists(path) is False:
        raise ValueError("path does not existe: {}".format(path))

    return pickle.load(open(join(path, filename + '.vocalpy'), 'rb'))


def load_checkpoint(checkpoint, model, device, optimizer=None):
    '''Loads model parameters (state_dict) from file_path.
    If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    '''
    if not exists(checkpoint):
        print("file doesn't exist {}".format(checkpoint))
        exit()

    checkpoint = torch.load(checkpoint, map_location=torch.device(device))
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def load_model(model_path, device):
    '''
    directly load a pretrained model
    '''
    return torch.load(model_path, map_location=torch.device(device))


def parse_input_path(path=None):
    ''' parse input path string;
    if it's a directory, return list of files;
    if it's a file, return the file path.

    Args:
        path: (string) path provided by the user
    '''
    if path is None:
        print('usage: python vocalpy.py --audio_path=\"/path/to/audio\"')
        exit()
    elif isdir(path):
        print('path is a directory, geting all .wav files')
        types = (join(path, '*.wav'), join(path, '*.WAV'))
        files_found = []
        for files in types:
            files_found.extend(glob.glob(files))
        return files_found
    elif isfile(path):
        print('path is a file.')
        return [path]
    else:
        print('path is not a file or directory: {}'.format(path))
        exit()

    return 0


def create_output_directory_structure(list_of_files=None):
    ''' create directory structure for output files from VocalPy

    Args:
        list_of_files: (list <string>) list of files provided by the user
    '''
    if list_of_files is None:
        print("list of input files can not be None")
        exit()

    list_of_output_dirs = []
    print("list of files detected:")
    for file in list_of_files:
        print(basename(file))
        # -- split "/path/to/file.wav" to ["/path/to/file", ".wav]
        basepath = splitext(file)
        # -- output dir will be "/path/to/file_outputs"
        output_dir = basepath[0] + '_outputs'
        list_of_output_dirs.append(output_dir)

    return list_of_output_dirs


def create_directory(path):
    if not exists(path):
        makedirs(path, exist_ok=True)
    return 0


def remove_directory(path):
    shutil.rmtree(path, ignore_errors=True)
    return 0
