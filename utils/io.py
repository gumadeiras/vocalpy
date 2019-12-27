# -*- coding: utf-8 -*-
'''VocalPy - A python version based on VocalMat'''

__email__ = 'gustavo.santana@yale.edu'
__license__ = 'Apache License, Version 2.0'
__copyright__ = '2020 Dietrich Lab - Yale University School of Medicine'

import os
import cv2
import glob
import torch
import pickle

import numpy as np
import matplotlib.pyplot as plt


def save_file(file, filename, path):
    if os.path.exists(path) is False:
        raise ValueError("path does not existe: {}".format(path))

    pickle.dump(file, open(os.path.join(path, filename + '.vocalpy'), 'wb'))


def load_file(filename, path):
    if os.path.exists(path) is False:
        raise ValueError("path does not existe: {}".format(path))

    return pickle.load(open(os.path.join(path, filename + '.vocalpy'), 'rb'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path.
    If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("file doesn't exist {}".format(checkpoint))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint, map_location=torch.device(device))
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def parse_input_path(path=None):
    """ parse input path string;
    if it's a directory, return list of files;
    if it's a file, return the file path.

    Args:
        path: (string) path provided by the user
    """
    if path is None:
        print('usage: python vocalpy.py --audio_path=\"/path/to/audio\"')
        exit()
    elif os.path.isdir(path):
        print('path is a directory, geting all .wav files')
        types = (os.path.join(path, '*.wav'), os.path.join(path, '*.WAV'))
        files_found = []
        for files in types:
            files_found.extend(glob.glob(files))
        return files_found
    elif os.path.isfile(path):
        print('path is a file.')
        return [path]
    else:
        print('path is not a file or directory: {}'.format(path))
        exit()

    return 0


def create_output_directory_structure(list_of_files=None):
    """ create directory structure for output files from VocalPy

    Args:
        list_of_files: (list <string>) list of files provided by the user
    """
    if list_of_files is None:
        print("list of input files can not be None")
        exit()

    list_of_output_dirs = []
    print("list of files detected:")
    for file in list_of_files:
        print(os.path.basename(file))
        # -- split "/path/to/file.wav" to ["/path/to/file", ".wav]
        basepath = os.path.splitext(file)
        # -- output dir will be "/path/to/file_outputs"
        output_dir = basepath[0] + '_outputs'
        list_of_output_dirs.append(output_dir)

    return list_of_output_dirs


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return 0


def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    plt.imshow(labeled_img)
    plt.show()
