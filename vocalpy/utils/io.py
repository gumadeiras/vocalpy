# -*- coding: utf-8 -*-
"""VocalPy - Vocal analysis framework"""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import glob
import yaml
import shutil
import pickle

import numpy as np
from os import makedirs
from os.path import basename, exists, isdir, isfile, join, splitext


def save_file(file, filename, path):
    """
    Saves vocalpy object to a path

    Parameters
    ----------
    file : Object
        object to be saved
    filename : str
        filename to be used
    path : str
        path to save the object

    Raises
    ------
    ValueError
        if the path does not exist
    """
    if exists(path) is False:
        raise ValueError(f"path does not existe: {path}")

    pickle.dump(file, open(join(path, filename + ".vocalpy"), "wb"))


def load_file(filename, path):
    """
    Loads vocalpy object from a path

    Parameters
    ----------
    filename : str
        object filename
    path : str
        path to object

    Raises
    ------
    ValueError
        if the path does not exist
    """
    if exists(path) is False:
        raise ValueError(f"path does not existe: {path}")

    return pickle.load(open(join(path, filename + ".vocalpy"), "rb"))


def load_recording_data(path):
    """
    Loads vocalpy recording object from a path

    Parameters
    ----------
    path : str
        full path to object

    Raises
    ------
    ValueError
        if the file does not exist
    """
    if exists(path) is False:
        raise ValueError(f"file does not existe: {path}")
    return np.load(path, allow_pickle=True)


def load_checkpoint(checkpoint, model, device, optimizer=None):
    """Loads model parameters (state_dict) from file_path.
    If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    import torch

    if not exists(checkpoint):
        print(f"file does not exist {checkpoint}")
        exit()

    checkpoint = torch.load(checkpoint, map_location=torch.device(device))
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optim_dict"])

    return checkpoint


def load_model(model_path, device):
    """
    Directly load a pretrained pytorch model

    Parameters
    ----------
    model_path : str
        path to model
    device : torch.device
        device to run (CPU or GPU)
    """
    import torch

    return torch.load(model_path, map_location=torch.device(device))


def parse_input_path(path=None, search_tree=False):
    """
    Parse input path. If it is a directory, return list of files; if it
    is a file, return the file path

    Parameters
    ----------
    path : str
        path provided by the user
    """
    if path is None:
        print("usage: python vocalpy.py --audio_path='/path/to/audio'")
        return -1
    if isdir(path):
        print("audio path is a directory, geting all .wav files")
        if search_tree:
            types = (
                join(path, "**/*.wav"),
                join(path, "**/*.WAV"),
                join(path, "**/*.flac"),
                join(path, "**/*.FLAC"),
            )
        else:
            types = (
                join(path, "*.wav"),
                join(path, "*.WAV"),
                join(path, "*.flac"),
                join(path, "*.FLAC"),
            )
        files_found = []
        for files in types:
            files_found.extend(glob.glob(files, recursive=search_tree))
        return files_found
    elif isfile(path):
        print("audio path is a file.")
        return [path]

    print(f"audio path is not a file or directory: {path}")
    return -1


def create_output_directory_structure(list_of_files):
    """
    Creates directory structure for output files from VocalPy

    Parameters
    ----------
    list_of_files : List[str]
        list of files provided by the user
    """

    list_of_output_dirs = []
    print("list of files detected:")
    for file in list_of_files:
        print(basename(file))
        # -- split '/path/to/file.wav' to ['/path/to/file', '.wav]
        basepath = splitext(file)
        # -- output dir will be '/path/to/file_outputs'
        output_dir = basepath[0] + "_outputs"
        list_of_output_dirs.append(output_dir)

    return list_of_output_dirs


def create_directory(path):
    """
    Creates a directory at the provided path

    Parameters
    ----------
    path : str
        path to be created
    """
    if not exists(path):
        makedirs(path, exist_ok=True)
    return 0


def remove_directory(path):
    """
    Removes a directory at the provided path

    Parameters
    ----------
    path : str
        path to be removed
    """
    shutil.rmtree(path, ignore_errors=True)
    return 0


def read_yaml(path_to_file):
    """
    Loads a YAML configuration file

    Parameters
    ----------
    path_to_file : str
        path to YAML file

    Returns
    -------
    yml_data : dict
        YAML file data read into a dictionary
    """
    with open(path_to_file, "r") as ymlfile:
        yml_data = yaml.safe_load(ymlfile)
    return yml_data
