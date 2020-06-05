# -*- coding: utf-8 -*-
"""VocalPy - Vocal analysis framework"""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import logging

import pandas as pd

from os.path import join, dirname, pardir
from multiprocessing import cpu_count

from vocalpy.utils.io import read_yaml


def create_logger(args=None, out_dir=None):
    """
    Creates a logger to log information during execution
    """
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
            datefmt="%d-%b-%y %H:%M:%S",
            handlers=[logging.FileHandler(f"{out_dir}/output.log"), logging.StreamHandler(),],
        )
        logging.info("verbose output on")
    else:
        print(f"logging to file: {join(out_dir, 'output.log')}")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
            datefmt="%d-%b-%y %H:%M:%S",
            handlers=[logging.FileHandler(f"{out_dir}/output.log"),],
        )


def validate_arguments(args):
    """
    Validates arguments passed by the user

    Parameter
    ---------
    args : ArgumentParser
    """
    validate_animal(args.animal)
    validate_bin_size(args.bin_size)
    args.frequency = validate_frequency_range(args.frequency, args.animal)
    validate_thread_count(args.threads)
    return args


def validate_animal(animal):
    if animal not in ["mouse", "rat", "guineapig"]:
        print("available pipelines are: mouse, rat, guineapig")
        print(f"provided value: {animal}")
        exit()
    return 0


def validate_bin_size(bin_size):
    if bin_size < 0:
        print("bin_size must be a positive integer.")
        print(f"provided value: {bin_size:2f}")
        exit()
    return 0


def validate_frequency_range(frequency_range, animal):
    if frequency_range == "default":
        if animal == "mouse":
            return "45000,-1"
        if animal == "rat":
            return "20000,-1"
        if animal == "guineapig":
            return "0,22000"
    else:
        low_freq, high_freq = [int(f) for f in frequency_range.split(",")]
        if (low_freq > high_freq) & (high_freq != -1):
            print(
                "low frequency cutoff must be lower \
                than the high frequency cutoff."
            )
            print(
                f"provided values: low_freq={low_freq}; \
                high_freq={high_freq}"
            )
            exit()
    return frequency_range


def validate_thread_count(threads):
    num_cores = cpu_count()
    if (threads < -1) or (threads == 0):
        print("number of threads must be a positive integer.")
        print(f"provided value: {threads}")
        print(f"computer core count: {num_cores}")
        exit()
    if threads > num_cores:
        print("WARNING: number of threads is equal or higher than number of available cores.")
        print("WARNING: if your CPU has hyperthreading, use number of physical cores for better performance.")
        print(f"provided value: {threads}")
        print(f"computer thread count: {num_cores}")
    return 0


def check_pipeline_avalability(animal):
    """
    Reads the YAML configuration file for pipeline availability and checks if the animal selected by the user has a
    pipeline implemented

    Parameters
    ----------
    animal : str
        animal pipeline selected by the user

    Returns
    -------
    has_identifier : bool
        availability of the vocalization identifier pipeline
    has_classifier : bool
        availability of the vocalization classification pipeline
    """

    # ToDo: also return pipeline parameters
    pipelines_configs = read_yaml(join(dirname(__file__), pardir, "configs", "pipelines_parameters.yml"))

    identifier_pipelines = []
    classifier_pipelines = []
    for animal_pipeline, available in pipelines_configs["pipelines"].items():
        if available["identifier"]:
            identifier_pipelines.append(animal_pipeline)
        if available["classifier"]:
            classifier_pipelines.append(animal_pipeline)

    has_identifier = animal in identifier_pipelines
    has_classifier = animal in classifier_pipelines
    return has_identifier, has_classifier


def create_dataframe_from_list_of_vocals(list_of_vocals):
    """
    Creates a Pandas DataFrame from a :class:`ListOfVocals`

    Parameters
    ----------
    list_of_vocals : :class:`ListOfVocals`
        list of vocals is a :class:`ListOfVocals` instance

    Returns
    -------
    dataframe : :class:`pandas.DataFrame`
        dataframe containing all vocals from the list of vocals
    """
    dataframe = pd.DataFrame(
        columns=[
            "bin_number",
            "start(s)",
            "end(s)",
            "duration(ms)",
            "interval(s)",
            "min_freq",
            "max_freq",
            "avg_freq",
            "bandwidth",
            "min_intensity",
            "max_intensity",
            "avg_intensity",
            "bg_intensity",
            "area(pixels)",
            "centroid_y",
            "class_top1",
            "class_top2",
        ]
    )

    for vocal in list_of_vocals.vocals_in_recording:
        dataframe = dataframe.append(
            {
                "bin_number": vocal.bin_number,
                "start(s)": vocal.start,
                "end(s)": vocal.end,
                "duration(ms)": vocal.duration,
                "interval(s)": vocal.interval,
                "min_freq": vocal.min_freq,
                "max_freq": vocal.max_freq,
                "avg_freq": vocal.avg_freq,
                "bandwidth": vocal.bandwidth,
                "min_intensity": vocal.min_intensity,
                "max_intensity": vocal.max_intensity,
                "avg_intensity": vocal.avg_intensity,
                "bg_intensity": vocal.bg_intensity,
                "area(pixels)": vocal.area,
                "centroid_y": vocal.centroid[0],
                "class_top1": vocal.top1,
                "class_top2": vocal.top2,
            },
            ignore_index=True,
        )

    # -- sort vocalizations by start time and save csv
    dataframe.sort_values(
        by="start(s)", ascending=True, inplace=True, kind="quicksort", na_position="last",
    )

    return dataframe
