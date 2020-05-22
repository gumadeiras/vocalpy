# -*- coding: utf-8 -*-
"""VocalPy - Vocal analysis framework"""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import logging

from os.path import join
from multiprocessing import cpu_count


def create_logger(args=None, out_dir=None):
    """
    Creates a logger to log information during execution
    """
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
            datefmt="%d-%b-%y %H:%M:%S",
            handlers=[
                logging.FileHandler(f"{out_dir}/output.log"),
                logging.StreamHandler(),
            ],
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
    validate_bin_size(args.bin_size)
    validate_thread_count(args.threads)
    validate_animal(args.animal)
    args.frequency = validate_frequency_range(args.frequency, args.animal)
    return args


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
        elif animal == "rat":
            return "20000,-1"
        elif animal == "guineapig":
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
        print(
            "WARNING: number of threads is equal or higher than number of available cores."
        )
        print(
            "WARNING: if your CPU has hyperthreading, use number of physical cores for better performance."
        )
        print(f"provided value: {threads}")
        print(f"computer thread count: {num_cores}")
    return 0


def validate_animal(animal):
    if animal not in ["mouse", "rat", "guineapig"]:
        print("available pipelines are: mouse, rat, guineapig")
        print(f"provided value: {animal}")
        exit()
    return 0
