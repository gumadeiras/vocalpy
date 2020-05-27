# -*- coding: utf-8 -*-
"""VocalPy - Vocal analysis framework"""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

from time import time
from logging import getLogger
from argparse import ArgumentParser

from vocalpy.classes.recording import Recording
from vocalpy.utils.misc import create_logger, validate_arguments
from vocalpy.utils.io import parse_input_path, create_output_directory_structure, create_directory

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("-a", "--animal", help="choose from ['mouse', 'rat']", type=str, default="mouse")
    p.add_argument("-p", "--path_to_audio", help="path to audio file or directory", type=str, default=None)
    p.add_argument(
        "-b", "--bin_size", help="bin size in seconds to split spectrogram processing (default=60)", type=int, default=60
    )
    p.add_argument(
        "-f",
        "--frequency",
        help="frequency range to compute spectrogram; string format: 'lower range,upper range'; '0,-1' to use full range",
        type=str,
        default="default",
    )
    p.add_argument("-t", "--threads", help="number of threads (default=max)", type=int, default=-1)
    p.add_argument("-v", "--verbose", help="enable output verbosity", action="store_true")
    p.add_argument(
        "-l", "--validation", help="saves overlay of segmentation on spectrogram for manual verifcation", action="store_true"
    )
    args = p.parse_args()

    # -- let the user know about the help menu and verbose output
    print("run 'python vocalpy.py -h' to show the help menu")
    print("use '-v' to enable verbose output (recommended)")

    # -- parse input audio path provided by the user
    list_of_files = parse_input_path(args.path_to_audio)
    list_of_output_dirs = create_output_directory_structure(list_of_files)

    # -- assert the number of input files and output dirs are the same
    try:
        assert len(list_of_files) == len(list_of_output_dirs)
    except AssertionError:
        print("list of audio files provided and list to be processed are different")
        print(
            "number of audio files: {}; number of files to be processed: {}".format(
                len(list_of_files), len(list_of_output_dirs)
            )
        )
        exit()

    # -- validate arguments provided by the user
    validate_arguments(args)

    # -- process each input file sequentially
    # -- each file is broken into chunks
    # -- chunks are processed in parallel
    timeAllrecordings = time()
    for file_idx in range(0, len(list_of_files)):
        audio_path = list_of_files[file_idx]
        output_dir = list_of_output_dirs[file_idx]

        # -- create output diretory and logger
        create_directory(output_dir)
        create_logger(args, output_dir)
        logger = getLogger()
        logger.info("selected file:\n{}".format(audio_path))
        logger.info("output files will be saved to:\n{}".format(output_dir))
        logger.info("selected animal pipeline: {}".format(args.animal))

        # -- create recording object
        timeStart = time()
        recording = Recording(recording_path=audio_path, args=args)
        # recording.save_recording_object(path=recording.output_dir)
        print(recording)

        logger.info("recording object created ({:.2f}s)".format((time() - timeStart)))
        logger.info("recording duration: {:.2f} seconds".format(recording.recording_duration))
        logger.info("splitting audio into {} chunks".format(recording.bins))

        # -- identify vocalizations
        recording.identify_vocalizations()

        # -- classify vocalizations
        recording.classify_vocalizations()

        # -- done, save output files :)
        recording.save_outputs(validation_flag=args.validation)

        logger.info("total runtime: {:.0f}m {:.0f}s".format((time() - timeStart) // 60, (time() - timeStart) % 60))

    logger.info(
        "total runtime for all recordings: {:.0f}m {:.0f}s".format(
            (time() - timeAllrecordings) // 60, (time() - timeAllrecordings) % 60
        )
    )
