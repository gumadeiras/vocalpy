# -*- coding: utf-8 -*-
'''VocalPy - A python version based on VocalMat'''

__email__ = 'gustavo.santana@yale.edu'
__license__ = 'Apache License, Version 2.0'
__copyright__ = '2020 Dietrich Lab - Yale University School of Medicine'

from time import time
from logging import getLogger
from argparse import ArgumentParser
from joblib import Parallel, delayed

from classes.recording import Recording
from classes.classifier import VocalClassifier
from classes.list_of_vocals import ListOfVocals
from utils.processing import parallel_audio_processing
from utils.misc import create_logger, validate_arguments
from utils.io import parse_input_path, create_output_directory_structure, create_directory

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-a', '--animal', help='choose from [\'mouse\', \'rat\']', type=str, default='mouse')
    p.add_argument('-p', '--path_to_audio', help='path to audio file or directory', type=str, default=None)
    p.add_argument('-b', '--bin_size', help='bin size in seconds to split spectrogram processing (default=60)', type=int, default=60)
    p.add_argument('-f', '--frequency', help='frequency range to compute spectrogram; string format: \'lower range,upper range\'; \'0,-1\' to use full range', type=str, default='default')
    p.add_argument('-t', '--threads', help='number of threads (default=max)', type=int, default=0)
    p.add_argument('-v', '--verbose', help='enable output verbosity', action='store_true')
    args = p.parse_args()

    # -- let the user know about the help menu and verbose output
    print('run \'python vocalpy.py -h\' to show the help menu')
    print('use \'-v\' to enable verbose output (recommended)')


    # -- parse input audio path provided by the user
    list_of_files = parse_input_path(args.path_to_audio)
    list_of_output_dirs = create_output_directory_structure(list_of_files)
    # -- assert the number of input files and output dirs are the same
    try:
        assert len(list_of_files) == len(list_of_output_dirs)
    except AssertionError:
        print("list of audio files provided and list to be processed are different")
        print("number of audio files: {}; number of files to be processed: {}".format(len(list_of_files), 
                                                                                      len(list_of_output_dirs)))
        exit()

    # -- validate arguments provided by the user
    validate_arguments(args)

    # -- process each input file sequentially
    # -- each file is broken into chunks
    # -- chunks are processed in parallel
    timeAllRecordings = time()
    for file_idx in range(0, len(list_of_files)):
        audio_path = list_of_files[file_idx]
        output_dir = list_of_output_dirs[file_idx]

        # -- create output diretory and logger
        create_directory(output_dir)
        create_logger(args, output_dir)
        logger = getLogger()
        logger.info('selected file:\n{}'.format(audio_path))
        logger.info('output files will be saved to:\n{}'.format(output_dir))
        logger.info('selected animal pipeline: {}'.format(args.animal))

        # -- create Recording object
        timeStart = time()
        recording = Recording(recording_path=audio_path, args=args)
        # recording.save_recording_object(path=recording.output_dir)
        print(recording)

        logger.info('recording object created ({:.2f}s)'.format((time() - timeStart)))
        logger.info('recording duration: {:.2f} seconds'.format(recording.recording_duration))
        logger.info('splitting audio into {} chunks'.format(recording.bins))

        # -- distribute Recording chunks to available cores
        # -- process each chunk and find candidate vocalizations
        timeAParallel = time()
        results = Parallel(n_jobs=args.threads, require='sharedmem')(delayed(parallel_audio_processing)(animal=args.animal, chunk=i) for i in recording.chunks)
        recording.recording_processing_finished()
        logger.info('recording parallel processing ({:.0f}m {:.0f}s)'.format((time() - timeAParallel) // 60, (time() - timeAParallel) % 60))

        # -- create list of vocals found in the recording
        logger.info('combining list of vocals from each bin')
        timeAcombining = time()
        list_of_vocals = ListOfVocals()
        list_of_vocals.combine_list_of_list_of_vocals(list_of_list_of_vocals=results)
        list_of_vocals.update_intervals()
        logger.info('done combining ({:.0f}s)'.format(time() - timeAcombining))
        logger.info(list_of_vocals)

        # -- update recording object and save data (images and csv)
        logger.info('saving spectrograms of candidate vocalizations')
        timeAsaving = time()
        recording._has_list_of_vocals = True
        recording.list_of_vocals = list_of_vocals
        recording.save_spectrograms(path=recording.output_dir)
        logger.info('done saving ({:.0f}s)'.format(time() - timeAsaving))

        if args.animal in ['mouse', 'rat']:
            # -- classify candidate vocalizations as Vocal or Noise; remove Noise
            logger.info('classifying candidate vocalizations as vocal or noise')
            timeAclassification = time()
            NoiseClassifier = VocalClassifier(type='noise', path_to_spectrograms=recording.spectrogram_dir)
            predictions = NoiseClassifier.classify_list_of_vocals(recording.list_of_vocals)
            logger.info('removing candidates classified as noise')
            recording.remove_vocals_classified_as_noise_from_list_of_vocals(predictions)
            recording.save_spectrograms_and_masks(path=recording.output_dir)
            logger.info('done classifying and removing ({:.0f}s)'.format(time() - timeAclassification))
            logger.info(recording._list_of_vocals)

            logger.info('classifying vocalizations')
            timeAclassification = time()
            ClassClassifier = VocalClassifier(type='class', path_to_spectrograms=recording.spectrogram_dir)
            predictions = ClassClassifier.classify_list_of_vocals(recording.list_of_vocals)
            logger.info('adding classification to vocals')
            recording.update_vocals_with_class_classification(predictions, ClassClassifier.classes)
            logger.info('done classifying and updating vocals ({:.0f}s)'.format(time() - timeAclassification))

        # -- we are done :)
        # -- save output files
        logger.info('saving recording object, vocalizations, and csv file')
        timeAsaving = time()
        recording.save_recording_object(path=recording.output_dir)
        recording.remove_spectrograms_and_masks_from_object()
        recording.save_recording_object(filename='recording_without_spectrograms', path=recording.output_dir)
        recording.save_recording_data_to_csv(path=recording.output_dir)
        logger.info('done saving ({:.0f}s)'.format(time() - timeAsaving))

        logger.info('total runtime: {:.0f}m {:.0f}s'.format((time() - timeStart) // 60, (time() - timeStart) % 60))
    logger.info('total runtime for all recordings: {:.0f}m {:.0f}s'.format((time() - timeAllRecordings) // 60, (time() - timeAllRecordings) % 60))
