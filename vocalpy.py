# -*- coding: utf-8 -*-
'''VocalPy - A python version based on VocalMat'''

__email__ = 'gustavo.santana@yale.edu'
__license__ = 'Apache License, Version 2.0'
__copyright__ = '2020 Dietrich Lab - Yale University School of Medicine'

import argparse
import logging
import multiprocessing

from time import time
from joblib import Parallel, delayed

from utils.misc import create_logger
from classes.recording import Recording
from classes.list_of_vocals import ListOfVocals
from utils.processing import parallel_audio_processing
from utils.io import parse_input_path, create_output_directory_structure, create_directory

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-a', '--audio_path', help='audio file name', type=str, default=None)
    p.add_argument('-b', '--bin_size', help='bin size in seconds to split spectrogram processing (default=60)', type=int, default=60)
    p.add_argument('-f', '--frequency', help='frequency range to compute spectrogram; tuple (lower range, upper range); -1 to use maximum range (default=(45000,-1))', type=tuple, default=(45000,-1))
    p.add_argument('-t', '--threads', help='number of threads (default=max)', type=int, default=0)
    p.add_argument('-p', '--plot', help='plot sample spectrogram after each image processing operation', action='store_true')
    p.add_argument('-v', '--verbose', help='enable output verbosity', action='store_true')
    args = p.parse_args()

    # audio_path = args.audio_path
    list_of_files = parse_input_path(args.audio_path)

    # output_dir = os.path.join(audio_path[0:-4] + '_outputs')

    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir, exist_ok=True)
    list_of_output_dirs = create_output_directory_structure(list_of_files)

    # -- assert the number of input files and output dirs are the same
    assert len(list_of_files) == len(list_of_output_dirs)

    # -- process each input file sequentially
    # -- each file is broken into chunks
    # -- chunks are processed in parallel
    for file_idx in range(0, len(list_of_files)):
        audio_path = list_of_files[file_idx]
        output_dir = list_of_output_dirs[file_idx]

        # -- create output diretory and logger
        create_directory(output_dir)
        create_logger(args, output_dir)
        logger = logging.getLogger()
        logger.info('selected file: {}'.format(audio_path))
        logger.info('output files will be saved to: {}'.format(output_dir))

        # -- create Recording object
        timeStart = time()
        audio_recording = Recording(recording_path=audio_path, args=args)
        # audio_recording.save_recording_object(path=audio_recording.output_dir)
        timeBRecording = time()
        print(audio_recording)

        logger.info('recording object created ({:.2f}s)'.format((timeBRecording - timeStart)))
        logger.info('recording duration: {:.2f} seconds'.format(audio_recording.recording_duration))
        logger.info('splitting audio into {} chunks'.format(audio_recording.bins))

        # -- get core count for parallelization
        if args.threads > 0:
            num_cores = args.threads
        else:
            num_cores = multiprocessing.cpu_count()

        # -- distribute Recording chunks to available cores
        # -- process each chunk and find candidate vocalizations
        timeAParallel = time()
        results = Parallel(n_jobs=num_cores, require='sharedmem')(delayed(parallel_audio_processing)(i) for i in audio_recording.chunks)
        audio_recording.recording_processing_finished()
        logger.info('recording parallel processing ({:.0f}m {:.0f}s)'.format((time() - timeAParallel) // 60, (time() - timeAParallel) % 60))

        # -- create list of vocals found in the recording
        list_of_vocals = ListOfVocals()
        list_of_vocals.combine_list_of_list_of_vocals(list_of_list_of_vocals=results)
        list_of_vocals.update_intervals()
        print(list_of_vocals)

        # -- update recording object and save data (images and excel)
        audio_recording._has_list_of_vocals = True
        audio_recording._list_of_vocals = list_of_vocals
        audio_recording.save_recording_object(path=audio_recording.output_dir)
        audio_recording.save_spectrograms_and_masks(path=audio_recording.output_dir)
        audio_recording.save_recording_data_to_excel(path=audio_recording.output_dir)
        audio_recording.remove_spectrograms_and_masks()
        audio_recording.save_recording_object(filename='recording_without_spectrograms', path=audio_recording.output_dir)

        logger.info('total time: {:.0f}m {:.0f}s'.format((time() - timeStart) // 60, (time() - timeStart) % 60))
