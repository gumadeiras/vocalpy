# -*- coding: utf-8 -*-
'''VocalPy - A python version based on VocalMat'''

__email__     = 'gustavo.santana@yale.edu'
__license__   = 'Apache License, Version 2.0'
__copyright__ = '2019 Dietrich Lab - Yale University School of Medicine'

#ToDo
#Numba maybe
#make sure masks only have that vocal's segmentation (zero out the rest)

import os
import utils
import argparse
import logging
import multiprocessing

import pandas          as     pd
from   time            import time
from   recording       import Recording
from   list_of_vocals  import ListOfVocals
from   joblib          import Parallel, delayed

# import tkinter as tk
# from tkinter import filedialog

# # -- create dialog to ask for audio file
# root = tk.Tk()
# root.withdraw()
# file_path = filedialogger.askopenfilename()

p    = argparse.ArgumentParser()
p.add_argument('-v', '--verbose', help='output verbosity', action='store_true')
p.add_argument('-a', '--audio_path', help='audio file name', type=str, default='/Users/gustavo/Documents/git/vocalpy/audio_example.wav')
p.add_argument('-p', '--plot', help='plot sample spectrogram after each operation', action='store_true')
p.add_argument('-b', '--bin_size', help='bin size in seconds to split spectrogram processing', type=int, default=60)
p.add_argument('-t', '--threads', help='number of threads', type=int, default=0)
args = p.parse_args()

audio_path = args.audio_path

# -- ToDo:
# -- if its a dir, run all audios in that dir
# -- if its a file, run that audio only
if os.path.isdir(audio_path):
    print('path is a directory')
elif os.path.isfile(audio_path):
    print('path is a file')
else:
	print('path is not a file or directory: {}'.format(audio_path))

output_dir = os.path.join(audio_path[0:-4] + '_outputs')

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

utils.create_logger(args, output_dir)
logger = logging.getLogger()
logger.info('selected file: {}'.format(audio_path))

timeStart       = time()
audio_recording = Recording(recording_path=audio_path, args=args)
# audio_recording.save_recording_object(path=audio_recording.output_dir)
timeBRecording  = time()

# logger.info('recording object created ({:.2f}s) and saved to: {}.format((timeBRecording - timeStart), audio_recording.output_dir))
print(audio_recording)
logger.info('recording object created ({:.2f}s)'.format((timeBRecording - timeStart)))
logger.info('audio duration: {:.2f} seconds'.format(audio_recording.recording_duration))
logger.info('splitting audio into {} chunks'.format(audio_recording.bins))

# -- get core count for parallelization
if args.threads > 0:
    num_cores = args.threads
else:
    num_cores = multiprocessing.cpu_count()

# -- distribute recording chuncks to available cores
timeAParallel = time()
results       = Parallel(n_jobs=num_cores, require='sharedmem')(delayed(utils.parallel_audio_processing)(i) for i in audio_recording.chunks)
audio_recording.recording_processing_finished()
timeBParallel = time() - timeAParallel
logger.info('recording parallel processing ({:.0f}m {:.0f}s)'.format(timeBParallel//60, timeBParallel%60))

# -- create list of vocals found in the recording
list_of_vocals = ListOfVocals()
list_of_vocals.combine_list_of_list_of_vocals(list_of_list_of_vocals=results)
list_of_vocals.update_intervals()
print(list_of_vocals)

# -- update recording object and save data (images and excel)
audio_recording._has_list_of_vocals = True
audio_recording._list_of_vocals     = list_of_vocals
audio_recording.save_recording_object(path=audio_recording.output_dir)
audio_recording.save_spectrograms_and_masks(path=audio_recording.output_dir)
audio_recording.save_recording_data_to_excel(path=audio_recording.output_dir)
audio_recording.remove_spectrograms_and_masks()
audio_recording.save_recording_object(filename='recording_wo_spectrograms', path=audio_recording.output_dir)

timeEnd   = time() - timeStart
logger.info('total time: {:.0f}m {:.0f}s'.format(timeEnd//60,timeEnd%60))
