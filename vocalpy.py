# -*- coding: utf-8 -*-
'''VocalPy - A python version based on (VocalMat by Antonio Fonseca)'''

__author__    = 'Gustavo Madeira Santana'
__email__     = 'gustavo.santana@yale.edu'
__copyright__ = '2019 Dietrich Lab - Yale University School of Medicine'

#ToDo
#Numba maybe

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
p.add_argument('-p', '--plot', help='plot sample spectrogram after each operation', action='store_true')
p.add_argument('-b', '--bin_size', help='bin size in seconds to split spectrogram processing', type=int, default=60)
p.add_argument('-t', '--threads', help='number of threads', type=int, default=0)
args = p.parse_args()

root_dir = '/Users/gustavo/Documents/git/vocalpy'
out_dir  = os.path.join(root_dir, 'outputs')
audio_f  = os.path.join(root_dir, 'audio_example.wav')

if not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)

utils.create_logger(args, out_dir)
logger = logging.getLogger()
logger.info('selected file: {}'.format(audio_f))

timeStart       = time()
audio_recording = Recording(recording_path=audio_f, args=args)
audio_recording.save_recording_object(path=audio_recording.output_dir)
timeBRecording  = time()

logger.info('recording object created ({:.2f}s) and saved to: "{}"'.format((timeBRecording - timeStart), audio_recording.output_dir))
logger.info('audio duration: {:.2f} seconds'.format(audio_recording.recording_duration))
logger.info('splitting audio into {} chunks'.format(audio_recording.bins))

# -- get core count for parallelization
if args.threads > 0 :
    num_cores = args.threads
else:
    num_cores = multiprocessing.cpu_count()

# -- distribute recording chuncks to available cores
timeAParallel = time()
results       = Parallel(n_jobs=num_cores, require='sharedmem')(delayed(utils.parallel_audio_processing)(i) for i in audio_recording.chunks)
audio_recording.recording_processing_finished()
timeBParallel = time()
logger.info('recording parallel processing ({:.2f}s)'.format(timeBParallel - timeAParallel))

# -- create list of vocals found in the recording
list_of_vocals = ListOfVocals()
list_of_vocals.combine_list_of_list_of_vocals(list_of_list_of_vocals=results)
list_of_vocals.update_intervals()
list_of_vocals.update_centroids_and_spectrograms()
list_of_vocals.save_list_of_vocals_object(path=audio_recording.output_dir)
print(list_of_vocals)

audio_recording.has_list_of_vocals = True
audio_recording.save_recording_object(path=audio_recording.output_dir)
audio_recording.save_recording_data_to_excel(list_of_vocals=list_of_vocals)

timeEnd   = time()
logger.info('total time: {:.2f}s'.format(timeEnd - timeStart))
