# -*- coding: utf-8 -*-
'''
VocalPy - A python version of (VocalMat by Antonio Fonseca)
'''

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
utils.save_file(audio_recording, audio_recording.output_dir)
timeB           = time()
logger.info('recording object created ({:.2f}s) and saved to: "{}"'.format((timeB - timeStart), audio_recording.output_dir))

logger.info('audio duration: {:.2f} seconds'.format(audio_recording.recording_duration))
logger.info('splitting audio into {} chunks'.format(audio_recording.bins))

# -- run one chunk in each available core
if args.threads > 0 :
    num_cores = args.threads
else:
    num_cores = multiprocessing.cpu_count()

results   = Parallel(n_jobs=num_cores, require='sharedmem')(delayed(utils.parallel_audio_processing)(i) for i in audio_recording.chunks)

# -- concatenate results
vocal_df  = pd.concat(results)

# -- sort vocalizations by start time and save to excel
vocal_df.sort_values(by='start', ascending=True, inplace=True, kind='quicksort', na_position='last')
vocal_df.to_excel(os.path.join(audio_recording.output_dir, 'vocal_stats.xlsx'))

timeEnd   = time()
logger.info('total time: {:.2f}s'.format(timeEnd - timeStart))