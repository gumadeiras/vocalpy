# -*- coding: utf-8 -*-
"""VocalPy Identifier - Finds candidate vocalizations in experimental recordings"""

__author__    = "Gustavo Madeira Santana"
__email__     = "gustavo.santana@yale.edu"
__copyright__ = "2019 Dietrich Lab - Yale University School of Medicine"

#ToDo
#Numba maybe

from utils import *

# import tkinter as tk
# from tkinter import filedialog

# # -- create dialog to ask for audio file
# root = tk.Tk()
# root.withdraw()
# file_path = filedialog.askopenfilename()

p    = argparse.ArgumentParser()
p.add_argument("-v", "--verbose", help="output verbosity", action="store_true")
p.add_argument("-p", "--plot", help="plot after each operation", action="store_true")
args = p.parse_args()
if args.verbose:
    log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
    log.info("verbose output on")
else:
    log.basicConfig(format="%(levelname)s: %(message)s")

timeStart = time()

file_path = '/Users/gustavo/Documents/git/vocalpy/audio_example.wav'
log.info("selected file: {}".format(file_path))

timeA = time()
sample_rate, samples = wavfile.read(file_path)
audio_duration       = samples.shape[0]/sample_rate
# -- rescale to be in the range (-1,1) so psd values match MATLAB's audioread
# samples = samples / np.max(samples)
timeB = time()
log.info("audio duration: {:.2f} seconds".format(audio_duration))
log.info("load audio runtime: {:.2f}".format(timeB - timeA))

# -- split audio in minute bins
bin_size = 60
bins     = ceil(audio_duration/60)
log.info("splitting audio into {} bins".format(bins))

dir_path = '/Users/gustavo/Documents/git/vocalpy/outputs/all/mask'
if not os.path.exists(dir_path):
    os.makedirs(dir_path, exist_ok=True)

chunks = []
for this_bin in range(1, bins+1):
    if this_bin == 1: # -- first bin
        start_range = ceil(0.3 * sample_rate)
        end_range   = bin_size * sample_rate
        time_range  = samples[start_range:end_range]
        chunks.append((sample_rate, time_range, this_bin, start_range, end_range, bin_size, args))
    elif this_bin == bins: # -- last bin
        start_range = (this_bin - 1) * bin_size * sample_rate
        end_range   = audio_duration * sample_rate
        time_range  = samples[start_range:]
        chunks.append((sample_rate, time_range, this_bin, start_range, end_range, bin_size, args))
    else: # -- all other bins
        start_range = (this_bin - 1) * bin_size * sample_rate
        end_range   = this_bin * bin_size * sample_rate
        time_range  = samples[start_range:end_range]
        chunks.append((sample_rate, time_range, this_bin, start_range, end_range, bin_size, args))
 
num_cores = multiprocessing.cpu_count()

results  = Parallel(n_jobs=num_cores)(delayed(parallel_audio_processing)(i) for i in chunks)
# log.info(results)
vocal_df = pd.concat(results)
vocal_df.sort_values(by='start', ascending=True, inplace=True, kind='quicksort', na_position='last')
vocal_df.to_excel('/Users/gustavo/Documents/git/vocalpy/outputs/output.xlsx')

timeEnd = time()
log.info("total time: {:.2f}".format(timeEnd - timeStart))