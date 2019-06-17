# -*- coding: utf-8 -*-
'''VocalPy Identifier - Finds candidate vocalizations in experimental recordings'''

__author__    = 'Gustavo Madeira Santana'
__email__     = 'gustavo.santana@yale.edu'
__copyright__ = '2019 Dietrich Lab - Yale University School of Medicine'

#ToDo
#Numba maybe

from utils import *

# import tkinter as tk
# from tkinter import filedialog

# # -- create dialog to ask for audio file
# root = tk.Tk()
# root.withdraw()
# file_path = filedialogger.askopenfilename()

p    = argparse.ArgumentParser()
p.add_argument('-v', '--verbose', help='output verbosity', action='store_true')
p.add_argument('-p', '--plot', help='plot after each operation', action='store_true')
args = p.parse_args()

dir_path = '/Users/gustavo/Documents/git/vocalpy/outputs/all/mask'
if not os.path.exists(dir_path):
    os.makedirs(dir_path, exist_ok=True)

if args.verbose:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)-5.5s]  %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S',
                        handlers=[
                            logging.FileHandler('{0}/{1}.log'.format('/Users/gustavo/Documents/git/vocalpy/outputs/', 'output')),
                            logging.StreamHandler()
                        ])
    logging.info('verbose output on')
else:
    logging.basicConfig(format='%(levelname)s: %(message)s')

logger = logging.getLogger()

timeStart = time()

file_path = '/Users/gustavo/Documents/git/vocalpy/audio_example.wav'
logger.info('selected file: {}'.format(file_path))

timeA = time()
sample_rate, samples = wavfile.read(file_path)
audio_duration       = samples.shape[0]/sample_rate
# -- rescale to be in the range (-1,1) so psd values match MATLAB's audioread
# -- change to be max possible number: 2^16/2 = 65536/2 = 32768
# samples = samples / np.max(samples)
timeB = time()
logger.info('audio duration: {:.2f} seconds'.format(audio_duration))
logger.info('load audio runtime: {:.2f}'.format(timeB - timeA))

# -- split audio in minute bins
bin_size = 60
bins     = ceil(audio_duration/60)
logger.info('splitting audio into {} bins'.format(bins))

# -- separate audio into chunks to be distributed to each process
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

# -- run one chunk in each available core
num_cores = multiprocessing.cpu_count()
results   = Parallel(n_jobs=num_cores)(delayed(parallel_audio_processing)(i) for i in chunks)

# -- concatenate results
vocal_df  = pd.concat(results)

# -- sort vocalizations by start time and save to excel
vocal_df.sort_values(by='start', ascending=True, inplace=True, kind='quicksort', na_position='last')
vocal_df.to_excel('/Users/gustavo/Documents/git/vocalpy/outputs/output.xlsx')

timeEnd   = time()
logger.info('total time: {:.2f}'.format(timeEnd - timeStart))