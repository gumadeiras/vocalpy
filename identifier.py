# -*- coding: utf-8 -*-
"""VocalPy Identifier - Finds candidate vocalizations in experimental recordings"""

__author__    = "Gustavo Madeira Santana"
__email__     = "gustavo.santana@yale.edu"
__copyright__ = "2019 Dietrich Lab - Yale University School of Medicine"

#ToDo
#Numba maybe

from utils import *

import argparse
import logging as log


p    = argparse.ArgumentParser()
p.add_argument("-v", "--verbose", help="output verbosity", action="store_true")
p.add_argument("-p", "--plot", help="output verbosity", action="store_true")
args = p.parse_args()
if args.verbose:
    log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
    log.info("verbose output on")
else:
    log.basicConfig(format="%(levelname)s: %(message)s")
# import tkinter as tk
# from tkinter import filedialog

# # -- create dialog to ask for audio file
# root = tk.Tk()
# root.withdraw()
# file_path = filedialog.askopenfilename()

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

# -- split spectrogram in minute bins
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
        chunks.append((time_range, this_bin, start_range, end_range, bin_size, args))
    elif this_bin == bins: # -- last bin
        start_range = (this_bin - 1) * bin_size * sample_rate
        end_range   = audio_duration * sample_rate
        time_range  = samples[start_range:]
        chunks.append((time_range, this_bin, start_range, end_range, bin_size, args))
    else: # -- all other bins
        start_range = (this_bin - 1) * bin_size * sample_rate
        end_range   = this_bin * bin_size * sample_rate
        time_range  = samples[start_range:end_range]
        chunks.append((time_range, this_bin, start_range, end_range, bin_size, args))

from joblib import Parallel, delayed
import multiprocessing
 
num_cores = multiprocessing.cpu_count()

def parallel_spectrogram(chunk):
    time_range, this_bin, start_range, end_range, bin_size, args = chunk

    if args.verbose:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    timeA           = time()
    fs              = sample_rate
    window          = signal.get_window('hamming', 256)
    noverlap        = 128
    nfft            = 1024
    time_range_secs = time_range.shape[0] / sample_rate
    log.info("computing spectrogram for bin: {}; time range: {}s; audio range: {:.2f}-{:.2f}s".format(this_bin,
                                                                                                   time_range_secs,
                                                                                                   start_range / sample_rate,
                                                                                                   end_range / sample_rate))
    f, t, Sxx  = signal.spectrogram(time_range, fs=fs,
                                                window=window,
                                                noverlap=noverlap,
                                                nfft=nfft,
                                                mode='psd')
    # log.info(t.shape)
    # log.info(f.shape)

    # -- remove lower frequencies
    freq_cutoff = 45000
    Sxx         = Sxx[(f>freq_cutoff)]
    f           = f[(f>freq_cutoff)]
    # log.info(Sxx.shape)
    # log.info(np.min(Sxx))
    # log.info(np.max(Sxx))

    time_res = time_range_secs/t.shape[0]
    freq_res = (np.max(f) - freq_cutoff) / f.shape[0]
    timeB    = time()
    log.info("spectrogram runtime: {:.2f}".format(timeB - timeA))
    log.info("time resolution: {:.2f}ms".format(time_res * 1000))
    log.info("freq resolution: {:.2f}Hz".format(freq_res))


    # -- convert to dB
    Pxx        = 10*np.log10(Sxx)
    # log.info(np.min(Pxx))
    # log.info(np.max(Pxx))
    if args.plot:
        plt.pcolormesh(t[35000:40000], f, Pxx[:,35000:40000], cmap='gray')
        plt.title('Pxx (dB)')
        plt.show()

    # -- rescale to save spectrograms in grayscale
    dtype      = np.uint8
    Pxx_scaled = exposure.rescale_intensity(Pxx, in_range='image', out_range=dtype)

    # -- normalize data
    B = np.abs(Pxx)
    B = B/np.max(B)
    if args.plot:
        plt.pcolormesh(t[35000:40000], f, B[:,35000:40000], cmap='gray')
        plt.title('B = Pxx normalized')
        plt.show()

    # -- rescale intensity values to be between (0,1)
    # B = exposure.rescale_intensity(B, in_range='image', out_range=(0,1))
    B = exposure.rescale_intensity(B, in_range='image', out_range=dtype)
    if args.plot:
        plt.pcolormesh(t[35000:40000], f, B[:,35000:40000], cmap='gray')
        plt.title('B intensity values rescaled to dtype')
        plt.show()

    # -- image complement
    ii8    = np.iinfo(dtype)
    B_orig = B
    B      = ii8.max - B
    if args.plot:
        plt.pcolormesh(t[35000:40000], f, B[:,35000:40000], cmap='gray')
        plt.title('B complement')
        plt.show()


    # -- contrast adjustment
    B = exposure.adjust_gamma(B)
    if args.plot:
        plt.pcolormesh(t[35000:40000], f, B[:,35000:40000], cmap='gray')
        plt.title('B contrast adjustment')
        plt.show()

    # -- binarize spectrogram
    BB = bradley_roth_numpy(B, t=10)
    BB = ii8.max - BB
    if args.plot:
        plt.pcolormesh(t[35000:40000], f, BB[:,35000:40000], cmap='gray')
        plt.title('BB = B binarized bradley roth')
        plt.show()

    # -- kernels for morphological operations
    kernel_rect  = np.ones((4,2), np.uint8)
    kernel_line1 = np.ones((4,1), np.uint8)
    kernel_line2 = np.ones((5,1), np.uint8)

    # -- morphological operations
    erode11  = cv2.erode(BB, kernel_line1, iterations=1)
    dilate12 = cv2.dilate(erode11, kernel_rect, iterations=1)
    dilate13 = cv2.dilate(dilate12, kernel_line2, iterations=1)
    erode14  = cv2.erode(dilate13, kernel_line1, iterations=2)
    if args.plot:
        plt.subplot(511)
        plt.pcolormesh(t[80000:85000], f, BB[:,80000:85000], cmap='gray')
        plt.title('BB')
        plt.subplot(512)
        plt.pcolormesh(t[80000:85000], f, erode11[:,80000:85000], cmap='gray')
        plt.title('BB + erode (4,1)')
        plt.subplot(513)
        plt.pcolormesh(t[80000:85000], f, dilate12[:,80000:85000], cmap='gray')
        plt.title('+ dilate (4,2)')
        plt.subplot(514)
        plt.pcolormesh(t[80000:85000], f, dilate13[:,80000:85000], cmap='gray')
        plt.title('+ dilate (5,1)')
        plt.subplot(515)
        plt.pcolormesh(t[80000:85000], f, erode14[:,80000:85000], cmap='gray')
        plt.title('+ erode (4,1)')
        plt.show()

    timeA  = time()
    connectivity = 4
    num_cc, output, stats, centroids = cv2.connectedComponentsWithStats(erode14, connectivity, cv2.CV_32S)

    # -- remove background stats
    num_cc = num_cc - 1
    areas  = stats[1:,4]
    # nr     = np.arange(num_cc)
    # ranked = sorted(zip(areas,nr))

    # -- filtered connected components placeholder
    grain = np.zeros((output.shape))

    # -- threshold connected components by minimum area
    min_area = 20
    for i in range(0, num_cc):
        if areas[i] >= min_area:
            grain[output == i + 1] = 255
    timeB = time()
    log.info("connected components runtime: {:.2f}".format(timeB - timeA))

    # -- one more opening to make sure segmentation covers *at least* the real area
    grain        = grain.astype(np.uint8)
    kernel_cross = np.array([[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]], dtype=np.uint8)
    kernel_line3 = np.ones((1,3), dtype=np.uint8)
    grain        = cv2.dilate(grain, kernel_line3, iterations=1)

    # -- get spectrogram area using the segmentation mask
    B_masked = B * (((255 - grain) > 0) * 1)
    if args.plot:
        plt.subplot(411)
        plt.pcolormesh(t[80000:85000], f, Pxx[:,80000:85000], cmap='gray')
        # plt.pcolormesh(t, f, Pxx, cmap='gray')
        plt.title('Pxx')
        plt.subplot(412)
        plt.pcolormesh(t[80000:85000], f, BB[:,80000:85000], cmap='gray')
        # plt.pcolormesh(t, f, BB, cmap='gray')
        plt.title('BB')
        plt.subplot(413)
        plt.pcolormesh(t[80000:85000], f, grain[:,80000:85000], cmap='gray')
        # plt.pcolormesh(t, f, grain, cmap='gray')
        plt.title('grain')
        plt.subplot(414)
        plt.pcolormesh(t[80000:85000], f, B_masked[:,80000:85000], cmap='gray')
        # plt.pcolormesh(t, f, B_masked, cmap='gray')
        plt.title('B_masked')
        plt.show()

    # -- get connected components stats
    timeA     = time()
    labels = measure.label(grain, background=0)
    props  = measure.regionprops(labels, intensity_image=Pxx, cache=True, coordinates='rc')
    props  = sorted(props, key=lambda p: np.min(p.coords[:,1]), reverse=False)
    timeB    = time()
    log.info("region props runtime: {:.2f}".format(timeB - timeA))
    # plt.subplot(311)
    # plt.pcolormesh(t[35000:40000], f, Pxx[:,35000:40000], cmap='gray')
    # plt.subplot(312)
    # plt.pcolormesh(t[35000:40000], f, B[:,35000:40000], cmap='gray')
    # plt.subplot(313)
    # plt.pcolormesh(t[35000:40000], f, labels[:,35000:40000], cmap='nipy_spectral')
    # plt.show()

    # -- region props available
    # area int
    # bbox tuple
    # bbox_area int
    # centroid array
    # coords
    # max_intensity float
    # mean_intensity float
    # min_intensity float
    # orientation float

    vocal_id = 0
    vocal_df = pd.DataFrame(columns=['start',
                                     'end',
                                     'duration',
                                     'interval',
                                     # 'min_freq_main',
                                     # 'max_freq_main',
                                     # 'avg_freq_main',
                                     'min_freq_all',
                                     'max_freq_all',
                                     'avg_freq_all',
                                     'bandwidth',
                                     'min_intensity',
                                     'max_intensity',
                                     'avg_intensity',
                                     'bg_intensity',
                                     'area',
                                     'points',
                                     'centroid',
                                     'orientation',
                                     ])
    
    for prop in props:
        start = np.min(prop.coords[:,1])
        try:
            interval = np.abs(end - start)
        except:
            interval = 0

        end      = np.max(prop.coords[:,1])
        duration = end - start

        if duration < 5:
            continue

        min_freq_all = (np.min(prop.coords[:,0]) * freq_res) + freq_cutoff
        max_freq_all = (np.max(prop.coords[:,0]) * freq_res) + freq_cutoff
        avg_freq_all = (np.mean(prop.coords[:,0]) * freq_res) + freq_cutoff
        bandwidth    = max_freq_all - min_freq_all
        vocal_df     = vocal_df.append({'start': (start * time_res) + ((this_bin - 1) * bin_size),
                                        'end': (end * time_res) + ((this_bin - 1) * bin_size),
                                        'duration': duration * time_res * 1000,
                                        'interval': interval * time_res,
                                        'min_freq_all': min_freq_all,
                                        'max_freq_all': max_freq_all,
                                        'avg_freq_all': avg_freq_all,
                                        'bandwidth': bandwidth,
                                        'min_intensity': prop.min_intensity,
                                        'max_intensity': prop.max_intensity,
                                        'avg_intensity': prop.mean_intensity,
                                        'bg_intensity': 0,
                                        'area': prop.area,
                                        'points': prop.coords,
                                        'centroid': prop.centroid,
                                        'orientation': prop.orientation,
                                        }, ignore_index=True)

        # -- save spectrogram and mask around each vocalization
        centroid_time = ceil(prop.centroid[1])
        spectro_range = 200
        try:
            img = np.flipud(Pxx_scaled[:,centroid_time-200:centroid_time+200])
            img = Image.fromarray(img)
            img = img.convert("L")
            img.save('/Users/gustavo/Documents/git/vocalpy/outputs/all/' + str(this_bin) + '_' + str(vocal_id) + '.jpg')

            img = np.flipud(grain[:,centroid_time-200:centroid_time+200])
            img = Image.fromarray(img)
            img = img.convert("L")
            img.save('/Users/gustavo/Documents/git/vocalpy/outputs/all/mask/' + str(this_bin) + '_' + str(vocal_id) + '.jpg')

            img = np.flipud(B_masked[:,centroid_time-200:centroid_time+200])
            img = Image.fromarray(img)
            img = img.convert("L")
            img.save('/Users/gustavo/Documents/git/vocalpy/outputs/all/' + str(this_bin) + '_' + str(vocal_id) + '_overlay.jpg')
        except:
            log.info('######## EXCEPT HERE')
            log.info(centroid_time)
            log.info(Pxx.shape)
            img = np.flipud(Pxx_scaled[:,centroid_time-200:-1])
            img = Image.fromarray(img)
            img = img.convert("L")
            img.save('/Users/gustavo/Documents/git/vocalpy/outputs/all/' + str(this_bin) + '_' + str(vocal_id) + '.jpg')

            img = np.flipud(grain[:,centroid_time-200:-1])
            img = Image.fromarray(img)
            img = img.convert("L")
            img.save('/Users/gustavo/Documents/git/vocalpy/outputs/all/mask/' + str(this_bin) + '_' + str(vocal_id) + '.jpg')

            img = np.flipud(B_masked[:,centroid_time-200:-1])
            img = Image.fromarray(img)
            img = img.convert("L")
            img.save('/Users/gustavo/Documents/git/vocalpy/outputs/all/' + str(this_bin) + '_' + str(vocal_id) + '_overlay.jpg')
        vocal_id = vocal_id + 1

    return vocal_df
    # min_area_index = vocal_df[ vocal_df['duration'] >= 5 ].index
    # vocal_df.drop(min_area_index, inplace=True)

    # log.info("dataframe contens ", vocal_df, sep='\n')

results  = Parallel(n_jobs=num_cores)(delayed(parallel_spectrogram)(i) for i in chunks)
# log.info(results)
vocal_df = pd.concat(results)
vocal_df.sort_values(by='start', ascending=True, inplace=True, kind='quicksort', na_position='last')
vocal_df.to_excel('/Users/gustavo/Documents/git/vocalpy/outputs/output.xlsx')
# for each cc add to dict: time points, freq points, area, intensity, min,avg,max intensity, centroids,

# img = cv2.imread("letters.jpg", cv2.IMREAD_GRAYSCALE)

# kernel = np.ones((3,3), np.uint8)

# blur = cv2.GaussianBlur(img,(3,3), 0)
# # erosion = cv2.erode(blur, kernel, iterations=3)
# # opening = cv2.dilate(erosion, kernel)

# show(th3)

# kernel2 = cv2.getGaussianKernel(6, 2) #np.ones((6,6))
# kernel2 = np.outer(kernel2, kernel2)
# th3 = cv2.dilate(th3, kernel2)
# th3 = cv2.erode(th3, kernel)

timeEnd = time()
log.info("total time: {:.2f}".format(timeEnd - timeStart))