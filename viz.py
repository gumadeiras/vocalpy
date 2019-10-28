# -*- coding: utf-8 -*-
'''VocalPy - A python version based on VocalMat'''

__email__     = 'gustavo.santana@yale.edu'
__license__   = 'Apache License, Version 2.0'
__copyright__ = '2019 Dietrich Lab - Yale University School of Medicine'

import os
import math

import numpy             as np
import pandas            as pd
import seaborn           as sns
import matplotlib.pyplot as plt
import PIL.Image         as Image



class Viz(object):
    '''
    visualization object
    '''
    def __init__(self,
                 recording_path = None,
                 bin_size       = 1):

        if recording_path is None:
            print("please provide a recording_path")
            return -1

        self._recording_path = recording_path
        self._recording_data = self.load_recording_data()
        self._list_of_vocals = self.create_list_of_vocals()
        self._recording_df   = self.create_object_dataframe()
        self._bin_size       = bin_size
        self._duration       = self._recording_data.recording_duration / 60
        self._bins           = int(math.floor(self._duration / self._bin_size)) + 1
        self._split_array    = self.get_split_indices_for_bin_size()

    def __str__(self):
        return "{}:\n recording path: {} \n recording duration: {:.2f} \n bin size: {} \n bins: {} \n vocals in recording: {}".format(self.__class__.__name__, self._recording_path, self._duration, self._bin_size, self._bins, self._recording_data._list_of_vocals.number_of_vocals)

    def load_recording_data(self):
        return np.load(self._recording_path, allow_pickle=True)

    def create_list_of_vocals(self):
        return self._recording_data._list_of_vocals

    def create_object_dataframe(self):
        recording_df = pd.DataFrame(columns = ['bin_number', 'start', 'end', 'duration', 'interval', 'min_freq', 'max_freq', 'avg_freq', 'bandwidth', 'min_intensity', 'max_intensity', 'avg_intensity', 'bg_intensity', 'area', 'centroid'])
        
        for vocal in iter(self._list_of_vocals.vocals_in_recording):
            recording_df = recording_df.append({'bin_number': vocal.bin_number, 'start': vocal.start, 'end': vocal.end, 'duration': vocal.duration, 'interval': vocal.interval, 'min_freq': vocal.min_freq, 'max_freq': vocal.max_freq, 'avg_freq': vocal.avg_freq, 'bandwidth': vocal.bandwidth, 'min_intensity': vocal.min_intensity, 'max_intensity': vocal.max_intensity, 'avg_intensity': vocal.avg_intensity, 'bg_intensity': vocal.bg_intensity, 'area': vocal.area, 'centroid': vocal.centroid}, ignore_index=True)
        
        recording_df.index = np.arange(1, len(recording_df)+1)
        
        return recording_df

    def get_split_indices_for_bin_size(self):
        split_array = []
        dataframe = self._recording_df[['bin_number']]
        for idx in range(1, self._bins):
            split_array.append(np.where(dataframe==idx*self._bin_size)[0][-1]+1)
        
        return split_array

    def split_data_by_indices(self, dataframe):
        return np.asarray(np.split(dataframe, self._split_array))

    def get_count_in_list_of_lists(self, dataframe):
        return np.asarray([frame.shape[0] for frame in dataframe])

    def get_mean_in_list_of_lists(self, dataframe):
        return np.asarray([frame.mean() for frame in dataframe])

    def get_data_column(self, dataname):
        return np.hstack(self._recording_df[[dataname]].to_numpy())

    def rugplot(self, dataname='start'):
        # -- visualize vocalizations throughtout the recording using a rug plot
        data_values = self._recording_df[[dataname]]/60

        sns.set(style="white", palette="muted", color_codes=True)

        f, ax = plt.subplots(1, 1, figsize=(12, 2), dpi=300)
        sns.rugplot(data_values, height=1, color="coral", alpha=0.5)
        
        plt.title(dataname + " | " + self._recording_data.recording_name)
        
        ax.set_xlabel('Time (minutes)')
        ax.set_yticks([])
        plt.tight_layout()

        return data_values

    def scatter_count(self, dataname='vocal_rate'):
        # -- visualize count vocalization data throughtout the recording using a line plot

        if dataname == 'vocal_rate':
            data_column = self.get_data_column('start')
        else:
            data_column = self.get_data_column(dataname)

        data_split  = self.split_data_by_indices(data_column)
        data_values  = self.get_count_in_list_of_lists(data_split)

        sns.set(style="whitegrid", palette="muted", color_codes=True)
        f, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=300)
        sns.scatterplot(y=data_values, x=range(1, self._bins+1), color="coral")

        plt.title(dataname + " | " + self._recording_data.recording_name)
        ax.set_xlabel('Bin')
        ax.set_xlim(0, self._bins+1)
        ax.set_xticks(range(1, self._bins+1))
        plt.tight_layout()

        return data_values


    def scatter_mean(self, dataname='duration'):
        # -- visualize mean vocalization data throughtout the recording using a line plot

        data_column = self.get_data_column(dataname)
        data_split  = self.split_data_by_indices(data_column)
        data_values  = self.get_mean_in_list_of_lists(data_split)

        sns.set(style="whitegrid", palette="muted", color_codes=True)
        f, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=300)
        sns.scatterplot(y=data_values, x=range(1, self._bins+1), color="coral")

        plt.title(dataname + " | " + self._recording_data.recording_name)
        ax.set_xlabel('Bin')
        ax.set_xlim(0, self._bins+1)
        ax.set_xticks(range(1, self._bins+1))
        plt.tight_layout()

        return data_values

    def lineplot_count(self, dataname='vocal_rate'):
        # -- visualize count vocalization data throughtout the recording using a line plot

        if dataname == 'vocal_rate':
            data_column = self.get_data_column('start')
        else:
            data_column = self.get_data_column(dataname)

        data_split  = self.split_data_by_indices(data_column)
        data_values  = self.get_count_in_list_of_lists(data_split)

        sns.set(style="whitegrid", palette="muted", color_codes=True)
        f, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=300)
        sns.lineplot(y=data_values, x=range(1, self._bins+1), color="coral")

        plt.title(dataname + " | " + self._recording_data.recording_name)
        ax.set_xlabel('Bin')
        ax.set_xlim(0, self._bins+1)
        ax.set_xticks(range(1, self._bins+1))
        plt.tight_layout()

        return data_values

    def lineplot_mean(self, dataname='duration'):
        # -- visualize mean vocalization data throughtout the recording using a line plot

        data_column = self.get_data_column(dataname)
        data_split  = self.split_data_by_indices(data_column)
        data_values  = self.get_mean_in_list_of_lists(data_split)

        sns.set(style="whitegrid", palette="muted", color_codes=True)
        f, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=300)
        sns.lineplot(y=data_values, x=range(1, self._bins+1), color="coral")

        plt.title(dataname + " | " + self._recording_data.recording_name)
        ax.set_xlabel('Bin')
        ax.set_xlim(0, self._bins+1)
        ax.set_xticks(range(1, self._bins+1))
        plt.tight_layout()

        return data_values

    def violinplot(self, dataname='avg_freq', inner='box'):
        # -- visualize vocalization data throughtout the recording using a violin plot

        data_column = self.get_data_column(dataname)
        data_split  = self.split_data_by_indices(data_column)
        data_values = pd.DataFrame([data for data in data_split]).T
        
        # -- shift row and columns to start at 1 instead of 0
        data_values.index   = np.arange(1, len(data_values)+1)
        data_values.columns = np.arange(1, len(data_values.columns)+1)

        sns.set(style="whitegrid", palette="muted", color_codes=True)
        f, ax = plt.subplots(1, 1, figsize=(12, 7), dpi=300)

        sns.violinplot(data=data_values, color="coral", inner=inner)

        plt.title(dataname + " | " + self._recording_data.recording_name)
        ax.set_xlabel('Bin')
        ax.set_xlim(-1, self._bins)
        
        if dataname[-4:] == 'freq':
            ax.set_ylim(40000, 125001)
            ax.set_yticks(range(40000, 125001, 17000))
        else:
            ax.set_ylim(0)

        # ax.set_ylabel(object_data.recording_name[0:4], rotation=90)
        plt.tight_layout()

        return data_values