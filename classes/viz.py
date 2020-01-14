# -*- coding: utf-8 -*-
'''VocalPy - A python version based on VocalMat'''

__email__ = 'gustavo.santana@yale.edu'
__license__ = 'Apache License, Version 2.0'
__copyright__ = '2020 Dietrich Lab - Yale University School of Medicine'

import os
import math

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from utils.io import load_recording_data


class Viz(object):
    '''
    visualization object for plotting data from recordings
    '''

    def __init__(self,
                 list_of_groups,
                 group_names=None,
                 bin_size=1):

        if len(list_of_groups) < 1:
            print('please provide at least one recording for analysis.')
            exit()

        self._list_of_groups = list_of_groups
        self._bin_size = bin_size

        if group_names is None:
            self._group_names = self.generate_group_names()
        else:
            self._group_names = np.asarray(group_names)

        self._number_of_groups = self._group_names.shape[0]
        self._list_of_viz = None
        self._list_of_group_viz = None

        self.create_viz_for_each_recording()
        self.create_group_viz()

    def generate_group_names(self):
        group_names = []
        for group in self.list_of_groups:
            for recording in group:
                # -- get recording name and remove trailing '_outputs' (-8)
                recording_name = os.path.basename(os.path.split(recording)[0])[0:-8]
                group_names.append(recording_name)
        self._group_names = np.asarray(group_names)

    def combine_viz_dataframes(self, list_of_viz=None):
        if list_of_viz is None:
            print('please provide a list_of_viz')
            return -1

        dfs = list_of_viz[0]._recording_df
        for i, viz in enumerate(list_of_viz):
            if i == 0:
                continue
            dfs = pd.concat([dfs, viz._recording_df], axis=0)

        return dfs.sort_values('bin_number')

    def combine_viz(self, list_of_viz, groupname):
        if len(list_of_viz):
            combined_df = self.combine_viz_dataframes(list_of_viz)
            list_of_viz[0]._recording_df = combined_df
            list_of_viz[0]._recording_path = None
            list_of_viz[0]._list_of_vocals = None
            list_of_viz[0]._bin_size = self._bin_size
            number_of_bins_in_df = combined_df['bin_number'].max()
            list_of_viz[0]._bins = int(math.floor(number_of_bins_in_df / self._bin_size)) + 1

            list_of_viz[0]._recording_data.group_name = groupname
            list_of_viz[0]._recording_data._list_of_vocals.number_of_vocals = np.sum([viz._recording_data._list_of_vocals.number_of_vocals for viz in list_of_viz])

            list_of_viz[0]._split_array = list_of_viz[0].get_split_indices_for_bin_size()

            return list_of_viz[0]
        else:
            return []

    def create_viz_for_each_recording(self):
        list_of_viz = []

        for group in range(self._number_of_groups):
            paths = self._list_of_groups[group]
            vizobjs = [SingleViz(recording_path=rec_path, bin_size=self._bin_size) for rec_path in paths]
            list_of_viz.append(vizobjs)

        self._list_of_viz = np.asarray(list_of_viz)
        return 0

    def create_group_viz(self):
        self._list_of_group_viz = [self.combine_viz(group, self._group_names[i]) for i, group in enumerate(self._list_of_viz)]
        return 0

    def plot(self, plot_type='group', dataname='avg_freq'):
        if plot_type not in ['group', 'individual']:
            print('type must be \"group\" or \"individual\".')
            exit()

        if dataname not in ['raster','rate','min_freq', 'max_freq',
                             'avg_freq', 'duration', 'bandwidth', 'min_intensity',
                             'max_intensity', 'avg_intensity', 'area']:
            print('possible datapoints are:')
            print('raster, rate, min_freq, max_freq, avg_freq, duration, bandwidth, min_intensity, max_intensity, avg_intensity, area.')
            exit()

        if plot_type is 'group':
            self.plot_group(dataname)
        else:
            self.plot_individual(dataname)

        return 0

    def plot_individual(self, dataname):
        for group in self._list_of_viz:
            for viz in group:
                if dataname is 'raster':
                    viz.rugplot()
                else:
                    viz.pointplot(dataname=dataname)
        return 0

    def plot_group(self, dataname):
        if dataname is 'raster':
            for group in self._list_of_group_viz:
                group.rugplot()
        else:
            self.group_pointplot(dataname=dataname)
        return 0

    def group_pointplot(self, dataname='avg_freq'):
        sns.set(style='whitegrid', palette='muted', color_codes=True)
        f, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=150)
        cmap = sns.color_palette()
        font_size = 16
        labels = []
        custom_lines = []
        for i, group in enumerate(self._list_of_group_viz):
            if group:
                data_values = group.get_datapoints(dataname)
                sns.pointplot(data=data_values, color=cmap[i], capsize=.1, label=group._recording_data.group_name)
                custom_lines.append(Line2D([0], [0], color=cmap[i], lw=2))
                labels.append(group._recording_data.group_name)
                ax.set_ylabel(dataname, fontsize=font_size)
                ax.set_xlabel('Bin', fontsize=font_size)
                ax.set_xlim(-1, 11)
                ax.set_xticks(list(range(11)))
                ax.tick_params(labelsize=font_size)
                plt.tight_layout()

        ax.legend(custom_lines, labels, fontsize=font_size)


class SingleViz(object):
    '''
    visualization object for a single recording
    '''

    def __init__(self,
                 recording_path=None,
                 bin_size=1):

        if recording_path is None:
            print('please provide a recording_path')
            exit()

        self._recording_path = recording_path
        self._recording_data = load_recording_data(self._recording_path)
        self._list_of_vocals = self.create_list_of_vocals()
        self._recording_df = self.create_object_dataframe()
        self._bin_size = bin_size
        self._duration = self._recording_data.recording_duration / 60
        self._bins = int(math.floor(self._duration / self._bin_size)) + 1
        self._split_array = self.get_split_indices_for_bin_size()

    def __str__(self):
        return '{}:\n recording path: {} \n recording duration: {:.2f} \n bin size: {} \n bins: {} \n vocals in recording: {}'.format(self.__class__.__name__, self._recording_path, self._duration, self._bin_size, self._bins, self._recording_data._list_of_vocals.number_of_vocals)

    def create_list_of_vocals(self):
        return self._recording_data._list_of_vocals

    def create_object_dataframe(self):
        recording_df = pd.DataFrame(columns=['bin_number',
                                             'start',
                                             'end',
                                             'duration',
                                             'interval',
                                             'min_freq',
                                             'max_freq',
                                             'avg_freq',
                                             'bandwidth',
                                             'min_intensity',
                                             'max_intensity',
                                             'avg_intensity',
                                             'bg_intensity',
                                             'area',
                                             'centroid_y'])

        for vocal in iter(self._list_of_vocals.vocals_in_recording):
            recording_df = recording_df.append({'bin_number': vocal.bin_number,
                                                'start': vocal.start,
                                                'end': vocal.end,
                                                'duration': vocal.duration,
                                                'interval': vocal.interval,
                                                'min_freq': vocal.min_freq,
                                                'max_freq': vocal.max_freq,
                                                'avg_freq': vocal.avg_freq,
                                                'bandwidth': vocal.bandwidth,
                                                'min_intensity': vocal.min_intensity,
                                                'max_intensity': vocal.max_intensity,
                                                'avg_intensity': vocal.avg_intensity,
                                                'bg_intensity': vocal.bg_intensity,
                                                'area': vocal.area,
                                                'centroid_y': vocal.centroid[0]},
                                                ignore_index=True)

        recording_df.index = np.arange(1, len(recording_df) + 1)
        return recording_df

    def get_split_indices_for_bin_size(self):
        split_array = []
        bin_column = self._recording_df[['bin_number']]
        for idx in range(1, self._bins):
            try:
                split_array.append(np.where(bin_column == idx * self._bin_size)[0][-1] + 1)
            except:
                print('recording {} had no vocals in bin {}'.format(self._recording_data.recording_name, idx))
        return split_array

    def split_data_by_indices(self, dataframe):
        return np.asarray(np.split(dataframe, self._split_array))

    def get_count_in_list_of_lists(self, dataframe):
        return np.asarray([frame.shape[0] for frame in dataframe])

    def get_mean_in_list_of_lists(self, dataframe):
        return np.asarray([frame.mean() for frame in dataframe])

    def get_data_column(self, dataname):
        return np.hstack(self._recording_df[[dataname]].to_numpy())

    def get_datapoints(self, dataname):
        data_column = self.get_data_column(dataname)
        data_split = self.split_data_by_indices(data_column)
        data_values = pd.DataFrame([data for data in data_split]).T

        # -- shift row and columns to start at 1 instead of 0
        data_values.index = np.arange(1, len(data_values) + 1)
        data_values.columns = np.arange(1, len(data_values.columns) + 1)

        return data_values

    def rugplot(self, dataname='start'):
        # -- visualize vocalizations throughtout the recording using a rug plot
        data_values = self._recording_df[[dataname]] / 60

        sns.set(style='white', palette='muted', color_codes=True)

        f, ax = plt.subplots(1, 1, figsize=(12, 2), dpi=300)
        sns.rugplot(data_values, height=1, color='coral', alpha=0.5)

        ax.set_title(self._recording_data.group_name)
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel(dataname)
        ax.set_yticks([])
        plt.tight_layout()

        return data_values

    def pointplot(self, dataname='avg_freq'):
        # -- visualize vocalization data throughtout the recording using a pointplot

        data_values = self.get_datapoints(dataname)

        sns.set(style='whitegrid', palette='muted', color_codes=True)
        f, ax = plt.subplots(1, 1, figsize=(12, 7), dpi=300)

        sns.pointplot(data=data_values, color='coral', capsize=.1)

        ax.set_ylabel(dataname)
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

    def violinplot(self, dataname='avg_freq', inner='box'):
        # -- visualize vocalization data throughtout the recording using a violin plot

        data_values = self.get_datapoints(dataname)

        sns.set(style='whitegrid', palette='muted', color_codes=True)
        f, ax = plt.subplots(1, 1, figsize=(12, 7), dpi=300)

        sns.violinplot(data=data_values, color='coral', inner=inner)

        ax.set_ylabel(dataname)
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
