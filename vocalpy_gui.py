import webbrowser
from pyforms.basewidget import BaseWidget
from pyforms.controls import ControlDir
from pyforms.controls import ControlLabel
from pyforms.controls import ControlCombo
from pyforms.controls import ControlButton
from pyforms.controls import ControlVisVis
from pyforms.controls import ControlNumber
from pyforms.controls import ControlProgress
from pyforms.controls import ControlCheckBox
from pyforms.controls import ControlCheckBoxList


class VocalPy(BaseWidget):

    def __init__(self, *args, **kwargs):
        super().__init__('VocalPy')

        # Definition of the forms fields
        # tab: choose directory
        self._dir_load_directory = ControlDir('choose a directory')
        self._checkboxlist_fileList = ControlCheckBoxList('select files')
        self._checkbox_searchSubDirs = ControlCheckBox('search subdirectories')
        self._button_select_all = ControlButton('select all')
        self._button_select_none = ControlButton('unselect all')
        self._button_select_clear = ControlButton('clear list')
        self._button_more_info = ControlButton('http://dietrich-lab.org/vocalpy')
        #   # actions
        self._dir_load_directory.changed_event = self.__directorySelectionEvent
        self._checkbox_searchSubDirs.changed_event = self.__directorySelectionEvent
        self._button_select_all.value = self.__selectAllCheckbox
        self._button_select_none.value = self.__selectNoneCheckbox
        self._button_select_clear.value = self.__selectClearCheckbox
        self._button_more_info.value = self.__openURLButton

        # tab: select animal pipeline
        self._combo_animal_dropdown = ControlCombo('choose animal pipeline')
        self._combo_animal_dropdown.add_item('Mouse', 'mouse')
        self._combo_animal_dropdown.add_item('Rat', 'rat')
        self._combo_animal_dropdown.add_item('Guinea pig', 'guineapig')
        self._number_lo_freq_cutoff = ControlNumber('lower frequency cutoff (Hz)', default=45000, minimum=0, maximum=150000)
        self._number_hi_freq_cutoff = ControlNumber('higher frequency cutoff (Hz)', default=125000, minimum=0, maximum=150000)
        #   # actions
        self._combo_animal_dropdown.changed_event = self.__changeFrequencyAnimal

        # tab: optional settings
        self._number_thread_count = ControlNumber('number of threads for parallelization', default=2, minimum=1, maximum=40)
        self._checkbox_save_spectrograms = ControlCheckBox('save vocal spectrogram images', default=True)
        self._checkbox_save_segmentation = ControlCheckBox('save vocal segmentation images', default=False)
        self._checkbox_save_overlay = ControlCheckBox('save vocal segmentation overlay images', default=False)
        self._checkbox_save_plots = ControlCheckBox('generate sample plots', default=True)
        #   # actions

        # tab: run analysis
        self._button_run_analysis = ControlButton('run!')
        self._label_run_analysis = ControlLabel('#')
        self._progress_bar = ControlProgress('Processing audio files')
        #   # actions
        self._progress_bar.min = 0
        self._progress_bar.max = self.__numberOfAudios()
        self._button_run_analysis.value = self.__runVocalPy

        # tab: inspect vocalizations
        self._dir_load_vocalpy_file = ControlDir('choose a recording output directory')
        self._vis_plot_vocal = ControlVisVis('vocalization')
        self._button_previous_vocal = ControlButton('Previous')
        self._button_accept_vocal = ControlButton('Accept')
        self._button_reject_vocal = ControlButton('Reject')
        self._button_next_vocal = ControlButton('Next')
        self._label_vocal_ID = ControlLabel('')
        self._label_vocal_start = ControlLabel('')
        self._label_vocal_end = ControlLabel('')
        self._label_vocal_duration = ControlLabel('')
        self._label_vocal_avg_freq = ControlLabel('')
        self._label_vocal_bandwidth = ControlLabel('')
        self._label_vocal_label = ControlLabel('')
        #   # actions

        # Define the event called before showing the image in the player
        # self._player.process_frame_event = self.__process_frame

        # Define the organization of the Form Controls
        self._formset = [
            {
                "a:choose directory": [
                    '',
                    (' ', 'h3: step 1: choose audio files', ' '),
                    ('_dir_load_directory', '_checkbox_searchSubDirs'),
                    ('_checkboxlist_fileList'),
                    (' ', '_button_select_all', '_button_select_none', '_button_select_clear', ' '),
                    ('h3: for more information, visit:', '_button_more_info', ' '),
                    ' ',
                ],
                "b:animal pipeline": [
                    '',
                    (' ', 'h3: step 2: select animal pipeline', ' '),
                    (' ', '_combo_animal_dropdown', ' '),
                    (' ', '_number_lo_freq_cutoff', ' '),
                    (' ', '_number_hi_freq_cutoff', ' '),
                    ' '
                ],
                "c:optional settings": [
                    '',
                    (' ', 'h3: step 3: select optional settings', ' '),
                    (' ', '_number_thread_count', ' '),
                    (' ', '_checkbox_save_spectrograms', ' '),
                    (' ', '_checkbox_save_segmentation', ' '),
                    (' ', '_checkbox_save_overlay', ' '),
                    (' ', '_checkbox_save_plots', ' '),
                    ' '
                ],
                "d:run analysis": [
                    '',
                    (' ', 'h3: step 4: run analysis', ' '),
                    (' ', '_button_run_analysis', ' '),
                    (' ', '_progress_bar', ' '),
                    (' ', '_label_run_analysis', ' '),
                    ' ',
                ],
                "e:inspect vocalizations": [
                    '',
                    (' ', 'h3: step 5: inspect vocalizations', ' '),
                    ('_dir_load_vocalpy_file'),
                    (' ', 'ID:', '_label_vocal_ID', 'start:', '_label_vocal_start', 'end:', '_label_vocal_end', 'duration:', '_label_vocal_duration', ' '),
                    (' ', 'average frequency:', '_label_vocal_avg_freq', 'bandwidth:', '_label_vocal_bandwidth', 'label:', '_label_vocal_label', ' '),
                    ('_vis_plot_vocal'),
                    (' ', '_button_previous_vocal', '_button_accept_vocal', '_button_reject_vocal', '_button_next_vocal', ' '),
                    ' '
                ],
            }
        ]

    def __directorySelectionEvent(self):
        """
        When the directory is selected instanciate the video in the player
        """
        from vocalpy.utils.io import parse_input_path
        self._checkboxList.clear()
        search_subdir = self._searchSubDirs.value
        filelist = parse_input_path(self._directory.value, search_subdir)
        if filelist == -1 or not filelist:
            self._checkboxList += ('no audios found, select a different directory or use "search subdirectories"')
        else:
            for file in filelist:
                self._checkboxList += (file, True)

    def __selectAllCheckbox(self):
        """
        Do some processing to the frame and return the result frame
        """
        items = self._checkboxList.items
        self._checkboxList.clear()
        for item in items:
                self._checkboxList += (item[0], True)

    def __selectNoneCheckbox(self):
        """
        Do some processing to the frame and return the result frame
        """
        items = self._checkboxList.items
        self._checkboxList.clear()
        for item in items:
                self._checkboxList += (item[0], False)

    def __selectClearCheckbox(self):
        """
        Do some processing to the frame and return the result frame
        """
        self._checkboxList.clear()

    def __openURLButton(self):
        url = 'http://dietrich-lab.org/vocalpy'
        webbrowser.open(url)

    def __changeFrequencyAnimal(self):
        selected_animal = self._animal_dropdown.value
        if selected_animal == 'mouse':
            self._lo_freq_cutoff.value = 45000
            self._hi_freq_cutoff.value = 125000
        elif selected_animal == 'rat':
            self._lo_freq_cutoff.value = 20000
            self._hi_freq_cutoff.value = 125000
        elif selected_animal == 'guineapig':
            self._lo_freq_cutoff.value = 100
            self._hi_freq_cutoff.value = 20000

    def __numberOfAudios(self):
        return 10**6

    def __runVocalPy(self):
        self._run_analysis_label.value = 'analysis running, check terminal for detailed logging information...'
        for i in range(0, 10**6):
            self._progress_bar.value = i
        self._progress_bar.value = self._progress_bar.max
        self._run_analysis_label.value = '!!!!!! analysis complete (optional: go to  "inspect vocalizations") !!!!!!'


if __name__ == '__main__':
    from pyforms import start_app
    start_app(VocalPy, geometry=(400, 400, 650, 450))
