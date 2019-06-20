





class ListOfVocals(object):
    '''
    list of vocalizations
    '''
    def __init__(self, vocals_in_audio=None, number_of_vocals=None):
        self.vocals_in_audio  = vocals_in_audio
        self.number_of_vocals = len(self.vocals_in_audio)
        self.vocals_processed = False