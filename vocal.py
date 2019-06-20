


class Vocal(object):
    '''
    vocalization
    '''
    def __init__(self, bin_number    = None,
                       start         = None,
                       end           = None,
                       min_freq_main = None,
                       max_freq_main = None,
                       avg_freq_main = None,
                       min_freq_all  = None,
                       max_freq_all  = None,
                       avg_freq_all  = None,
                       min_intensity = None,
                       max_intensity = None,
                       avg_intensity = None,
                       bg_intensity  = None,
                       area          = None,
                       points        = None,
                       centroid      = None,
                       orientation   = None):

        self.bin           = bin_number
        self.start         = start
        self.end           = end
        self.duration      = self.end - self.start
        self.min_freq_main = min_freq_main
        self.max_freq_main = max_freq_main
        self.avg_freq_main = avg_freq_main
        self.min_freq_all  = min_freq_all
        self.max_freq_all  = max_freq_all
        self.avg_freq_all  = avg_freq_all
        self.bandwidth     = bandwidth
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.avg_intensity = avg_intensity
        self.bg_intensity  = bg_intensity
        self.area          = area
        self.points        = points
        self.centroid      = centroid
        self.orientation   = orientation