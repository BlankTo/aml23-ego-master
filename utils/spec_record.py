class ActionEMGspecRecord(object):
    def __init__(self, tup, dataset_conf):
        self._series = tup
        self.dataset_conf = dataset_conf

    @property
    def uid(self):
        return self._series['uid']

    @property
    def right_spectrogram(self):
        return self._series['right_spectrogram']

    @property
    def left_spectrogram(self):
        return self._series['left_spectrogram']

    @property
    def label(self):
        return self._series['label']