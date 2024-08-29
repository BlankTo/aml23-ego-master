class ActionEMGRecord(object):
    def __init__(self, tup, dataset_conf):
        self._series = tup
        self.dataset_conf = dataset_conf

    @property
    def uid(self):
        return self._series['uid']

    @property
    def myo_right_readings(self):
        return self._series['myo_right_readings']

    @property
    def myo_left_readings(self):
        return self._series['myo_left_readings']

    @property
    def label(self):
        return self._series['label']