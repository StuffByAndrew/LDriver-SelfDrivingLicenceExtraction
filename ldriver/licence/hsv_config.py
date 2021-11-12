import numpy as np

class HSVRanges:
    """stores two lists of

    Attributes:
        lower (list): list of lower hsv values
        upper (list): list of upper hsv values
    """
    def __init__(self):
        self.lower = []
        self.upper = []

    def add_range(self, hsv_upper, hsv_lower):
        self.upper.append(np.array(hsv_upper))
        self.lower.append(np.array(hsv_lower))

    def get_ranges(self):
        return zip(self.lower, self.upper)

uh = 0
us = 0
uv1 = 125
lh = 0
ls = 0
lv1 = 90
lv2 = 155
uv2 = 210

licence_ranges = HSVRanges()
licence_ranges.add_range([uh,us,uv1], [lh,ls,lv1])
licence_ranges.add_range([uh,us,uv2],[lh,ls,lv2])