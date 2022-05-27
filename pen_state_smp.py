import sklearn.preprocessing
from sklearn import svm
import numpy as np
import csv


class PenStateSMP:
    TOUCH = 0
    HOVER = 1
    AWAY = 2

    BUFSIZE = 24
    PAST_STATES = 9

    def __init__(self):
        self.labels = [PenStateSMP.TOUCH, PenStateSMP.HOVER, PenStateSMP.AWAY]
        print('FINISH INIT SMP')
        self.brightest = []
        self.radii = []
        self.brt_touch = []
        self.rad_touch = []
        self.brt_max = -1
        self.rad_max = -1
        self.states = []

    def predict(self, x, y, radius, brightest_spot, aspect_ratio):
        # print(">>>> ", x, y, radius, brightest_spot, aspect_ratio)
        self.brightest.append(brightest_spot)
        self.brightest = self.brightest[-PenStateSMP.BUFSIZE:]
        #self.brt_max = max(self.brightest)
        self.brt_max = max(self.brt_max, brightest_spot)
        self.radii.append(radius)
        self.radii = self.radii[-PenStateSMP.BUFSIZE:]
        # self.rad_max = max(self.radii)
        self.rad_max = max(self.rad_max, radius)
        print(self.brt_max, brightest_spot, self.rad_max, radius)

        if len(self.brightest) < 5 or len(self.radii) < 5:
            state = 99 # unknown
        else:
            if brightest_spot < self.brt_max - 15: # 20
                state = PenStateSMP.AWAY
            elif radius < self.rad_max - 2:  # 2
                # check if radius stays the same
                avg = sum(self.radii) / len(self.radii)
                if abs(radius - avg) < 0.3:   # 0.3
                    state = PenStateSMP.TOUCH
                else:
                    state = PenStateSMP.HOVER
            else:
                state = PenStateSMP.HOVER

        self.states.append(state)
        self.states = self.states[-PenStateSMP.PAST_STATES:]
        num_touch = sum([1 for s in self.states if s == PenStateSMP.TOUCH])
        if num_touch > len(self.states) // 2:
            return PenStateSMP.TOUCH
        else:
            return state