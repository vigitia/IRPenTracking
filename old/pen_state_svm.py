import sklearn.preprocessing
from sklearn import svm
import numpy as np
import csv


class PenStateSVM:
    TOUCH = 0
    HOVER = 1
    AWAY = 2

    @staticmethod
    def create(files):
        labels = [PenStateSVM.TOUCH, PenStateSVM.HOVER, PenStateSVM.AWAY]
        pssvm = PenStateSVM(files, labels)
        return pssvm

    def __init__(self, files, labels):
        self.scaler = None
        self.c = self.create_pen_state_svm(files, labels)
        print('FINISH INIT SVM')

    def fetch(self, files, labels):
        data = []
        for f, l in zip(files, labels):
            data += self.read_csv_file(f, l)
        return np.asarray(data)

    def read_csv_file(self, f, l, delim=","):
        ret = []
        with open(f, newline='') as csvfile:
            csvr = csv.reader(csvfile, delimiter=delim, quotechar='|')

            ret = []
            for r in csvr:
                # print(r)
                radius = r[0]
                brightness = r[1]
                aspect_ratio = r[2]
                x = r[3]
                y = r[4]
                #ret.append([float(radius), float(brightness), float(aspect_ratio), l])
                ret.append([float(radius), float(brightness), float(x), float(y), l])

            #ret = [[*map(float, delim.join(r).strip().split(delim)), l] for r in csvr]

        return ret[10:-10]

    def train(self, c, data):
        X, y = data[:, :-1], data[:, -1]
        self.scaler = sklearn.preprocessing.StandardScaler().fit(X)
        X_scaled = self.scaler.transform(X)
        c.fit(X_scaled, y)
        #c.fit(X, y)

    def predict(self, data):
        X_scaled = self.scaler.transform([data])
        print(">>>", X_scaled)
        return int(self.c.predict(X_scaled)[0])
        #return int(self.c.predict([data])[0])

    def create_pen_state_svm(self, files, labels):
        c = svm.SVC()
        data = self.fetch(files, labels)
        self.train(c, data)
        return c


if __name__ == '__main__':
    c = PenStateSVM.create(["test_touch.csv", "test_hover.csv"], )
    to_predict = [5, 1, 6]  # use a python list with [radius, brightness, aspect ratio]
    res = c.predict(to_predict)
    print(res)
