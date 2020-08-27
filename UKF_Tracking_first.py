from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
from scipy.spatial import distance as dist
from collections import OrderedDict
# from track import trackedObj
import numpy as np


def f_cv(x, dt):
    F = np.array([[1, dt, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, dt],
                  [0, 0, 0, 1]], dtype=float)
    return np.dot(F, x)


def h_cv(x):
    return x[0], x[2]


' # Class definition of the tracker'


class ObjectTracker:
    def __init__(self, maxDisappeared=30):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.bBox = OrderedDict()
        self.classify = OrderedDict()
        self.UKF = OrderedDict()
        self.predList = OrderedDict()
        self.noise = OrderedDict()

    def register(self, centroid, bBox, clas):  # Registers a new object to be tracked
        dt = 1.
        std_x, std_y = 0.3, 0.3
        sigmas = MerweScaledSigmaPoints(4, alpha=0.1, beta=2., kappa=1.)
        self.UKF[self.nextObjectID] = UKF(dim_x=4, dim_z=2, fx=f_cv, hx=h_cv, dt=dt, points=sigmas)
        self.UKF[self.nextObjectID].R = np.diag([0.09, 0.09])
        self.UKF[self.nextObjectID].Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=1, var=0.02)
        self.UKF[self.nextObjectID].Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=1, var=0.02)
        self.UKF[self.nextObjectID].x = np.array([float(centroid[0]), 0., float(centroid[1]), 0.])
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.bBox[self.nextObjectID] = bBox
        self.classify[self.nextObjectID] = clas
        self.predList[self.nextObjectID] = 0
        self.UKF[
            self.nextObjectID].predict()  # Object point is predicted and estimated to tune other params of the filter
        self.UKF[self.nextObjectID].update([float(centroid[0]), float(centroid[1])])
        self.noise[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):  # delete the object which is no longer tracked
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.bBox[objectID]
        del self.classify[objectID]
        del self.predList[objectID]
        del self.UKF[objectID]

    def update(self, pxl, bBox, clas):  # procedure to determine if object is previously tracked or new
        if len(pxl) == 0:
            for objectID in self.disappeared.keys():
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return

        inputCentroids = np.zeros((len(pxl), 2), dtype="double")

        for i in range(len(pxl)):
            inputCentroids[i] = (pxl[i][0], pxl[i][1])

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], bBox[i], clas[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            print("ObjCent: ", objectCentroids)
            print("imgCent: ", inputCentroids)
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            print("D: ", D)
            newD = D.min(axis=1)
            print("newD: ", newD)
            rows = newD.argsort()
            print("rows: ", rows)
            cols = D.argmin(axis=1)[rows]
            print("cols: ", cols)
            usedRows = set()
            usedCols = set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.UKF[objectID].predict()
                self.UKF[objectID].update(inputCentroids[col])
                self.noise[objectID] = inputCentroids[col] - self.objects[objectID]
                print("Noise: ", self.noise[objectID])
                if self.predList[objectID] < 5:  # Original points considered for associating an object
                    self.predList[objectID] += 1
                    self.objects[objectID] = inputCentroids[col]
                else:  # UKF filter will provide estimations which will improve the association of a tracked object
                    self.objects[objectID] = [self.UKF[objectID].x[0], self.UKF[objectID].x[2]]
                self.bBox[objectID] = bBox[col]
                self.classify[objectID] = clas[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]

                    self.UKF[objectID].predict()  # Estimate a lost object by using the generated noise
                    self.UKF[objectID].update(self.objects[objectID] + self.noise[objectID])
                    self.objects[objectID] = [self.UKF[objectID].x[0], self.UKF[objectID].x[2]]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], bBox[col], clas[col])

        return
