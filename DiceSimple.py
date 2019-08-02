import numpy as np
import open3d as o3d
import pandas as pd
import Net
import os
from scipy import spatial
import cv2
import matplotlib.pyplot as plt

def slice(xyz, pts, vectors):
    sliceWidth = 0.03
    newxyz = xyz
    image = np.zeros((448, 448, 3), dtype=np.uint8)
    for jfor in range(-1, 2):
        II = (newxyz[:, 0] >  jfor * sliceWidth - sliceWidth / 2) & (
                    newxyz[:, 0] < jfor * sliceWidth + sliceWidth / 2)
        h, xe, ye = np.histogram2d(newxyz[II, 1], newxyz[II, 2],
                                   (np.linspace(-0.5, 0.5, 449), np.linspace(-0.5, 0.5, 449)))
        m = np.max(h)
        if m > 0:
            h = (h / m * 255).astype(np.uint8)
        h[h > 0] = 255
        image[:, :, jfor + 1] = h
        cv2.imshow("check", image)
        cv2.waitKey(10)

    return image

class Samples():

    def __init__(self):
        self.df = pd.DataFrame()

    def add(self, predictions, spot, vectors):
        predictions = predictions[0]
        for pi in range(len(predictions)):
            pred = predictions[pi]
            dict = {}
            dict['coord'] = [(pred[0], pred[1])]
            dict['bx'] = [pred[2]]
            dict['by'] = [pred[3]]
            dict['objectness'] = [pred[4]]
            dict['class'] = [pred[5]]
            dict['xyz'] = [spot]
            dict['vectors'] = [vectors.flatten()]
            df = pd.DataFrame(dict, index = [0])
            if len(self.df) == 0:
                self.df = df
            else:
                self.df = self.df.append(df, ignore_index = True)

    def save(self, fileName):
        self.df.to_pickle(fileName)

    def load(self, fileName):
        df = pd.read_pickle(fileName)
        return df

class Dice():
    # Vectors
    V1 = np.identity(3)
    V2 = V1[:, [1,2,0]]
    V3 = V1[:, [2,0,1]]
    VV = [V1, V2, V3]

    def __init__(self, fileIn, fileOut, samples, model):
        self.samples = samples
        self.model = model

        fileIn = os.path.expanduser(fileIn)
        print("Loading file {}".format(fileIn))
        self.pcd = o3d.io.read_point_cloud(fileIn)
        self.xyz = np.asarray(self.pcd.points)
        print("File loaded with {} points".format(self.xyz.shape[0]))

        self.build_KDTree()
        XYZ = self.defineGrids()
        self.scanGrids(XYZ)
        self.samples.save(fileOut)

    def build_KDTree(self):
        print("Building kd Tree")
        self.tree = spatial.KDTree(self.xyz)
        print("KD Tree complete")

    def getSample(self, sampleLocation, vec, sphereSize=1.5):
        subsetIndicies = self.tree.query_ball_point(x=sampleLocation, r=sphereSize)
        region = self.xyz[subsetIndicies, :]
        region = region - sampleLocation
        projection = np.matmul(region, vec)

        return slice(projection, vec, sampleLocation)

    def defineGrids(self):
        """
        Make three sets of three dimensional grids.
        Locations and vectors for extraction need to be defined.
        :return:
        """
        mins = np.min(self.xyz, axis=0)
        maxs = np.max(self.xyz, axis=0)
        iranges = []
        stepSizes = [param['bigStep'], param['smallStep'], param['bigStep']]
        for ifor in range(3):
            r = maxs[ifor] - mins[ifor]
            iranges.append(np.linspace(mins[ifor], maxs[ifor], np.ceil(r / stepSizes[ifor] + 1)))

        X, Y, Z = (iranges[0], iranges[1], iranges[2])
        total = np.prod([X.shape[0], Y.shape[0], Z.shape[0]])
        print('Number of grids = {} x {} x {} = {}'.format(X.shape[0], Y.shape[0], Z.shape[0], total))

        return [X, Y, Z]

    def scanGrids(self, XYZ):
        X, Y, Z = tuple(XYZ)
        R = np.sqrt(0.5**2 + 0.5**2 + 0.5**2)
        spot = 0
        for xi in X:
            for zi in Z:
                for yi in Y:
                    if spot % 100 == 0:
                        print(".", end="", flush=True)
                    if spot % 1000 == 0:
                        print("\n", end="", flush=True)

                    loc = np.array([xi, yi, zi])
                    vectors = Dice.V2
                    sampleImage = self.getSample(loc, vectors, R)
                    predictions = self.model(sampleImage)
                    if len(predictions) != 0:
                        self.samples.add(predictions, loc, vectors)
                    spot = spot + 1

if __name__ == '__main__':

    param = {'bigStep' : 0.75,
             'smallStep' : 0.06}
    S = Samples()
    Yolo = Net.Yolo()
    D = Dice('~/cheap.pcd', '/home/scott/pointsDataFrame.pkl', S, Yolo)

