import numpy as np
import open3d as o3d
import pandas as pd
import Net
import os
from scipy import spatial
import cv2
import matplotlib.pyplot as plt
from pyquaternion import Quaternion


def slice(xyz):
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
        cv2.waitKey(1)

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
        print("Loading {}".format(fileName))
        self.df = pd.read_pickle(fileName)
        print("Length = {} samples".format(self.df.shape[0]))
        return self.df

    def filter(self, classNumber):
        self.df = self.df[self.df['class']==classNumber]

    def filterGreater(self, field, value):
        self.df = self.df[self.df[field]>value]

    def __getitem__(self, item):
        if len(self.df):
            row = self.df.iloc[item]
            coord = (np.array(row['coord']) - 448//2)/448
            vectors = row['vectors'].reshape(3,3)
            xyz = row['xyz']
            radius = (row['bx'] + row['by'])/4.0/448
            center = xyz + vectors[:,1] * (coord[1] + radius) + vectors[:,2] * (coord[0] + radius)
            return (center, radius)

    def __len__(self):
        if len(self.df):
            return self.df.shape[0]
        else:
            return 0

def defineGridsProjected(xyz):
    """
    Make three sets of three dimensional grids.
    Locations and vectors for extraction need to be defined.
    :return:
    """
    mins = np.min(xyz, axis=0)
    maxs = np.max(xyz, axis=0)
    iranges = []
    focusDirection = 0
    for ifor in range(3):
        r = maxs[ifor] - mins[ifor]
        if ifor == focusDirection:
            stepSize = param['smallStep']
        else:
            stepSize = param['bigStep']

            # Only shrink the big step directions
            if r > param['W']:
                mins[ifor] += param['W'] / 2.0
                maxs[ifor] -= param['W'] / 2.0
                r = r - param['W']

        iranges.append(np.linspace(mins[ifor], maxs[ifor], np.ceil(r / stepSize) + 1))

    X, Y, Z = (iranges[0], iranges[1], iranges[2])
    total = np.prod([X.shape[0], Y.shape[0], Z.shape[0]])
    print('Number of grids = {} x {} x {} = {}: '.format(X.shape[0], Y.shape[0], Z.shape[0], total), end="")

    return [X, Y, Z]


def makeVectors(theta, elevation):

    theta_quat = Quaternion(axis=[0, 0, 1], angle=theta)
    vprime = theta_quat.rotate([0, 1., 0.])
    print(vprime)
    phi_quat = Quaternion(axis=vprime, angle=-elevation)
    spherical = phi_quat * theta_quat

    # print("{}\n".format(theta_quat.rotation_matrix))
    # print("{}\n".format(phi_quat.rotation_matrix))
    # print("{}\n".format((phi_quat.rotation_matrix).dot(theta_quat.rotation_matrix)))
    # print("{}\n".format(spherical.rotation_matrix))

    return spherical.rotation_matrix

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

        fortyFive = np.pi/4
        for theta in [0, fortyFive, 2*fortyFive, 3*fortyFive]:
            if theta == 0:
                eRange = [-fortyFive, 0, fortyFive, 2*fortyFive]
            else:
                eRange = [-fortyFive, 0, fortyFive]

            for elevation in eRange:
                vectors = makeVectors(theta, elevation)
                projected = self.xyz.dot(vectors)
                XYZprojected = defineGridsProjected(projected)
                self.scanGrids(XYZprojected, vectors)
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

        return slice(projection)



    def scanGrids(self, XYZ, vectors):
        R = np.sqrt(0.5**2 + 0.5**2 + 0.5**2)
        spot = 0
        x1Index = 0
        x2Index = 1
        x3Index = 2
        for x2 in XYZ[x2Index]:
            for x3 in XYZ[x3Index]:
                for x1 in XYZ[x1Index]:
                    if spot and spot % 100 == 0:
                        print(".", end="", flush=True)

                    loc = np.array([x1, x2, x3]). dot(np.linalg.inv(vectors))
                    sampleImage = self.getSample(loc, vectors, R)
                    SKIP_NNET = False
                    if not SKIP_NNET:
                        predictions = self.model(sampleImage)
                        if len(predictions) != 0:
                            self.samples.add(predictions, loc, vectors)
                    spot = spot + 1
        print("")

if __name__ == '__main__':

    Ready = True
    if Ready:
        param = {'bigStep' : 0.65,
                 'smallStep' : 0.06,
                 'W' : 1}
        S = Samples()
        Yolo = Net.Yolo()
        #D = Dice('~/sites/tetraTech/BoilerRoom/chunkSmallest.pcd', 'superPoints/pointsDataFrameB.pkl', S, Yolo)

        D = Dice('~/sites/tetraTech/BoilerRoom/chunk_cheap.pcd', 'superPoints/chunk_cheap45.pkl', S, Yolo)
        #D = Dice('~/sites/tetraTech/BoilerRoom/full_5mm.pcd', 'superPoints/full_5mmB.pkl', S, Yolo)









