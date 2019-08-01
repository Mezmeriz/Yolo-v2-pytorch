import numpy as np
import open3d as o3d
import pandas as pd
import Net

def slice(xyz, pts, vectors):

    newxyz = xyz
    image = np.zeros((448, 448, 3))
    for jfor in range(-1, 2):
        II = (newxyz[:, 0] > 1 + jfor * sliceWidth - sliceWidth / 2) & (
                    newxyz[:, 0] < 1 + jfor * sliceWidth + sliceWidth / 2)
        h, xe, ye = np.histogram2d(newxyz[II, 1], newxyz[II, 2],
                                   (np.linspace(-0.5, 0.5, 449), np.linspace(-0.5, 0.5, 449)))
        m = np.max(h)
        h = (h / m * 254).astype(np.uint8)
        h[h > 0] = 255
        image[:, :, jfor + 1] = h
    return image

class Samples():
    # Vectors
    V1 = np.identity(3)
    V2 = V1[:, [1,2,0]]
    V3 = V1[:, [2,0,1]]
    VV = [V1, V2, V3]

    def __init__(self):
        self.df = pd.DataFrame()

    def add(self, predictions, coordinate, vectors):
        Broken Here!
        Use parameters to fill sample.
        df = pd.DataFrame({'score': [number], 'coordinate': [coord], 'vectors': [matrix.flatten()]}, index = [0])
        self.df = self.df.append(df, ignore_index = True)

    def save(self, fileName):
        self.df.to_pickle(fileName)

    def load(self, fileName):
        df = pd.read_pickle(fileName)
        return df

class Dice():

    def __init__(self, fileIn, fileOut, samples, param, model):
        self.samples = samples
        self.model = model

        print("Loading file {}".format(fileIn))
        pcd = o3d.io.read_point_cloud(fileIn)
        self.xyz = np.asarray(pcd.points)
        print("File loaded with {} points".format(self.xyz.shape[0]))

        self.build_KDTree()
        XYZ = self.defineGrids(param)
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
        mins = np.min(self.xyz)
        maxs = np.max(self.xyz)
        iranges = []
        for ifor in range(3):
            r = maxs[ifor] - mins[ifor]
            iranges.append(np.linspace(mins[ifor], maxs[ifor], np.ceil(r/param['bigStep'])))

        X,Y,Z = np.meshgrid(iranges[0], iranges[1], iranges[2])
        return [X,Y,Z]

    def scanGrids(self, XYZ):
        R = np.sqrt(0.5**2 + 0.5**2 + 0.1**2)
        XYZ = np.hstack([i.flatten() for i in range(3)]).T
        for spot in range(XYZ.shape[0]):
            for vectors in VV:
                sampleImage = self.getSample(XYZ[spot, :], vectors, R)
                predictions = self.model(sampleImage)
                if len(predictions) != 0:
                    self.samples.add(predictions, XYZ[spot, :], vectors)

if __name__ == '__main__':

    param = {'bigStep' : 0.75,
             'smallStep' : 0.06}
    S = Samples()
    Yolo = Net.Yolo()
    D = Dice(fileIn = '/home/sadams/sites/tetraTech/Enfield Boiler Room/chunkSmallest.pcd', Samples, param, Yolo)

