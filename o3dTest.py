import open3d as o3d
import numpy as np

def pose(xyz):
    m = np.identity(4)
    m[0:3,3] = xyz
    return m

def show(xyz, XYZ):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    obs = [pcd]
    for ifor in range(XYZ.shape[0]):
        sp = o3d.create_mesh_sphere(0.05).transform(pose(XYZ[ifor, :]))
        obs.append(sp)
    o3d.draw_geometries(obs)

class TestClass():

    def __init__(self, xyz):

        self.xyz = xyz
        XYZ = self.defineGrids()
        XYZ = self.scanGrids(XYZ)
        show(self.xyz, XYZ)

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

        X, Y, Z = np.meshgrid(iranges[0], iranges[1], iranges[2])
        print('Number of grids = {} x {} x {} = {}'.format(*X.shape, np.prod(X.shape)))

        return [X, Y, Z]


    def scanGrids(self, XYZ):
        R = np.sqrt(0.5 ** 2 + 0.5 ** 2 + 0.5 ** 2)
        XYZ = np.vstack([XYZ[i].flatten() for i in range(3)]).T
        return XYZ
        # for spot in range(XYZ.shape[0]):
        #     if spot % 100 == 0:
        #         print(".", end="", flush=True)
        #     if spot % 1000 == 0:
        #         print("\n", end="", flush=True)
        #
        #     DV = Dice.VV
        #     for vectors in Dice.VV:
        #         sampleImage = self.getSample(XYZ[spot, :], vectors, R)
        #         predictions = self.model(sampleImage)
        #         if len(predictions) != 0:
        #             self.samples.add(predictions, XYZ[spot, :], vectors)


xyz = np.random.random((3000,3))
xyz[:,0] *= 3
xyz[:,1] *= 6
xyz[:,2] *= 2
param = {'bigStep' : 0.75,
             'smallStep' : 0.06}
t = TestClass(xyz)
