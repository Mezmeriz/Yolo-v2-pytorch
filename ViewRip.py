import open3d as o3d
import numpy as np
from DiceSimple import Samples
import os

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

class ViewRip():

    def __init__(self, fileIn, superPoints):
        fileIn = os.path.expanduser(fileIn)
        print("Loading file {}".format(fileIn))
        self.pcd = o3d.io.read_point_cloud(fileIn)
        self.xyz = np.asarray(self.pcd.points)
        print("File loaded with {} points".format(self.xyz.shape[0]))

        self.showObjects = [self.pcd]

        self.superPoints = superPoints
        self.addSamples()
        o3d.draw_geometries(self.showObjects)

    def addSamples(self):
        for ifor in range(len(self.superPoints)):
            center, radius = self.superPoints[ifor]
            sphere = o3d.create_mesh_sphere(radius).transform(pose(center))
            sphere.paint_uniform_color([0.1, 0.1, 0.7])
            sphere.compute_vertex_normals()
            self.showObjects.append(sphere)


pairs = [('~/cheap.pcd', 'superPoints/pointsDataFrameB.pkl'),
         ('~/sites/tetraTech/BoilerRoom/chunk_cheap.pcd', 'superPoints/chunk_cheap.pkl'),
         ('~/sites/tetraTech/BoilerRoom/full_5mm.pcd', 'superPoints/full_5mm.pkl')]
pair = 1

superPoints = Samples()
superPoints.load(pairs[pair][1])
print("Length pre filter {}".format(len(superPoints)))
superPoints.filter(classNumber='circle')
print("Length post filter {}".format(len(superPoints)))
VR = ViewRip(pairs[pair][0], superPoints)