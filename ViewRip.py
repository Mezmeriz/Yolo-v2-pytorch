import open3d as o3d
import numpy as np
from DiceSimple import Samples
import os
import connect.leggo
from pyquaternion import Quaternion

def pose(xyz, rot = None):
    m = np.identity(4)
    m[0:3,3] = xyz
    if rot is not None:
        m[0:3, 0:3] = rot
    return m

def show(xyz, XYZ):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    obs = [pcd]
    for ifor in range(XYZ.shape[0]):
        sp = o3d.create_mesh_sphere(0.05).transform(pose(XYZ[ifor, :]))
        obs.append(sp)
    o3d.draw_geometries(obs)

COLOR_BLUE = [0,0,1]
COLOR_GREEN = [0.1, 0.9, 0.1]
COLOR_RED = [1, 0, 0]

class ViewRip():

    def __init__(self, fileIn, superPoints, straightSegments = None, chain = None):
        if len(fileIn):
            fileIn = os.path.expanduser(fileIn)
            print("Loading file {}".format(fileIn))
            self.pcd = o3d.io.read_point_cloud(fileIn)
            self.xyz = np.asarray(self.pcd.points)
            print("File loaded with {} points".format(self.xyz.shape[0]))
            self.showObjects = [self.pcd]
        else:
            self.showObjects = []

        self.superPoints = superPoints
        self.addSamples(R = 0.0145, chain= chain)
        self.addHeads()
        self.addTails()
        if straightSegments is not None:
            self.addStraight(straightSegments)
        o3d.draw_geometries(self.showObjects)
        # pp = self.getPickedPoints()
        # print(pp)


        # total = self.showObjects[0]
        # for i in range(1,len(self.showObjects)):
        #     total += self.showObjects[i]
        # o3d.write_triangle_mesh('test.ply', total)

    def getPickedPoints(self):
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(self.showObjects)
        vis.run()  # user picks points
        vis.destroy_window()

    def addSamples(self, R = None, chain = None):
        CHAIN = [i for i in self.superPoints.df.keys()].index('chain')
        for ifor in range(len(self.superPoints)):
            center, radius = self.superPoints[ifor]
            chainiFor = self.superPoints.df.iloc[ifor, CHAIN]
            if R is None:
                R = radius
            sphere = o3d.create_mesh_sphere(R,8).transform(pose(center))

            if chain is not None and chainiFor == chain:
                sphere.paint_uniform_color(COLOR_GREEN)
            else:
                sphere.paint_uniform_color(COLOR_BLUE)

            sphere.compute_vertex_normals()
            self.showObjects.append(sphere)

    def addCylinder(self, start, end, rotate=True, color=[0.9, 0.0, 0.3], radius = None):

        DEFAULT_CYLINDER_RADIUS = 0.005
        if radius is None:
            radius = DEFAULT_CYLINDER_RADIUS

        length = np.linalg.norm(start - end)
        n = (end - start) / length
        phi = np.arccos(n[2])
        theta = np.arctan2(n[1], n[0])

        theta_quat = Quaternion(axis=[0, 0, 1], angle=theta)
        vprime = theta_quat.rotate([0, 1., 0.])
        phi_quat = Quaternion(axis=vprime, angle=phi)
        rot = phi_quat.rotation_matrix

        cyl = o3d.create_mesh_cylinder(radius, length, resolution=8)
        if rotate:
            cyl = cyl.transform(pose(np.array((start + end) / 2.0), rot))
        #     .transform(pose(center))
        cyl.paint_uniform_color(color)
        cyl.compute_vertex_normals()
        self.showObjects.append(cyl)

    def addStraight(self, straightSegments):
        for row in straightSegments:
            chain, startIndex, endIndex, radius = row
            start = self.superPoints.df.loc[startIndex].at['centers']
            end = self.superPoints.df.loc[endIndex].at['centers']
            self.addCylinder(start, end, True, COLOR_RED, radius)

    def addHeads(self):
        if "head" in self.superPoints.df.keys():
            for ifor in range(len(self.superPoints)):
                start = self.superPoints.df.iloc[ifor].at['centers']
                head = self.superPoints.df.iloc[ifor].at['head']
                if head != -1:
                    try:
                        end = self.superPoints.df.loc[head].at['centers']
                        self.addCylinder(start, end, True, COLOR_BLUE)
                    except:
                        pass

    def addTails(self):
        if "head" in self.superPoints.df.keys():
            for ifor in range(len(self.superPoints)):
                start = self.superPoints.df.iloc[ifor].at['centers']
                head = self.superPoints.df.iloc[ifor].at['tail']
                if head != -1:
                    try:
                        end = self.superPoints.df.loc[head].at['centers']
                        self.addCylinder(start, end, True, COLOR_BLUE)
                    except:
                        pass

if __name__ == '__main__':
    pairs = [('~/cheap.pcd', 'superPoints/pointsDataFrameB.pkl'),
             ('~/sites/tetraTech/BoilerRoom/chunk_cheap.pcd', 'superPoints/chunk_cheap.pkl'),
             ('~/sites/tetraTech/BoilerRoom/full_5mm.pcd', 'superPoints/full_5mm.pkl'),
             ('~/sites/tetraTech/BoilerRoom/chunk_cheap.pcd', 'superPoints/chunk_cheapB.pkl'),
             ('', 'superPoints/chunk_cheapB.pkl'),
             ('~/sites/tetraTech/BoilerRoom/chunk_cheap.pcd', 'superPoints/chunk_cheap45.pkl'),
             ('', 'superPoints/chunk_cheap45.pkl'),
             ('', 'superPoints/synthA.pkl')]
    pair = pairs[-5]

    superPoints = Samples()
    superPoints.load(pair[1])
    print("Length pre filter {}".format(len(superPoints)))
    #superPoints.filter(classNumber='circle')
    superPoints.filterGreater('objectness', 0.5)
    superPoints = connect.leggo.orphanFilter(superPoints, N=3)

    print("Length post filter {}".format(len(superPoints)))


    import Chains
    C = Chains.Chains(superPoints.df)
    C.save('tmp/chain981.pkl')


    straightSegments = connect.leggo.makeStraight(superPoints)
    # straightSegments = []
    if len(straightSegments):
        VR = ViewRip(pair[0], superPoints, straightSegments, chain = 981)
    else:
        VR = ViewRip(pair[0], superPoints)
    print(superPoints.df.head(100))
    print(superPoints.df.keys())