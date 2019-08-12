import open3d as o3d
import numpy as np
from DiceSimple import Samples
import os
import connect.leggo
from pyquaternion import Quaternion
import time

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

def save_view_point(vis, filename):
    vis.run() # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.write_pinhole_camera_parameters(filename, param)


def load_view_point(vis, filename):
    ctr = vis.get_view_control()
    param = o3d.read_pinhole_camera_parameters(filename)
    ctr.convert_from_pinhole_camera_parameters(param)

class OTest():

    def __init__(self, fileIn, superPoints):
        if len(fileIn):
            fileIn = os.path.expanduser(fileIn)
            print("Loading file {}".format(fileIn))
            self.pcd = o3d.io.read_point_cloud(fileIn)
            self.xyz = np.asarray(self.pcd.points)
            print("File loaded with {} points".format(self.xyz.shape[0]))
            self.showObjects = [self.pcd]
        else:
            self.showObjects = []

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.vis.add_geometry(self.pcd)
        self.vis.add_geometry(self.makeSphere())

        save_view_point(self.vis, 'test.json')
        load_view_point(self.vis, 'test.json')

        self.moveSphere()

        self.vis.destroy_window()

    def getView(self):
        ctr = self.vis.get_view_control()
        self.vis.run()
        param = ctr.convert_to_pinhole_camera_parameters()
        trajectory = o3d.PinholeCameraTrajectory()
        trajectory.intrinsic = param.intrinsic
        trajectory.extrinsic = param.extrinsic #Matrix4dVector([param[1]])
        o3d.write_pinhole_camera_trajectory("test.json", trajectory)

    def setView(self):
        ctr = vis.get_view_control()
        trajectory = o3d.read_pinhole_camera_trajectory("test.json")
        ctr.convert_from_pinhole_camera_parameters(trajectory.intrinsic, trajectory.extrinsic[0])

    def moveSphere(self):
        for ifor in range(10):
            self.sphere.transform(np.linalg.inv(self.sphereCurrentPose))
            self.sphereCurrentPose[:3,3] = self.sphereCurrentPose[:3,3] + np.array([0.25,0,0])
            self.sphere.transform(self.sphereCurrentPose)

            self.vis.update_geometry()
            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(0.24)

        time.sleep(4)



    def makeSphere(self, R = 1, center = None):
        if center is None:
            center = np.mean(self.xyz, axis=0)
        self.sphereCurrentPose = pose(center)
        sphere = o3d.create_mesh_sphere(R).transform(pose(center))
        sphere.paint_uniform_color([0.1, 0.7, 0.2])
        sphere.compute_vertex_normals()
        self.sphere = sphere
        return sphere

    def addSamples(self, R = None):
        for ifor in range(len(self.superPoints)):
            center, radius = self.superPoints[ifor]
            if R is None:
                R = radius
            sphere = o3d.create_mesh_sphere(R).transform(pose(center))
            sphere.paint_uniform_color([0.1, 0.1, 0.7])
            sphere.compute_vertex_normals()
            self.showObjects.append(sphere)

    def addCylinder(self, start, end, rotate=True, color=[0.9, 0.0, 0.3]):
        length = np.linalg.norm(start - end)
        n = (end - start) / length
        phi = np.arccos(n[2])
        theta = np.arctan2(n[1], n[0])

        theta_quat = Quaternion(axis=[0, 0, 1], angle=theta)
        vprime = theta_quat.rotate([0, 1., 0.])
        phi_quat = Quaternion(axis=vprime, angle=phi)
        rot = phi_quat.rotation_matrix

        cyl = o3d.create_mesh_cylinder(0.05, length)
        if rotate:
            cyl = cyl.transform(pose(np.array((start + end) / 2.0), rot))
        #     .transform(pose(center))
        cyl.paint_uniform_color(color)
        cyl.compute_vertex_normals()
        self.showObjects.append(cyl)

    def addHeads(self):
        if "head" in self.superPoints.df.keys():
            for ifor in range(len(self.superPoints)):
                start = self.superPoints.df.iloc[ifor].at['centers']
                head = self.superPoints.df.iloc[ifor].at['head']
                if head != -1:
                    try:
                        end = self.superPoints.df.loc[head].at['centers']
                        self.addCylinder(start, end)
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
                        self.addCylinder(start, end, True, [0,1,0])
                    except:
                        pass

if __name__ == '__main__':
    pairs = [('~/cheap.pcd', 'superPoints/pointsDataFrameB.pkl'),
             ('~/sites/tetraTech/BoilerRoom/chunk_cheap.pcd', 'superPoints/chunk_cheap.pkl'),
             ('~/sites/tetraTech/BoilerRoom/full_5mm.pcd', 'superPoints/full_5mm.pkl'),
             ('~/sites/tetraTech/BoilerRoom/chunk_cheap.pcd', 'superPoints/chunk_cheapB.pkl'),
             ('', 'superPoints/synthA.pkl')]
    pair = pairs[-2]

    superPoints = Samples()
    superPoints.load(pair[1])
    VR = OTest(pair[0], superPoints)
