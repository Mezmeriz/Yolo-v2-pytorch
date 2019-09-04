"""
ModelView

Intendet to help in the debugging process.
Someday I would like it to be multithreaded to allow non-blocking event processing while updating the model.
Currently, if events are processed, it runs in a blocking mode.
Non-blocking updates work, but then the mouse rotate events are not processed.

"""

import open3d as o3d
import numpy as np
from pyquaternion import Quaternion
import numpy.matlib as matlib
import time

def pose(xyz, rot = None):
    m = np.identity(4)
    m[0:3,3] = xyz
    if rot is not None:
        m[0:3, 0:3] = rot
    return m

COLOR_BLUE = [0,0,1]
COLOR_GREEN = [0.1, 0.9, 0.1]
COLOR_RED = [1, 0, 0]

class ModelView():

    def __init__(self):
        self.pts = None
        self.vis = o3d.Visualizer()
        self.vis.create_window()

        self.objects = []

    def update(self):
        self.vis.update_geometry()
        self.vis.poll_events()
        self.vis.update_renderer()
        self.vis.run()

    def updateNonBlocking(self):
        self.vis.update_geometry()
        self.vis.poll_events()
        self.vis.update_renderer()

    def addPoints(self, pts, colors):
        pcd = o3d.PointCloud()
        pcd.points = o3d.Vector3dVector(pts)

        colors = np.array(colors).reshape((-1,3))
        if colors.shape[0] == 1:
            colors = np.matlib.repmat(colors, pts.shape[0],1)

        pcd.colors = o3d.Vector3dVector(colors)

        self.vis.add_geometry(pcd)
        self.objects.append(pcd)
        return pcd, len(self.objects) - 1

    def addCylinder(self, start, end, color=[0.9, 0.0, 0.3], radius = None):
        start = np.array(start)
        end = np.array(end)

        defaultCylinderResolution = 16
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

        cyl = o3d.create_mesh_cylinder(radius, length, resolution=defaultCylinderResolution)
        cyl = cyl.transform(pose(np.array((start + end) / 2.0), rot))
        cyl.paint_uniform_color(color)
        cyl.compute_vertex_normals()

        self.vis.add_geometry(cyl)
        self.objects.append(cyl)
        return cyl, len(self.objects) - 1

    def addSphere(self, point, radius, color = COLOR_GREEN):
        """Make a sphere at point"""

        defaultSphereSegments = 12
        sphere = o3d.create_mesh_sphere(radius, defaultSphereSegments).transform(pose(point))
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()

        self.vis.add_geometry(sphere)
        self.objects.append(sphere)
        return sphere, len(self.objects) - 1

    def addSpheres(self, points, radii, color = COLOR_GREEN):
        """Make a sphere at point"""
        defaultSphereSegments = 12
        if type(radii) == float:
            radii = np.zeros_like(points[:,0]) + radii

        spheres = []
        for index in range(points.shape[0]):
            sphere, index = self.addSphere(points[index,:], radii[index], color)
            spheres.append(sphere)

        return spheres, len(self.objects) - 1

    def removeObject(self, index):
        print("Removing object {}".format(index))
        self.vis.remove_geometry(self.objects[index])

if __name__ == '__main__':

    MV = ModelView()

    N = 10
    pts0 = np.random.random((N, 3))
    colors = np.random.random((N, 3))
    pcd_points, pIndex = MV.addPoints(pts0, colors)
    print("Added object index = {}".format(pIndex))
    MV.updateNonBlocking()

    N = 10
    pts = np.random.random((N, 3))
    colors = [0,1,1]
    pcd_points, pIndex = MV.addPoints(pts, colors)
    print("Added object index = {}".format(pIndex))
    MV.updateNonBlocking()

    point = [0,0,0]
    radius = 0.1
    color = [1, 0, 0]
    pcd_sphere, sIndex = MV.addSphere(point, radius, color)
    print("Added object index = {}".format(sIndex))

    start = [1,0,0]
    end = [2, 0,0]
    radius = 0.2
    color = [0.5, 0, 0.7]
    pcd_cylinder, cIndex = MV.addCylinder(start, end, color, radius)
    print("Added object index = {}".format(cIndex))
    MV.updateNonBlocking()

    # for ifor in range(100):
    #     time.sleep(0.05)
    #     MV.updateNonBlocking()

    MV.removeObject(0)
    MV.addSpheres(pts0, 0.1)

    for ifor in range(100):
        time.sleep(0.05)
        MV.updateNonBlocking()





