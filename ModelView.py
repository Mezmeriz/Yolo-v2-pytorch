"""
ModelView

Intendet to help in the debugging process.
Someday I would like it to be multithreaded to allow non-blocking event processing while updating the model.
Currently, if events are processed, it runs in a blocking mode.
Non-blocking updates work, but then the mouse rotate events are not processed.

"""

import open3d as o3d
import numpy as np
import math

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

        self.pts =[]
        self.ptSpheres = []

    def update(self):
        self.vis.update_geometry()
        self.vis.poll_events()
        self.vis.update_renderer()
        self.vis.run()

    def updateNonBlocking(self):
        self.vis.update_geometry()
        self.vis.poll_events()
        self.vis.update_renderer()


    def colorOne(self, i, color = COLOR_RED):
        self.ptSpheres[i].paint_uniform_color(color)

    def colorAll(self):
        for s in self.ptSpheres:
            s.paint_uniform_color(COLOR_GREEN)

    def addPoints(self, pts, radius = None, color = COLOR_GREEN):
        pOriginal = np.array(self.pts)
        pNew = np.array(pts)

        if False: #pOriginal.shape[0]:
            self.pts = np.vstack((pOriginal, pNew))
        else:
            self.pts = pNew

        self.ptSpheres = self.make(radius, color)
        for s in self.ptSpheres:
            self.vis.add_geometry(s)


    def make(self, radius, color = COLOR_GREEN):
        """Make spheres for each point"""
        pts = self.pts

        if radius is None:
            radius = np.zeros(self.pts.shape[0]) + 0.05

        obj = []
        for i in range(pts.shape[0]):
            center = pts[i,:]
            sphere = o3d.create_mesh_sphere(radius[i], 12).transform(pose(center))
            sphere.paint_uniform_color(color)
            sphere.compute_vertex_normals()
            obj.append(sphere)
        return obj

    def addTube(self, start, end, radius, color=COLOR_GREEN):
        # Determine length and direction of tube
        vec = np.subtract(end, start)
        height = math.sqrt(np.dot(vec, vec))
        norm_vec = vec / height

        # Find axis of rotation, angle of rotation, create vector that encodes both
        perp = np.cross([0, 0, 1], norm_vec)
        dot = np.dot([0, 0, 1], norm_vec)
        angle = math.acos(dot)
        perp = perp * angle

        # Move end of cylinder to origin, rotate it, then translate to start pt.
        cyl = o3d.create_mesh_cylinder(radius, height)\
            .translate([0, 0, height/2])\
            .rotate(perp, center=False, type=o3d.RotationType.AxisAngle)\
            .translate(start)
        cyl.paint_uniform_color(color)
        cyl.compute_vertex_normals()
        self.vis.add_geometry(cyl)
