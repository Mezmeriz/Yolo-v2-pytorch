import open3d as o3d
import numpy as np

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

    def addPoints(self, pts, radius = None):
        pOriginal = np.array(self.pts)
        pNew = np.array(pts)

        if pOriginal.shape[0]:
            self.pts = np.vstack((pOriginal, pNew))
        else:
            self.pts = pNew

        self.ptSpheres = self.make(radius)
        for s in self.ptSpheres:
            self.vis.add_geometry(s)


    def make(self, radius):
        """Make spheres for each point"""
        pts = self.pts

        if radius is None:
            radius = np.zeros(self.pts.shape[0]) + 0.05

        obj = []
        for i in range(pts.shape[0]):
            center = pts[i,:]
            sphere = o3d.create_mesh_sphere(radius[i], 12).transform(pose(center))
            sphere.paint_uniform_color(COLOR_GREEN)
            sphere.compute_vertex_normals()
            obj.append(sphere)
        return obj