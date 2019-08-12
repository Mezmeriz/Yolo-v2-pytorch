import open3d as o3d
from pyquaternion import Quaternion
import numpy as np

def pose(xyz, rot = None):
    m = np.identity(4)
    m[0:3,3] = xyz
    if rot is not None:
        m[0:3, 0:3] = rot
    return m

def addSphere(center, radius):
    sphere = o3d.create_mesh_sphere(radius).transform(pose(center))
    sphere.paint_uniform_color([0.1, 0.1, 0.7])
    sphere.compute_vertex_normals()
    return sphere

def addCylinder(start, end, rotate = True, color = [0.9,0.0,0.3]):
    length = np.linalg.norm(start - end)
    n = (end - start) / length
    phi = np.arccos(n[2])
    theta = np.arctan2(n[1], n[0])

    theta_quat = Quaternion(axis = [0, 0, 1], angle = theta)
    vprime = theta_quat.rotate([0, 1., 0.])
    phi_quat = Quaternion(axis = vprime, angle = phi)
    rot = phi_quat.rotation_matrix
    print(rot)

    cyl = o3d.create_mesh_cylinder(0.05, length)
    if rotate:
        cyl = cyl.transform(pose(np.array((start+end)/2.0), rot))
    #     .transform(pose(center))
    cyl.paint_uniform_color(color)
    cyl.compute_vertex_normals()
    return cyl

start = np.array([0,0,0])
end = np.array([1,1,1])


pcd = [addSphere(start, 0.1)]
pcd.append(addSphere(end, 0.1))
pcd.append(addCylinder(start, end))
pcd.append(addCylinder(start, end, rotate = False, color = [0,1,0]))
o3d.draw_geometries(pcd)