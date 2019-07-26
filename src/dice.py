import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

import cv2



pcd = o3d.io.read_point_cloud('/home/sadams/sites/tetraTech/Enfield Boiler Room/chunkSmallest.pcd')
xyz = np.asarray(pcd.points)

print(xyz.shape)
print(np.min(xyz, axis = 0))
print(np.max(xyz, axis = 0))

for ifor in range(10):
    center = [10.85-0.6, 0.5+ifor*0.2, 0.5]
    newxyz = xyz - center

    # plt.figure()
    II = (newxyz[:,1]> 1) & (newxyz[:,1]< 1.1)
    # plt.plot(newxyz[II,0], newxyz[II,2], 'k.')
    # plt.axis('equal')

    h, xe, ye = np.histogram2d(newxyz[II, 2], newxyz[II,0], (np.linspace(-0.5,0.5,449), np.linspace(-0.5,0.5,449)))
    m = np.max(h)
    h = (h/m * 254).astype(np.uint8)
    cv2.imshow("foo{}".format(ifor), h)
    cv2.imwrite("../test_images/boiler/foo{}.png".format(ifor), h)
cv2.waitKey()
