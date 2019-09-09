import numpy as np
import open3d as o3d
import pandas as pd
import Net
import os
from scipy import spatial
import cv2
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
import imutils
import Samples

def slice(xyz, mins, maxs):
    sliceWidth = 0.03
    newxyz = xyz
    delta = 1/448
    nx = int(np.ceil((maxs[1] - mins[1]) / delta) + 1)
    ny = int(np.ceil((maxs[2] - mins[2]) / delta) + 1)

    image = np.zeros((ny, nx, 3), dtype=np.uint8)
    for jfor in range(-1, 2):
        II = (newxyz[:, 0] >  jfor * sliceWidth - sliceWidth / 2) & (
                    newxyz[:, 0] < jfor * sliceWidth + sliceWidth / 2)
        h, xe, ye = np.histogram2d(newxyz[II, 2], newxyz[II, 1],
                                   (np.linspace(mins[2], maxs[2], ny+1), np.linspace(mins[1], maxs[1], nx+1)))
        m = np.max(h)
        if m > 0:
            h = (h / m * 255).astype(np.uint8)
        h[h > 0] = 255
        image[:, :, jfor + 1] = h
        # cv2.imshow("check", image)
        # cv2.waitKey(1)

    return image


def defineGridsProjected(xyz):
    """
    Make three sets of three dimensional grids.
    Locations and vectors for extraction need to be defined.
    :return:
    """
    mins = np.min(xyz, axis=0)
    maxs = np.max(xyz, axis=0)
    iranges = []
    focusDirection = 0
    for ifor in range(3):
        r = maxs[ifor] - mins[ifor]
        if ifor == focusDirection:
            stepSize = param['smallStep']
        else:
            stepSize = param['bigStep']

            # Only shrink the big step directions
            if r > param['W']:
                mins[ifor] += param['W'] / 2.0
                maxs[ifor] -= param['W'] / 2.0
                r = r - param['W']

        iranges.append(np.linspace(mins[ifor], maxs[ifor], np.ceil(r / stepSize) + 1))

    X, Y, Z = (iranges[0], iranges[1], iranges[2])
    total = np.prod([X.shape[0], Y.shape[0], Z.shape[0]])
    print('Number of grids = {} x {} x {} = {}: '.format(X.shape[0], Y.shape[0], Z.shape[0], total), end="")

    return [X, Y, Z]


def makeVectors(theta, elevation):

    theta_quat = Quaternion(axis=[0, 0, 1], angle=theta)
    vprime = theta_quat.rotate([0, 1., 0.])
    print(vprime)
    phi_quat = Quaternion(axis=vprime, angle=-elevation)
    spherical = phi_quat * theta_quat

    # print("{}\n".format(theta_quat.rotation_matrix))
    # print("{}\n".format(phi_quat.rotation_matrix))
    # print("{}\n".format((phi_quat.rotation_matrix).dot(theta_quat.rotation_matrix)))
    # print("{}\n".format(spherical.rotation_matrix))

    return spherical.rotation_matrix

def calcSpot(mins, maxs, depth):
    av = (mins + maxs) / 2.0
    return [av[0], av[1], depth]

class Dice():

    def __init__(self, fileIn, fileOut, samples, model):
        self.samples = samples
        self.model = model

        fileIn = os.path.expanduser(fileIn)
        print("Loading file {}".format(fileIn))
        self.pcd = o3d.io.read_point_cloud(fileIn)
        self.xyz = np.asarray(self.pcd.points)
        print("File loaded with {} points".format(self.xyz.shape[0]))

        # self.build_KDTree()
        Make_Movie = True
        count = 0
        fortyFive = np.pi/4
        for theta in [0]:
            eRange = [0*np.pi/2]

            for elevation in eRange:
                vectors = makeVectors(theta, elevation)
                print(vectors)
                projected = self.xyz.dot(vectors)
                mins = np.min(projected, axis=0)
                maxs = np.max(projected, axis=0)
                nDepth = int((maxs[0] - mins[0]) / (param['smallStep']/4))
                for depth in np.linspace(mins[0], maxs[0], nDepth + 1):
                    image = slice(projected - np.array([depth, 0, 0]), mins, maxs)
                    image = imutils.rotate(image, angle = 180)

                    spot = calcSpot(mins, maxs, depth)

                    if Make_Movie:
                        cv2.imwrite('movieTMP/img{:04d}.png'.format(count), image)

                    image = self.interpret(image, spot, vectors)

                    if Make_Movie:
                        cv2.imwrite('movieTMP/img{:04d}.jpg'.format(count), imutils.resize(image, width=800))
                        count += 1

                    if image.shape[0] < image.shape[1]:
                        image = imutils.resize(image, width = 448*3)
                    else:
                        image = imutils.resize(image, height=1000)



                    cv2.imshow("Slice", image)
                    cv2.waitKey(1)
                # self.scanGrids(XYZprojected, vectors)
                # self.samples.save(fileOut)


    def interpret(self, image, spot, vectors):
        threshold = 0.75
        nx = int(np.ceil(image.shape[1]/448.0))
        ny = int(np.ceil(image.shape[0]/448.0))
        for i in range(ny):
            for j in range(nx):
                startx = j * 448
                endx = min(image.shape[1], (j + 1) * 448)
                starty = i * 448
                endy = min(image.shape[0], (i + 1) * 448)

                sampleImage = np.zeros((448, 448, 3), dtype = np.uint8)
                sampleImage[0:(endy - starty), 0:(endx-startx), :] = image[starty:endy, startx:endx, :]

                predictions = self.model(sampleImage)
                if len(predictions) != 0:
                    for pi in range(len(predictions[0])):
                        pred = predictions[0][pi]
                        sample = {}
                        sample['coord'] = [(pred[0], pred[1])]
                        sample['bx'] = [pred[2]]
                        sample['by'] = [pred[3]]
                        sample['objectness'] = [pred[4]]
                        if pred[4] > threshold and pred[5] == 'circle':
                            image = cv2.rectangle(image, (int(pred[0] + startx), int(pred[1] + starty)),
                                                  (int(pred[0] + pred[2]+ startx), int(pred[1] + pred[3] + starty)),
                                                  (255, 255, 255), 1)

                        predictions[0][pi][0] += startx
                        predictions[0][pi][0] += starty

                    self.samples.add(predictions, spot, vectors)
        return image

    def build_KDTree(self):
        print("Building kd Tree")
        self.tree = spatial.KDTree(self.xyz)
        print("KD Tree complete")

    def getSample(self, sampleLocation, vec, sphereSize=1.5):
        subsetIndicies = self.tree.query_ball_point(x=sampleLocation, r=sphereSize)
        region = self.xyz[subsetIndicies, :]
        region = region - sampleLocation
        projection = np.matmul(region, vec)

        return slice(projection)



    def scanGrids(self, XYZ, vectors):
        R = np.sqrt(0.5**2 + 0.5**2 + 0.5**2)
        spot = 0
        x1Index = 0
        x2Index = 1
        x3Index = 2
        ymax = XYZ[x3Index].shape[0]
        xmax = XYZ[x2Index].shape[0]
        step = 228
        canvas = np.zeros((XYZ[x3Index].shape[0] * step + step, XYZ[x2Index].shape[0] * step + step, 3), dtype=np.uint8)
        for x1 in XYZ[x1Index]:
            xi = 0
            for x2 in XYZ[x2Index]:
                yi = 0
                for x3 in XYZ[x3Index]:
                    if spot and spot % 100 == 0:
                        print(".", end="", flush=True)

                    loc = np.array([x1, x2, x3]). dot(np.linalg.inv(vectors))
                    sampleImage = self.getSample(loc, vectors, R)
                    SKIP_NNET = True
                    if not SKIP_NNET:
                        predictions = self.model(sampleImage)
                        if len(predictions) != 0:
                            self.samples.add(predictions, loc, vectors)
                    spot = spot + 1

                    yii = ymax - 1 - yi
                    xii = xmax - 1 - xi
                    canvas[yii*step:yii*step+448,  xii*step:xii*step+448, :] = cv2.flip(imutils.rotate(sampleImage, angle = 90), flipCode=0)
                    yi += 1
                xi += 1
            cv2.imshow("Slice", imutils.resize(canvas, width = 2000))
            cv2.waitKey(1)

        print("")

if __name__ == '__main__':

    Ready = True
    if Ready:
        param = {'bigStep' : 0.5,
                 'smallStep' : 0.06,
                 'W' : 1}
        S = Samples.Samples()
        Yolo = Net.Yolo()
        #D = Dice('~/sites/tetraTech/BoilerRoom/chunkSmallest.pcd', 'superPoints/pointsDataFrameB.pkl', S, Yolo)

        D = Dice('~/sites/tetraTech/BoilerRoom/chunk_cheap.pcd', 'superPoints/fullSizeLabels.pkl', S, Yolo)
        #D = Dice('~/sites/tetraTech/BoilerRoom/full_5mm.pcd', 'superPoints/full_5mmB.pkl', S, Yolo)









