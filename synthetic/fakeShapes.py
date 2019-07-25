import cv2
import numpy as np
from abc import ABC, abstractmethod
import imutils
import os
import shutil

from pathlib import Path

from numpy.compat import os_PathLike

from Annotations import *

RESOLUTION = 448

def placement(radii, borders, window = 1):
    """Takes a list of radii,
    Creates a list of centers for each object assuming that the two should not overlap.
    Starts by putting the first one at the origin. Why not? It's all relative.
    Subsequent tries are within the window, i.e. typically +- 0.5m.
    """
    centers = []
    fails = 0
    minSeparation = 2.54e-2 # 1 inch
    for circleNumber in range(len(radii)):
        overlap = True
        while overlap:
            if circleNumber == 0:
                theta = np.random.random()*np.pi*2.0
                x, y = (0.5 + radii[0]*np.cos(theta),0.5 + radii[0]*np.sin(theta))
            else:
                x = np.random.random() * (window - 2 * borders[circleNumber][0]) + borders[circleNumber][0]
                y = np.random.random() * (window - 2 * borders[circleNumber][1]) + borders[circleNumber][1]
            overlap = False
            for index, center in enumerate(centers):
                distance = ((center[0] - x)**2 + (center[1] - y)**2)**0.5
                if distance <= (radii[circleNumber] + radii[index] + minSeparation):
                    overlap = True
                    fails += 1
            if fails>1000:
                x = -1
                y = -1
                overlap = False

        centers.append([x, y])

    #print("fails = {}".format(fails))
    return centers

class Canvas():
    def __init__(self):
        self.canvas = np.zeros((RESOLUTION, RESOLUTION, 3), dtype = np.uint8)
        self.N = 0

    def addShape(self, shape, center):
        self.N += 1
        thickness = np.random.randint(8,12)

        for ind in range(shape.x.shape[0]-1):
            x = shape.x[ind] + center[0]
            y = shape.y[ind] + center[1]
            x2= shape.x[ind+1] + center[0]
            y2 = shape.y[ind+1] + center[1]

            x = int(x * self.canvas.shape[1])
            y = int(y * self.canvas.shape[0])
            x2 = int(x2 * self.canvas.shape[1])
            y2 = int(y2 * self.canvas.shape[0])

            self.canvas = cv2.line(self.canvas, (x,y), (x2, y2), (255, 255, 255), thickness)

    def erode(self, percent = 10):
        II = np.where(self.canvas > 0)
        n = II[0].shape[0]
        IIrandom0 = np.random.randint(0, n, int(n * percent / 100.0))
        IIrandom1 = np.random.randint(0, n, int(n * percent / 100.0))
        IIrandom2 = np.random.randint(0, n, int(n * percent / 100.0))

        self.canvas[II[0][IIrandom0], II[1][IIrandom0], 0] = 0
        self.canvas[II[0][IIrandom1], II[1][IIrandom1], 1] = 0
        self.canvas[II[0][IIrandom2], II[1][IIrandom2], 2] = 0

    def vary(self, percent = 90):
        II = np.where(self.canvas > 0)
        n = II[0].shape[0]
        nsub = int(n * percent / 100.0)
        IIrandom0 = np.random.randint(0, n, nsub)
        IIrandom1 = np.random.randint(0, n, nsub)
        IIrandom2 = np.random.randint(0, n, nsub)

        self.canvas[II[0][IIrandom0], II[1][IIrandom0], 0] = np.random.randint(0, 254, nsub)
        self.canvas[II[0][IIrandom1], II[1][IIrandom1], 1] = np.random.randint(0, 254, nsub)
        self.canvas[II[0][IIrandom2], II[1][IIrandom2], 2] = np.random.randint(0, 254, nsub)

    def blur(self):
        self.canvas = cv2.GaussianBlur(self.canvas,(5,5),0)

    def __len__(self):
        return self.N

    def save(self, file):
        cv2.imwrite(file, self.canvas)

    def show(self):
        cv2.imshow('Canvas', cv2.flip(imutils.resize(self.canvas, width = 500), 0))
        return cv2.waitKey(0)

class Shape():

    CIRCLE = 0
    RECTANGLE = 1

    def __init__(self, args):
        self.dimensions = None
        self.x = None
        self.y = None
        self.percentage = 100
        self.lines = []
        self.borders = None

        # Override defaults with args
        for k in args.keys():
            setattr(self, k, args[k])

        self.make()

    @abstractmethod
    def make(self):
        pass

    @abstractmethod
    def bbox(self):
        return (1,1)

    def modify(self, percentage, center):

        x = np.copy(self.x)
        y = np.copy(self.y)

        N = self.x.shape[0]

        done = False
        count = 0
        while not done:
            start = np.random.randint(0,N)
            n = int(N * percentage / 100.0)
            indicies = np.arange(start, start+n) % N
            self.x = x[indicies]
            self.y = y[indicies]
            if self.inside(center):
                done = True
            elif count > 100:
                done = True
                self.x = x
                self.y = y
            count = count + 1
        return self

    def inside(self, center):
        x = self.x + center[0]
        y = self.y + center[1]
        if np.all((x > self.borders[0]) & (x < 1-self.borders[0])) and np.all((y > self.borders[1]) & (y < 1-self.borders[1])):
            return True
        else:
            return False

class Circle(Shape):
    SEGMENTS = 40
    RADII = np.arange(1,10) * 2.54e-2

    def __init__(self, **args):
        super().__init__(args)

        self.classNumber = Shape.CIRCLE
        self.name = 'circle'
        self.minFraction = 0.25
        self.borders = (0.01, 0.01)

    def make(self):
        startSegment = np.random.randint(0, Circle.SEGMENTS)
        thetaTwice = np.linspace(0,2, 2*Circle.SEGMENTS) * 2.0 * np.pi
        nSegments = np.round(self.percentage/100.0 * Circle.SEGMENTS)
        theta = thetaTwice[startSegment:startSegment + np.int(nSegments)]
        if hasattr(self, 'radius'):
            R = self.radius
        else:
            R = Circle.RADII[np.random.randint(0,Circle.RADII.shape[0])]
            self.radius = R

        self.x = R * np.cos(theta)
        self.y = R * np.sin(theta)

    def bbox(self):
        return (self.radius*2, self.radius*2)

class Rectangle(Shape):

    def __init__(self, **args):
        super().__init__(args)

        self.classNumber = Shape.RECTANGLE
        self.name = 'rectangle'
        self. minFraction = 0.8
        self.borders = (self.w, self.h)

    def getBoundingRadius(self):
        return np.sqrt((self.h / 2) ** 2 + (self.w / 2) ** 2)

    def extendLines(self, lines):
        try:
            # Check if lines[0] is iterable, i.e. is a list. Exception occurs if it is not iterable and just appends it.
            # Yes....this is the python way. Try and ask for forgiveness.
            _ = iter(lines[0])
            for line in lines:
                self.lines.append(line)

        except TypeError:
            self.lines.append(lines)


    def make(self):
        r = 0.02
        points = []
        points.append([-self.w / 2.0 + r, -self.h / 2])
        points.append([self.w / 2 - r, -self.h / 2])
        points.extend(makeArc(r, self.w/2-r, -self.h/2 + r, 270, 360))
        points.append([self.w / 2, self.h / 2 - r])
        points.extend(makeArc(r, self.w / 2-r, self.h / 2 - r, 0, 90))

        points.append([-self.w / 2 + r, self.h / 2])
        points.extend(makeArc(r, -self.w / 2 + r, self.h / 2 - r, 90, 180))

        points.append([-self.w /2 , -self.h / 2 +r])
        points.extend(makeArc(r, -self.w / 2 + r, -self.h / 2 + r, 180, 270))

        points = np.array(points)
        self.x = points[:, 0]
        self.y = points[:, 1]

    def bbox(self):
        return (self.w, self.h)


def makeArc(radius, cx, cy, start, stop, steps=5):
    points = []
    center = np.array([cx, cy])
    angles = np.linspace(start, stop, steps + 1)
    for angle in angles[1:]:
        angleRad = angle * np.pi / 180
        points.append(np.array([np.cos(angleRad), np.sin(angleRad)]) * radius + center)
    return points

def makeCircleInside(center, classIn, **args):
    done = False
    count = 0
    while not done:
        count+=1
        c = classIn(**args)
        done = (c.inside(center) or count > 100)

    if c.inside(center):
        return c
    else:
        print("Failed")
        return None

def selector():
    """ Return either a circle or a rectangle"""
    rectangles = [(5,10), (10,5), (6,6)]
    classNumber = np.random.randint(0,2)
    if classNumber == 0:
        radius = np.random.randint(1,5)* 2.54e-2;
        height = 0
        width = 0
        shape = Circle(radius = radius)
        return shape, radius
    elif classNumber == 1:
        radius = 0.04
        height, width = rectangles[np.random.randint(0,3)]
        height = height * 2.54e-2
        width = width * 2.54e-2
        shape = Rectangle(h = height, w = width)
        return shape, max(height, width)
    else:
        assert("Really?")

def makeSample(canvas, annotation, sampleIndex, N=5):
    """Makes a sample with N objects intended.
    If the random sizes fail to find a solution, fewer are possible."""

    radii = []
    shapes = []
    borders = []
    for ifor in range(N):
        shape, dimension = selector()
        radii.append(dimension)
        shapes.append(shape)
        borders.append(shape.borders)

    centers = placement(radii, borders)

    for ind, radius in enumerate(radii):
        fraction = np.random.random()

        if fraction < shapes[ind].minFraction:
            fraction = shapes[ind].minFraction
        percentage = fraction * 100

        c = shapes[ind].modify(percentage, centers[ind])
        if c is not None and centers[ind][0] != -1:
            canvas.addShape(c, center=centers[ind])
            if annotation is not None:
                annotation.add(sampleIndex, c.name, c.classNumber, centers[ind], c.bbox())

def convertSampleFiles(sampleFiles):
    path = Path(sampleFiles[0])
    annotationFilename = path / "annotations" / (sampleFiles[1] + "_{}.pkl".format(sampleFiles[2]))
    dataFilenameTemplate = path / "images" / sampleFiles[2] / (sampleFiles[1] + "_{:05d}.png")
    if not path.exists():
        os.makedirs()(path)
    if not (path / "annotations" ).exists():
        os.makedirs(annotationFilename.parent)
    if not (path / "images" / sampleFiles[2]).exists():
        os.makedirs(dataFilenameTemplate.parent)

    print("Annotation file: {}".format(annotationFilename.as_posix()))
    print("Data file template: {}".format(dataFilenameTemplate.as_posix()))

    return annotationFilename, dataFilenameTemplate.as_posix()

def makeSampleSet(sampleFiles, maxSampleIndex, refresh):
    done = False
    sampleIndex = 0
    annotationFilename, dataFilenameTemplate = convertSampleFiles(sampleFiles)
    anno = Annotations(annotationFilename, refresh)

    while (not done):
        canvas = Canvas()
        makeSample(canvas, anno, sampleIndex, 5)
        if len(canvas):
            canvas.vary()
            canvas.erode(25)
            canvas.blur()
            cv2.imwrite(dataFilenameTemplate.format(sampleIndex), canvas.canvas)
            sampleIndex += 1

        INTERACTIVE = False
        if INTERACTIVE:

            key = canvas.show()
            if key == ord('q'):
                done = True
        else:
            if sampleIndex >= maxSampleIndex:
                done = True

        if sampleIndex % 100 == 0:
            print(".", end="", flush=True)
        if sampleIndex % 1000 == 0:
            print("\n{}".format(sampleIndex // 1000), end="", flush=True)

    cv2.destroyAllWindows()
    anno.save()

def showSampleSet():
    done = False

    while (not done):
        canvas = Canvas()
        makeSample(canvas, None, None, 5)

        INTERACTIVE = True
        if INTERACTIVE:
            canvas.vary()
            canvas.erode(25)
            canvas.blur()
            key = canvas.show()
            if key == ord('q'):
                done = True


    cv2.destroyAllWindows()


if __name__ == '__main__':
    """Note: Currently the canvas size represents 1m x 1m. Centers and box sizes
    are in meters, but coincidentally are fractions of the image size. IF THE 
    CANVAS SIZE IS CHANGED, E.G. 1.2M X 1.2M CENTERS AND BOX SIZES WILL NEED TO BE
    NORMALIZED TO REPRESENT FRACTIONS OF THE IMAGE."""

    BUILD = True
    if BUILD:
        maxSampleIndex = 5000
        sampleFiles = ("../data/TwoShapes", "test1", "train")
        makeSampleSet(sampleFiles, maxSampleIndex, refresh=True)

        maxSampleIndex = 500
        sampleFiles = ("../data/TwoShapes", "test1", "val")
        makeSampleSet(sampleFiles, maxSampleIndex, refresh=True)
    else:
        showSampleSet()
