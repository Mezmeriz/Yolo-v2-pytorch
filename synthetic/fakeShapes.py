import cv2
import numpy as np
from abc import ABC, abstractmethod
import imutils
import os
import shutil

from pathlib import Path

from numpy.compat import os_PathLike

from synthetic.Annotations import *

RESOLUTION = 448
FLAG_FOR_FAILED_PLACEMENT = -1000

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
            x = np.random.random() * (window - 2 * borders[circleNumber][0]) + borders[circleNumber][0]
            y = np.random.random() * (window - 2 * borders[circleNumber][1]) + borders[circleNumber][1]
            overlap = False
            for index, center in enumerate(centers):
                distance = ((center[0] - x)**2 + (center[1] - y)**2)**0.5
                if distance <= (radii[circleNumber] + radii[index] + minSeparation):
                    overlap = True
                    fails += 1
            if fails>1000:
                x = FLAG_FOR_FAILED_PLACEMENT
                y = FLAG_FOR_FAILED_PLACEMENT
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
        thickness = np.random.randint(5,10)

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
        self.buffer = None

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
            start = np.random.randint(0, N)
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
        if np.all((x > self.buffer) & (x < 1-self.buffer)) and np.all((y > self.buffer) & (y < 1-self.buffer)):
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
        self.buffer = 0

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
        return np.array([-self.radius, -self.radius, self.radius, self.radius])

class Rectangle(Shape):

    def __init__(self, **args):
        super().__init__(args)

        self.classNumber = Shape.RECTANGLE
        self.name = 'rectangle'
        self. minFraction = 0.8
        buffer = 0.03
        self.borders = (self.w/2. + buffer, self.h/2. + buffer)
        self.buffer = buffer

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
        return np.array([-self.w/2.0, -self.h/2.0, self.w/2.0, self.h/2.0])


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
    # Rectangles expressed in terms of feature sizes, i.e. 1m/14
    rectangles = [(2, 4), (4, 2), (2,2), (5, 10), (10, 5), (6,6)]
    classNumber = np.random.randint(0,2)
    if classNumber == 0:
        radius = np.random.randint(1, 12)* 2.54e-2;
        height = 0
        width = 0
        shape = Circle(radius = radius)
        return shape, radius
    elif classNumber == 1:
        radius = 0.04
        height, width = rectangles[np.random.randint(0, len(rectangles))]
        fac = 1.0/14
        height = height * fac
        width = width * fac
        shape = Rectangle(h = height, w = width)
        dim = np.sqrt((height**2)/4. + (width**2)/4.)
        return shape, dim
    else:
        assert("Really?")

def kludge(bbox):
    """Annotations from one synthetic source are flipped x/y.
    the primary annotation setup flips it to correct the data.
    But that means correctly formed data does not get through.
    You got the idea. If this is around in a week, please shoot me.
    """
    return np.array([bbox[1], bbox[0], bbox[3], bbox[2]])

def makeSample(canvas, annotation, sampleIndex, N=1):
    """Makes a sample with N objects intended.
    If the random sizes fail to find a solution, fewer are possible.

    Steps
        1. Randomly select 5 shapes
        2. Determine placement to avoid overlaps
        3. Erase some of each shape
        4. """

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

        allInsidePriorToModify = shapes[ind].inside(centers[ind])
        c = shapes[ind].modify(percentage, centers[ind])
        if c.inside(centers[ind]) and centers[ind][0] != FLAG_FOR_FAILED_PLACEMENT:
            canvas.addShape(c, center=centers[ind])
            if annotation is not None:
                annotation.add(sampleIndex, c.name, c.classNumber, kludge(c.bbox()), centers[ind])

def convertSampleFiles(sampleFiles):
    path = Path(os.path.expanduser(sampleFiles[0]))
    annotationFilename = path / "annotations" / (sampleFiles[1] + "_anno.pkl")

    dataFilenameTemplate = path / "images" / sampleFiles[1] / (sampleFiles[1] + "_{:05d}.png")
    if not path.exists():
        os.makedirs(path)
    if not (path / "annotations" ).exists():
        os.makedirs(annotationFilename.parent)
    if not (path / "images" / sampleFiles[1]).exists():
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
        makeSample(canvas, anno, sampleIndex, 7)
        if len(canvas):
            canvas.vary()
            canvas.erode(25)
            canvas.blur()
            fileOut =imageFileFromIndex(Path(os.path.expanduser(sampleFiles[0])) / "annotations" , sampleFiles[1], sampleIndex)
            cv2.imwrite(fileOut.as_posix(), canvas.canvas)
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
        makeSample(canvas, None, None, 10)

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

    BUILD = False
    if BUILD:
        maxSampleIndex = 1000
        sampleFiles = ("~/dataNeural/yolo1", "fakePositive1k")
        makeSampleSet(sampleFiles, maxSampleIndex, refresh=True)
    else:
        showSampleSet()
