# import context
import numpy as np
import json
# from synth2D import LabelMaker
# from utils import JustThePaths
from pathlib import Path
import imutils
import pandas as pd

# import context
# from NNet import TrainingSets
import cv2
import argparse
from synthetic.Annotations import AnnotationsCombined as Anno

# with open(Path(JustThePaths.Paths.NEURAL_NET_DEFINITION) / 'nnet.json', 'r') as fh:
#     nnet = json.load(fh)
#
# N = nnet['N']
# labelMaker = LabelMaker.Label(nnet)

N = 448
newImageSize = 448

def view(anno, viewTime = 0):

    buf = 2
    ny = int( 1080/(newImageSize+buf))
    nx = int(1920/(newImageSize+buf))

    Ny = ny*(newImageSize+buf)
    Nx = nx*(newImageSize+buf)

    NPanes = nx*ny
    maxChunk = int(len(anno)/NPanes)
    startIndex = 0 #np.random.randint(0,maxChunk-1)*NPanes

    print("Starting Index {}, chunks {}".format(startIndex, NPanes))
    canvas = np.zeros((Ny, Nx ,3), dtype = np.uint8)
    canvas[:, :, 2] = 100

    CLASS_DATA_SIZE = 1+2*4+5

    K = ord('n')
    currentIndex = 0
    while (K != ord('q') and len(anno) > currentIndex):
        for index in range(NPanes):
            currentIndex = index + startIndex
            if currentIndex < len(anno):
                sample = anno[currentIndex]
                imgFile = anno.getImageInfo(index+startIndex)
                img = cv2.imread(imgFile)
                objects = anno.getBBoxes(currentIndex)

                for boundingBoxIndex in range(len(sample)):
                    xc, yc, bx, by, catID = objects[boundingBoxIndex,:]
                    xUpperLeft = xc
                    yUpperLeft = yc
                    xLowerRigt = xUpperLeft + bx
                    yLowerRight = yUpperLeft + by
                    img = cv2.rectangle(img, (int(xUpperLeft), int(yUpperLeft)),
                                        (int(xLowerRigt), int(yLowerRight)), (0,0,255), 1)

                img = imutils.resize(img, width=newImageSize)
                x = index % nx
                y = int(index/nx)
                px = x * (newImageSize + buf)
                py = y * (newImageSize + buf)
                if index < nx*ny:
                    canvas[py:py+newImageSize,px:px+newImageSize,:] = img

        cv2.imshow('Start'.format(startIndex), canvas)
        if viewTime > 0:
            cv2.waitKey(viewTime)
        else:
            K = cv2.waitKey(0)

        print("Len train {}, Start {}, Key {}".format(len(anno), startIndex, K))
        startIndex += NPanes

    cv2.destroyAllWindows()

def arguments(defaults = None):
    parser = argparse.ArgumentParser(description=
                                     'Viewer for neural net sample sets.')

    parser.add_argument('-a', '--annotations', required=True,
                        help='Path to the test/train annotations.')

    parser.add_argument('-s', '--subdir', default=None,
                        help='Subdirectory where the sets reside, e.g TRAIN, TEST, or VALIDATE.')
    parser.add_argument('-fp', '--filePrefix', default=None,
                        help='Subdirectory where the sets reside, e.g TRAIN, TEST, or VALIDATE.')

    parser.add_argument('-v', '--viewTime', default=0, type=int,
                        help='Time to wait between panes.')
    args = parser.parse_args(defaults)
    # print(args)
    return args

if __name__ == '__main__':
    # defaults = ['-p', '../tests/tmp/testShapes.pkl']
    args = arguments()
    anno = Anno(args.annotations)
    if args.filePrefix is not None:
        anno.filter('filePrefix', args.filePrefix)
    view(anno, args.viewTime)
