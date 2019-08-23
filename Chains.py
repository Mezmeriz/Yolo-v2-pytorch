import numpy as np
import open3d as o3d
import pandas as pd
import ModelView
import time
import Samples
import scipy.spatial.kdtree as KD

def pose(xyz, rot = None):
    m = np.identity(4)
    m[0:3,3] = xyz
    if rot is not None:
        m[0:3, 0:3] = rot
    return m

COLOR_BLUE = [0,0,1]
COLOR_GREEN = [0.1, 0.9, 0.1]
COLOR_RED = [1, 0, 0]

class Chains:

    def __init__(self, samples):
        """
        Takes the data samples and finds the centers and radii.
        Prepare structures for neighbors
        :param df:
        """
        print(samples.df.keys())
        N = len(samples)
        self.N = N
        self.centers = np.zeros((N,3))
        self.radii = np.zeros((N,1))

        for i in range(N):
            center, R = samples[i]
            self.centers[i,:] = center
            self.radii[i] = R

        print("Making KD Tree")
        self.KD = KD.KDTree(self.centers)

        self.neighbors = [[] for i in range(N)]

    def findNeighbors(self):
        # self.neighbors = self.KD.query_ball_tree(other=self.KD, r=0.1)
        self.neighbors = []
        factor = 0.5
        for i in range(self.N):
            R = self.radii[i]
            connections = self.KD.query_ball_point(self.centers[i,:], R * factor)
            connections.remove(i)
            self.neighbors.append(connections)
        print(self.neighbors)

    def findMergeCandidates(self):
        reciprocals = self.findReciprocalConnectionCount()


    def findReciprocalConnectionCount(self):
        reciprocalConnections = []
        for i in range(self.N):
            conn = 0
            for j in self.neighbors[i]:
                if i in self.neighbors[j]:
                    conn += 1
            reciprocalConnections.append(conn)
        return reciprocalConnections

    def keyIndex(self, key):
        dfKeys = [i for i in self.df.keys()]
        return dfKeys.index(key)

    def getCenters(self):
        return self.dfChain['centers']

    def make(self):
        df = self.dfChain
        R = 0.015
        obj = []

        for i in range(len(df)):
            center = df.iloc[i, self.keyIndex('centers')]
            tailFlag, headFlag = df.iloc[i, [self.keyIndex('tailFlag'), self.keyIndex('headFlag')]]
            sphere = o3d.create_mesh_sphere(R, 12).transform(pose(center))
            if tailFlag:
                sphere.paint_uniform_color(COLOR_GREEN)
            elif headFlag:
                sphere.paint_uniform_color(COLOR_RED)
            else:
                sphere.paint_uniform_color(COLOR_BLUE)

            sphere.compute_vertex_normals()
            obj.append(sphere)
        self.obj = obj

    def save(self, fileName):
        self.df.to_pickle(fileName)





def makeStraight(df, viewer):
    """
    Go through each chain.
    Start one link in if possible
    go forward until
    :param superPoints:
    :return:
    """
    dfKeys = [i for i in df.keys()]
    print(dfKeys)
    CHAIN = dfKeys.index('chain')
    HEAD = dfKeys.index('head')

    straightSegments = []
    chains = df['chain'].unique()

    addedSegment = False

    chain = df.iloc[0, CHAIN]

    # Process chain
    start = df[df['tailFlag']]
    if len(start) > 0:
        dfPrint = df[['chain', 'head', 'tail', 'count', 'tailFlag', 'headFlag']]
        print(dfPrint.head(50))
        startIndex = start.index.tolist()[0]
        next = start[start.index == startIndex]

        headNormalStart = headNormal(df, start)
        inchWormHead = start
        inchWormSegments = 0
        thresholdStraightAngleDeg = 40
        radii = []
        infiniteLoop = set([])

        while (next is not None and headNormalStart is not None and next.at[next.index[0], 'chain'] == chain and
               next.index[0] not in infiniteLoop):

            try:
                print("{}, ".format(next.index[0]), end="")
                previous = next
                next = df.loc[[next.at[next.index[0], 'head']]]

                startI = int(np.flatnonzero(df.index == startIndex))
                nextI = int(np.flatnonzero(df.index == next.index[0]))
                viewer.colorAll()
                viewer.colorOne(startI, (1,0,0))
                viewer.colorOne(nextI, (0,0,1))
                viewer.updateNonBlocking()
                time.sleep(1)

                headNormalNext = headNormal(df, next)
                dot = headNormalStart.dot(headNormalNext)
                acos = np.arccos(dot) * 180 / np.pi
                print("{}, {}, {:4.2f} >? {:4.2f}: {}".format(headNormalStart, headNormalNext, dot,
                                           np.cos(thresholdStraightAngleDeg * np.pi / 180), acos))
                if (np.abs(headNormalNext.dot(headNormalStart)) > np.cos(thresholdStraightAngleDeg * np.pi/180)):
                    infiniteLoop = infiniteLoop.union(set([previous.index[0]]))
                    inchWormHead = next
                    inchWormSegments += 1
                    radius = (next.at[next.index[0], 'bx'] + next.at[next.index[0], 'by'])/4.0 * 1/448.
                    radii.append(radius)
                    print("i", end = "")
                else:
                    if inchWormSegments > 2:
                        straightSegments.append([chain, start.index[0], inchWormHead.index[0], np.median(radii)])
                        addedSegment = True
                    infiniteLoop = infiniteLoop.union(set([previous.index[0]]))
                    start = next
                    headNormalStart = headNormal(df, start)
                    inchWormHead = start
                    inchWormSegments = 0
                    radii = []



            except Exception as e:
                print(e)
                next = None
                if inchWormSegments > 2:
                    straightSegments.append([chain, start.index[0], previous.at[previous.index[0], 'head'], np.median(radii)])
                    addedSegment = True
            # head = next.at[next.index[0], 'head']
            # nextIndex = head
            # next = df.loc[[head]]

            #
        print("{}, {}".format(infiniteLoop, addedSegment))
    print("Segments created {}".format(len(straightSegments)))
    return straightSegments

def headNormal(df, start):
    p1 = start.at[start.index[0], 'centers']
    head = start.at[start.index[0], 'head']

    if head != -1 and head in df.index.tolist():
        p2 = df.at[head, 'centers']
        # print("P1 {}, P2 {}".format(p1, p2))
        n = p2 - p1
        n = n/np.linalg.norm(n)
        return n
    else:
        return None


def load(fileName):
    print("Loading {}".format(fileName))
    samples = pd.read_pickle(fileName)
    print("Length = {} samples".format(len(samples)))
    return Chains(samples)

if __name__ == '__main__':
    pairs = [('~/cheap.pcd', 'superPoints/pointsDataFrameB.pkl'),
             ('~/sites/tetraTech/BoilerRoom/chunk_cheap.pcd', 'superPoints/chunk_cheap.pkl'),
             ('~/sites/tetraTech/BoilerRoom/full_5mm.pcd', 'superPoints/full_5mm.pkl'),
             ('~/sites/tetraTech/BoilerRoom/chunk_cheap.pcd', 'superPoints/chunk_cheapB.pkl'),
             ('', 'superPoints/synthA.pkl')]
    pair = pairs[-1]

    superPoints = Samples.Samples()
    superPoints.load(pair[1])


    C = Chains(superPoints)
    C.findNeighbors()
    C.findMergeCandidates()

    MV = ModelView.ModelView()
    MV.addPoints(C.centers, C.radii)
    MV.update()


    # MV = ModelView.ModelView()
    # MV.addPoints(np.vstack(C.df['centers'].to_numpy()))
    #
    # chains = C.df['chain'].to_numpy()
    # score = C.df['objectness'].to_numpy()
    #
    # colors = {}
    # for index, chain in enumerate(chains):
    #     key = "{}".format(chain)
    #     if key not in colors.keys():
    #         newColor = np.random.random(3)
    #         newColor = newColor/np.max(newColor) * score[index]
    #         colors[key] = newColor
    #
    #     MV.colorOne(index, list(colors[key]))
    #
    # MV.update()

