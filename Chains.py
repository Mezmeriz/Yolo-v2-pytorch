import numpy as np
import open3d as o3d
import pandas as pd

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

    def __init__(self, df, chainNumber = None):
        self.df = df

        if chainNumber is None:
            self.chainNumber = None
            self.dfChain = None
        else:
            self.setChain(chainNumber)

    def setChain(self, chainNumber):
        self.chainNumber = chainNumber
        self.dfChain = self.df[self.df['chain'] == chainNumber]
        self.make()

    def keyIndex(self, key):
        dfKeys = [i for i in self.df.keys()]
        return dfKeys.index(key)

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

    def show(self):
        o3d.draw_geometries(self.obj)

    def showInteractive(self):
        self.custom_draw_geometry_with_key_callback(self.obj)

    def save(self, fileName):
        self.df.to_pickle(fileName)

    def custom_draw_geometry_with_key_callback(self, pcd):

        highlightIndex = 0
        self.obj[highlightIndex].paint_uniform_color(COLOR_GREEN)

        def next(vis):
            nonlocal highlightIndex
            highlightIndex += 1
            self.obj[highlightIndex].paint_uniform_color(COLOR_GREEN)
            vis.update_geometry()
            # vis.poll_events()
            vis.update_renderer()

        key_to_callback = {}
        key_to_callback[ord("N")] = next
        o3d.visualization.draw_geometries_with_key_callbacks(pcd, key_to_callback)


def makeStraight(superPoints):
    """
    Go through each chain.
    Start one link in if possible
    go forward until
    :param superPoints:
    :return:
    """
    dfKeys = [i for i in superPoints.df.keys()]
    print(dfKeys)
    CHAIN = dfKeys.index('chain')
    HEAD = dfKeys.index('head')

    straightSegments = []
    chains = superPoints.df['chain'].unique()
    inspectChain = -1000
    for chain in chains:


        df = superPoints.df[superPoints.df['chain']==chain]
        addedSegment = False

        # Process chain
        start = df[df['tailFlag']]
        if len(start) > 0:
            dfPrint = df[['chain', 'head', 'tail', 'count', 'tailFlag', 'headFlag']]
            print(dfPrint.head(50))
            startIndex = start.index.tolist()[0]
            print("Chain {}: ".format(chain), end="")
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
                    headNormalNext = headNormal(df, next)
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

                        # print(np.abs(headNormalNext.dot(headNormalStart)))
                        # print(headNormalStart)
                        # print(headNormalNext)
                        # print()

                except:
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
    p1 = start.at[start.index[0], 'xyz']
    head = start.at[start.index[0], 'head']

    if head != -1 and head in df.index.tolist():
        p2 = df.at[head, 'xyz']
        # print("P1 {}, P2 {}".format(p1, p2))
        n = p2 - p1
        n = n/np.linalg.norm(n)
        return n
    else:
        return None


def load(fileName):
    print("Loading {}".format(fileName))
    df = pd.read_pickle(fileName)
    print("Length = {} samples".format(df.shape[0]))
    return Chains(df)

if __name__ == '__main__':
    C = load('tmp/chain981.pkl')
    C.setChain(981)
    C.showInteractive()
