"""
Chains

During the rip process, superpoints are extracted from the point cloud and saved for later use.
This routine loads in the superpoints and tries to filter and connect the superpoints.
I had a method for working my way through straight sections, but I went back to the filtering/interconnection part.

Useful information in terms of sizing:
    NNET steps are 6cm.
    I thought 2.2 stepSizes was a reasonable minimum connection distance.

Next up:
    Connect the dots to nearest neighbors
    Originally, I used the directions from the dicing routines to establish a front and a back. One connection was
    made to each. I'm trying a new approach now beacause the front and back from the dicing routine has no real
    relation to the pipe directions. Instead, I may just
        1) find the nearest neighbors < max(Radius or 2.2 * STEP SIZE)
        2) some sort of best fit lines to ever increasing group sizes

"""
from typing import List, Set, Dict, Any
import numpy as np
import open3d as o3d
import pandas as pd
import ModelView
import time
import Samples
import scipy.spatial.kdtree as KD
from abc import abstractmethod

def pose(xyz, rot = None):
    m = np.identity(4)
    m[0:3,3] = xyz
    if rot is not None:
        m[0:3, 0:3] = rot
    return m

COLOR_BLUE = [0,0,1]
COLOR_GREEN = [0.1, 0.9, 0.1]
COLOR_RED = [1, 0, 0]

MERGE_ALWAYS_DISTANCE = 0.05
STEP_SIZE = 0.06
STEP_SIZE_FACTOR = 2.2

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

        # Parameters that need updates after size changes
        self.KD = None
        self.neighbors = None
        self.distances = None

        self.mergeCloselySpaced()
        self.removeOrphans()
        self.chains = self.connect()

    def mergeCloselySpaced(self):
        """
        Merge closely spaced superpoints
        :return:
        """
        print("Merging superpoints < {:3.1f} cm apart".format(MERGE_ALWAYS_DISTANCE*100))
        self.update()
        merged = True
        count = 0
        while merged:
            mergeList = self.findMergeCandidates()
            if mergeList is not None:
                self.merge(mergeList)
                self.update()
            else:
                merged = False
            count += 1
            assert count < 100, "Really? {} merges. Probably something wrong".format(count)
        print("")

    def update(self):
        """Done at the beginning of an operation to ensure fresh data"""
        self.N = self.centers.shape[0]
        print("Updating: Found {} objects, ".format(self.N), end = "")
        print("Making KD Tree")
        self.KD = KD.KDTree(self.centers)
        self.findNeighbors()

    def findNeighbors(self, factor = 0.5):
        """
        Determine all of the neighbors within some fraction of the radius'
        :parms:
        factor is a multiplier on the raidus
        :return:
        """

        self.neighbors = []
        self.distances = []

        for i in range(self.N):
            R = self.radii[i]
            distanceThreshold = max([R*factor, STEP_SIZE_FACTOR * STEP_SIZE])
            connections = self.KD.query_ball_point(self.centers[i, :], distanceThreshold)
            connections.remove(i)
            self.neighbors.append(connections)
            self.distances.append(self.neighborDistances(i))

    def findMergeCandidates(self):
        """
        Find the reciprocal connection count. At one point I wanted to merge the smallest connection among groups >= 3 connections.
        Maybe not anymore.

        Make a unique merge list with the smaller node number first.
        """
        # reciprocals = self.findReciprocalConnectionCount()
        distances = self.pairDistances()

        mergeList = []
        for ni in range(self.N):
            for j, nj in enumerate(self.neighbors[ni]):
                if len(distances[ni]) and distances[ni][j] < MERGE_ALWAYS_DISTANCE:
                    mergeList.append([ni, nj])

        if len(mergeList):
            mergeList = np.array(mergeList, dtype = np.int)
            mergeList.sort(axis = 1)
            mergeList = np.unique(mergeList, axis = 0)
            # print(mergeList)
            # self.filterMergeList() TODO Don't want to merge dissimlar radii
            return mergeList
        else:
            return None


    def merge(self, mergeList):
        """
        Average the nodes to be merged.
        Put result in lower index of each pair.
        Kill the upper index of each pair.

        :param mergeList:
        :return:
        """
        keepSet = {i for i in range(self.N)}
        killSet = []

        for i,j in mergeList:
            if i not in killSet and j not in killSet:
                self.centers[i] = (self.centers[i] + self.centers[j]) / 2.0
                self.radii[i] = (self.radii[i] + self.radii[j]) / 2.0
                killSet.append(j)
        killSet = set(killSet)
        keepSet = keepSet.difference(killSet)

        self.centers = self.centers[list(keepSet), :]
        self.radii = self.radii[list(keepSet), :]

    def removeOrphans(self):
        """
        Average the nodes to be merged.
        Put result in lower index of each pair.
        Kill the upper index of each pair.

        :param mergeList:
        :return:
        """
        keepSet = {i for i in range(self.N)}
        killSet = []

        for index, neighbors in enumerate(self.neighbors):
            if not len(neighbors):
                killSet.append(index)
        killSet = set(killSet)
        keepSet = keepSet.difference(killSet)

        self.centers = self.centers[list(keepSet), :]
        self.radii = self.radii[list(keepSet), :]

    def pairDistances(self):
        distances = []
        for i in range(self.N):
            distancesI = self.neighborDistances(i)
            distances.append(distancesI)
        return distances

    def neighborDistances(self, i):
        distanceI = []
        for j in self.neighbors[i]:
            distance = np.linalg.norm(self.centers[i] - self.centers[j])
            distanceI.append(distance)
        return distanceI

    def findReciprocalConnectionCount(self):
        reciprocalConnections = []
        for i in range(self.N):
            conn = 0
            for j in self.neighbors[i]:
                if i in self.neighbors[j]:
                    conn += 1
            reciprocalConnections.append(conn)
        return reciprocalConnections

    def direction(self, i, j):
        u = self.centers[j, :] - self.centers[i, :]
        distance = np.linalg.norm(u)
        u = u / distance
        return u, distance

    def nearest(self, i):
        return np.neighbors(np.argmin(self.neighborDistances(i)))

    def getDistance(self, i, j):
        index = self.neighbors.index(j)
        return self.neighborDistances(index)

    def connect(self):
        consumed = {}
        chains = []
        for i in range(self.N):
            if i not in consumed:
                neigbors = self.neighbors[i]
                if len(neigbors):
                    nearestNeighbor = nearest(i)
                    nearestDirection, distance = direction(i, nearestNeighbor)
                    chain, consumedChain = self.chain(i, nearestNeighbor, nearestDirection, consumed)
                    chains.append(chain)
                    consumed.update(consumedChain)
        return chains

    def chain(self, index, nearestNeighbor, nearestDirection, consumed):
        links = []
        link0 = Link(index, nearestNeighbor, nearestDirection, self.radii[index], self.centers[index])
        link = link0

        # Follow forward, first link already created
        done = False
        while not done:
            links.append(link)
            consumed.update(link.consumed)
            next = follow(Link.FORWARD, link)
            if next is not None:
                link.next = next
                link = next
            else:
                done = True

        # Follow back
        done = False
        link = link0
        while not done:
            previous = follow(Link.BACKWARD, link)
            if previous is not None:
                links.insert(previous)
                consumed.update(link.consumed)
                link.previous = previous
                link = previous
            else:
                done = True

    def add(self, index, center, radius):
        self.indicies.append(index)
        self.centers.append(radius)

    def follow(self, forward, link):

        nearestNeighbor = self.nearestWithDirectionLimit(link, forward)
        if nextNextIndex is not None:
            nextLink = Link(forward, link.nextIndex, nearestNeighbor,
                            self.direction(nearestNeighbor, link.index)[0],
                            self.getDistance(link.index, nearestNeighbor), self.radii[nearestNeighbor],
                            self.centers[nearestNeighbor])
            return nextLink
        else:
            return None

    def nearestWithDirectionLimit(self, link, forward):
        """ find the subset of neighbors that are allong the follow direction.
        Find the one with the minimum distance.
        Return that as the next link.
        """
        angularThreshold = np.cos(45 * np.pi/180)

        if forward:
            direction = link.direction
            index = link.nextIndex
        else:
            direction = -1.0 * link.direction
            index = link.previousIndex

        neighbors = self.neighbors[index]
        distances = self.distances[index]
        centers = self.centers[neighbors]
        centersLocal = centers - link.center
        directions = centersLocal / distances
        dotProduct = np.matmul(directions, direction)
        II = np.where(dotProduct > angularThreshold)[0]
        if len(II):
            # find the minimum distance of the remaining
            II2 = np.argmin(distances[II])
            nearestNeighbor = II[II2]
            return nearestNeighbor
        else:
            return None




class Link():
    FORWARD = True
    BACKWARD = False

    def __init__(self, forward, index, nearestNeighbor, nearestDirection, length, radius, center):
        """
        Use the node at index and the direction to find the next node in the chain.
        :param index: link node index
        :param neighbors: neighbor node indicies
        :param direction: unit vector = Forward
        :param centers: all node centers
        :param radii:
        """
        self.center = center
        self.index = index

        if forward:
            self.nextIndex = nearestNeighbor
            self.direction = np.copy(nearestDirection)
        else:
            self.previousIndex = nearestNeighbor
            self.direction = np.copy(nearestDirection)

        self.length = length
        self.consumed = nearestNeighbor
        self.radius = radius

        # To be updated
        self.next = None
        self.previous = None

    @property
    def terminated(self):
        return self.nextIndex is None

    def follow(self, direction, index, centers):
        "Factory for the next link"

# def makeStraight(df, viewer):
#     """
#     Go through each chain.
#     Start one link in if possible
#     go forward until
#     :param superPoints:
#     :return:
#     """
#     dfKeys = [i for i in df.keys()]
#     print(dfKeys)
#     CHAIN = dfKeys.index('chain')
#     HEAD = dfKeys.index('head')
#
#     straightSegments = []
#     chains = df['chain'].unique()
#
#     addedSegment = False
#
#     chain = df.iloc[0, CHAIN]
#
#     # Process chain
#     start = df[df['tailFlag']]
#     if len(start) > 0:
#         dfPrint = df[['chain', 'head', 'tail', 'count', 'tailFlag', 'headFlag']]
#         print(dfPrint.head(50))
#         startIndex = start.index.tolist()[0]
#         next = start[start.index == startIndex]
#
#         headNormalStart = headNormal(df, start)
#         inchWormHead = start
#         inchWormSegments = 0
#         thresholdStraightAngleDeg = 40
#         radii = []
#         infiniteLoop = set([])
#
#         while (next is not None and headNormalStart is not None and next.at[next.index[0], 'chain'] == chain and
#                next.index[0] not in infiniteLoop):
#
#             try:
#                 print("{}, ".format(next.index[0]), end="")
#                 previous = next
#                 next = df.loc[[next.at[next.index[0], 'head']]]
#
#                 startI = int(np.flatnonzero(df.index == startIndex))
#                 nextI = int(np.flatnonzero(df.index == next.index[0]))
#                 viewer.colorAll()
#                 viewer.colorOne(startI, (1,0,0))
#                 viewer.colorOne(nextI, (0,0,1))
#                 viewer.updateNonBlocking()
#                 time.sleep(1)
#
#                 headNormalNext = headNormal(df, next)
#                 dot = headNormalStart.dot(headNormalNext)
#                 acos = np.arccos(dot) * 180 / np.pi
#                 print("{}, {}, {:4.2f} >? {:4.2f}: {}".format(headNormalStart, headNormalNext, dot,
#                                            np.cos(thresholdStraightAngleDeg * np.pi / 180), acos))
#                 if (np.abs(headNormalNext.dot(headNormalStart)) > np.cos(thresholdStraightAngleDeg * np.pi/180)):
#                     infiniteLoop = infiniteLoop.union(set([previous.index[0]]))
#                     inchWormHead = next
#                     inchWormSegments += 1
#                     radius = (next.at[next.index[0], 'bx'] + next.at[next.index[0], 'by'])/4.0 * 1/448.
#                     radii.append(radius)
#                     print("i", end = "")
#                 else:
#                     if inchWormSegments > 2:
#                         straightSegments.append([chain, start.index[0], inchWormHead.index[0], np.median(radii)])
#                         addedSegment = True
#                     infiniteLoop = infiniteLoop.union(set([previous.index[0]]))
#                     start = next
#                     headNormalStart = headNormal(df, start)
#                     inchWormHead = start
#                     inchWormSegments = 0
#                     radii = []
#
#
#
#             except Exception as e:
#                 print(e)
#                 next = None
#                 if inchWormSegments > 2:
#                     straightSegments.append([chain, start.index[0], previous.at[previous.index[0], 'head'], np.median(radii)])
#                     addedSegment = True
#             # head = next.at[next.index[0], 'head']
#             # nextIndex = head
#             # next = df.loc[[head]]
#
#             #
#         print("{}, {}".format(infiniteLoop, addedSegment))
#     print("Segments created {}".format(len(straightSegments)))
#     return straightSegments
#
# def headNormal(df, start):
#     p1 = start.at[start.index[0], 'centers']
#     head = start.at[start.index[0], 'head']
#
#     if head != -1 and head in df.index.tolist():
#         p2 = df.at[head, 'centers']
#         # print("P1 {}, P2 {}".format(p1, p2))
#         n = p2 - p1
#         n = n/np.linalg.norm(n)
#         return n
#     else:
#         return None


if __name__ == '__main__':
    # For chains, only the second file is needed.
    pairs = [('~/cheap.pcd', 'superPoints/pointsDataFrameB.pkl'),
             ('~/sites/tetraTech/BoilerRoom/chunk_cheap.pcd', 'superPoints/chunk_cheap.pkl'),
             ('~/sites/tetraTech/BoilerRoom/full_5mm.pcd', 'superPoints/full_5mm.pkl'),
             ('~/sites/tetraTech/BoilerRoom/chunk_cheap.pcd', 'superPoints/chunk_cheapB.pkl'),
             ('', 'superPoints/synthA.pkl')]
    pair = pairs[-1]

    superPoints = Samples.Samples()
    superPoints.load(pair[1])

    C = Chains(superPoints)

    view = True
    if view:
        MV2 = ModelView.ModelView()
        MV2.addPoints(C.centers, C.radii)
        MV2.update()

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

