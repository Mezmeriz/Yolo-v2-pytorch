import pandas as pd
import ViewRip
import DiceSimple
import numpy as np

# class DoubleLink():
#
#     def __init__(self, ):
def addCenters(superPoints):
    N = len(superPoints)
    centers = []
    radii = []
    for i in range(N):
        center, radius = superPoints[i]
        centers.append(center)
        radii.append(radius)
    superPoints.df['centers'] = centers
    superPoints.df['radii'] = radii
    superPoints.df['head'] = np.zeros((N, 1), dtype = np.int) + -1
    superPoints.df['tail'] = np.zeros((N, 1), dtype = np.int) + -1
    superPoints.df['chain'] = np.zeros((N, 1), dtype = np.int) + -1
    superPoints.df['count'] = np.zeros((N, 1), dtype=np.int) + -1
    return np.vstack(centers)

def follow(direction, superPoints, index, chain):
    while superPoints.df.at[index, (direction)] != -1:
        index = superPoints.df.at[index, (direction)]
        if superPoints.df.at[index, ('chain')] == -1:
            superPoints.df.at[index, 'chain'] = chain
        else:
            break

def enumerateChain(superPoints):
    chain = 0
    for index in range(len(superPoints.df)):
        if superPoints.df.at[index, ('chain')] == -1:
            superPoints.df.at[index, 'chain'] = chain
            follow('head', superPoints, index, chain)
            follow('tail', superPoints, index, chain)
            chain = chain + 1

def nearestNeighbors(superPoints, centers):
    """Find the closest point in each direction.
    Later, weight the "closest" based on radius and angle too
    """
    epsilon = 1e-4
    distanceCriteria = 2.2 * 0.06
    for index in range(len(superPoints.df)):
        currentRadius = superPoints.df.at[index, 'radii']
        localCenters = centers - centers[index, :]
        localCenters = localCenters.dot(superPoints.df.iloc[index]['vectors'].reshape((3,3)))
        IIforward = np.where(localCenters[:, 0] > epsilon)[0]
        IIbackward = np.where(localCenters[:, 0] < -epsilon)[0]
        distances = np.linalg.norm(localCenters, axis = 1)
        if len(IIforward):
            IIforwardMin = np.argmin(distances[IIforward])
            minRadius = superPoints.df.at[IIforward[IIforwardMin],'radii']
            # distanceCriteria = (minRadius + currentRadius)
            if IIforwardMin is not None and distances[IIforward][IIforwardMin] < distanceCriteria:
                superPoints.df.at[index, 'head'] = IIforward[IIforwardMin]
            else:
                print("Failed: Current radius = {:4.1f} cm, distance = {:4.1f} cm".format(currentRadius*1e2, distances[IIforward][IIforwardMin]* 1e2))
        if len(IIbackward):
            IIbackwardMin = np.argmin(distances[IIbackward])
            # distanceCriteria = (minRadius + currentRadius)

            if IIbackwardMin is not None and distances[IIbackward][IIbackwardMin] < distanceCriteria:
                superPoints.df.at[index, 'tail'] = IIbackward[IIbackwardMin]
            else:
                print("Failed: Current radius = {:4.1f} cm, distance = {:4.1f} cm".format(currentRadius * 1e2,
                                                                                distances[IIbackward][IIbackwardMin]* 1e2))
    print(id(superPoints.df))

def chainCount(superPoints):
    for chain in superPoints.df['chain'].unique():
        n = len(superPoints.df[superPoints.df['chain']==chain])
        superPoints.df.loc[superPoints.df['chain']==chain,'count']=n

def orphanFilter(superPoints, N=1):
    print("Len berfore orphan filter {}".format(len(superPoints.df)))
    centers = addCenters(superPoints)
    nearestNeighbors(superPoints, centers)
    enumerateChain(superPoints)
    chainCount(superPoints)
    superPoints.df = superPoints.df[superPoints.df['count']>N]
    print("Len after orphan filter {}".format(len(superPoints.df)))
    return superPoints

def makeStraight(superPoints):
    """
    Go through each chain.
    Start one link in if possible
    go forward until
    :param superPoints:
    :return:
    """
    NChains = superPoints.df['chain'].max()

if __name__ == '__main__':
    pairs = [('~/cheap.pcd', 'superPoints/pointsDataFrameB.pkl'),
             ('~/sites/tetraTech/BoilerRoom/chunk_cheap.pcd', 'superPoints/chunk_cheap.pkl'),
             ('~/sites/tetraTech/BoilerRoom/full_5mm.pcd', 'superPoints/full_5mm.pkl'),
             ('~/sites/tetraTech/BoilerRoom/chunk_cheap.pcd', 'superPoints/chunk_cheapB.pkl'),
             ('', '../superPoints/synthA.pkl')]
    pair = pairs[-1]

    superPoints = DiceSimple.Samples()
    superPoints.load(pair[1])


    centers = addCenters(superPoints)
    nearestNeighbors(superPoints, centers)
    enumerateChain(superPoints)
    chainCount(superPoints)
    superPoints = orphanFilter(superPoints, N=1)
    print(superPoints.df.head(20))

    #ViewRip.ViewRip("", superPoints)
    print(superPoints.df.keys())