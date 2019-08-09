import pandas as pd
import ViewRip
import DiceSimple

# class DoubleLink():
#
#     def __init__(self, ):

if __name__ == '__main__':
    pairs = [('~/cheap.pcd', 'superPoints/pointsDataFrameB.pkl'),
             ('~/sites/tetraTech/BoilerRoom/chunk_cheap.pcd', 'superPoints/chunk_cheap.pkl'),
             ('~/sites/tetraTech/BoilerRoom/full_5mm.pcd', 'superPoints/full_5mm.pkl'),
             ('~/sites/tetraTech/BoilerRoom/chunk_cheap.pcd', 'superPoints/chunk_cheapB.pkl'),
             ('', '../superPoints/synthA.pkl')]
    pair = pairs[-1]

    superPoints = DiceSimple.Samples()
    superPoints.load(pair[1])
    # ViewRip.ViewRip("", superPoints)
    print(superPoints.df.head())

    df = superPoints.df
    