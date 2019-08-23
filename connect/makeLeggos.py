import pandas as pd
import numpy as np
import ViewSamples
import DiceSimple

anno = DiceSimple.Samples()

x = 200
y = 200
bx = 90
by = 90
objectness = 1
classNumber = 0
step = 0.15
N = 5
xx = np.linspace(0,N*step, N)
yy = np.zeros_like(xx)
zz = yy
xyz = np.vstack([xx, yy, zz]).T
vectors = np.identity(3)
predictions = [[[x, y, bx, by, objectness, classNumber]]]

anno.add(predictions, np.array([0,0.8, 0.4]), vectors)
anno.add(predictions, np.array([0,0.4, 0.8]), vectors)

for ifor in range(N):

    anno.add(predictions, xyz[ifor,:], vectors)

    anno.add(predictions, xyz[ifor,:] + np.array([0,0,0.045]), vectors)

anno.add(predictions, np.array([0,0.4, 0.4]), vectors)
anno.save('../superPoints/synthA.pkl')
