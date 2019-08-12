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
step = 0.06
N = 5
xx = np.linspace(0,N*step, N)
yy = np.zeros_like(xx)
zz = yy
xyz = np.vstack([xx, yy, zz]).T
vectors = np.identity(3)

for ifor in range(N):
    predictions = [[[x, y, bx, by, objectness, classNumber]]]
    anno.add(predictions, xyz[ifor,:], vectors)
    anno.add(predictions, xyz[ifor,:] + np.array([0,0,0.2]), vectors)

anno.add(predictions, np.array([0,0.4, 0.4]), vectors)
anno.save('../superPoints/synthA.pkl')
