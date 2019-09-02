import pandas as pd
import numpy as np
import ViewSamples
import DiceSimple

anno = DiceSimple.Samples()
step = 0.1
x = 200
y = 200
bx = 1.5*step*448*2
by = bx
objectness = 1
classNumber = 0

N = 5
xx = np.linspace(0, (N-1)*step, N)
yy = np.zeros_like(xx)
zz = yy
xyz = np.vstack([xx, yy, zz]).T
vectors = np.identity(3)
predictions = [[[x, y, bx, by, objectness, classNumber]]]

# Add a couple orphans
# anno.add(predictions, np.array([0, 0.8, 0.4]), vectors)
# anno.add(predictions, np.array([0, 0.4, 0.8]), vectors)

for ifor in range(N):
    anno.add(predictions, xyz[ifor,:], vectors)
    # Parallel set to be merged in
    anno.add(predictions, xyz[ifor, :] + np.array([0, 0, 0.4 + 0.045]), vectors)

    # Parallel extension
for ifor in range(N):
    anno.add(predictions, xyz[ifor, :] + np.array([N * step, 0, 0.005]), vectors)

for ifor in range(N):
    anno.add(predictions, -xyz[ifor,:] - np.array([0, 0, 0.005]), vectors)

anno.add(predictions, np.array([0, 0.4, 0.4]), vectors)
anno.save('../superPoints/synthA.pkl')
