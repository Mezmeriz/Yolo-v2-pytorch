import pandas as pd
import numpy as np

d = pd.DataFrame({'A': 1, 'B': [1,2,3,4], 'C':pd.Categorical(['fi','fie','foe', 'fum'])})
d2 = pd.DataFrame({'A': 1, 'B': [1,2,3,4], 'C':pd.Categorical(['fi','fie','foe', 'fum'])})
d = d.append(d2, ignore_index=True)

d = pd.DataFrame()
for ifor in range(1000):

    di = pd.DataFrame({'image': 'foo{}.jpg'.format(ifor), 'bbox': np.random.random(())}, index = [0])
    if ifor == 0:
        d = di
    else:
        if ifor == 4:
            for j in range(10):
                d = d.append(di, ignore_index=True)

        d = d.append(di, ignore_index=True)

import time
start = time.time()
print(d[d['image']=='foo4.jpg'])
end = time.time()
print("Elapsed time = {}".format(end-start))