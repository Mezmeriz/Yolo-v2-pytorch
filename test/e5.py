import pye57
import sys
sys.path.append('/home/sadams/green/greenUtilities')

import numpy as np

e57 = pye57.E57("/home/sadams/sites/MezSnake/MezSnake.e57")

# the ScanHeader object wraps most of the scan information:
header = e57.get_header()

# all the header information can be printed using:
for line in header.pretty_print():
    print(line)

data = e57.read_scan(0, colors = True, ignore_missing_fields = True)
print(data.keys())
x = data['cartesianX']
y = data['cartesianY']
z = data['cartesianZ']

red = data['colorRed']
green = data['colorGreen']
blue = data['colorBlue']

points = np.hstack((
	np.expand_dims(x, axis = 1),
	np.expand_dims(y, axis = 1),
	np.expand_dims(z, axis = 1), ))

colors = np.hstack((
	np.expand_dims(red, axis = 1),
	np.expand_dims(green, axis = 1),
	np.expand_dims(blue, axis = 1), ))

print(points.shape)
