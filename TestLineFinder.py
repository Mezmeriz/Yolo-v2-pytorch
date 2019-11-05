import Samples
from FeatureFinder import FeatureFinder
import DebugDisplay2D
from DebugDisplay3D import FeatureRenderer3D

# =================================================================================================

# Run the algorithm on the test data set, displaying the results
file = 'superPoints/chunk_cheapB.pkl'
superPoints = Samples.Samples()
superPoints.load(file)
featureFinder = FeatureFinder(superPoints)
print("Using ", len(featureFinder.centers), " points")

do_one = False
if do_one:
    indices = [61]
    lines = featureFinder.find_lines(indices)
    featureFinder.debug_output = True
else:
    lines = featureFinder.find_lines()

print("Found ", len(lines), " lines")

renderer = FeatureRenderer3D()
renderer.render_feature_points(featureFinder.centers, featureFinder.radii, featureFinder.test_status,
                               featureFinder.debug_status, True, True, True, False)
# renderer.render_debug_tubes(featureFinder)
renderer.render_lines(lines)
renderer.update()

if do_one:
    DebugDisplay2D.plot_feature_line(lines[0], featureFinder.centers)
