import Samples
from FeatureFinder import FeatureFinder
from DebugDisplay3D import FeatureRenderer3D

# =================================================================================================

# Run the algorithm on the test data set, displaying the results
file = 'superPoints/chunk_cheapB.pkl'
superPoints = Samples.Samples()
superPoints.load(file)
featureFinder = FeatureFinder(superPoints)
print("Using ", len(featureFinder.centers), " points")

do_selected = False
if do_selected:
    indices = [38]
    featureFinder.debug_output = True
    arcs = featureFinder.find_arcs(indices)
else:
    # featureFinder.debug_output = True
    arcs = featureFinder.find_arcs()

print("\nFound ", len(arcs), " arcs")

renderer = FeatureRenderer3D()
renderer.render_feature_points(featureFinder.centers, featureFinder.radii, featureFinder.test_status,
                               featureFinder.debug_status, True, True, True, False)
renderer.render_arcs(arcs)
renderer.update()
