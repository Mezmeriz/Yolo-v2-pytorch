import Samples
from FeatureFinder import FeatureFinder
from DebugDisplay3D import FeatureRenderer3D

# =================================================================================================

# Run the algorithm on the test data set, displaying the results
file = 'superPoints/chunk_cheapB.pkl'
superPoints = Samples.Samples()
superPoints.load(file)
featureFinder = FeatureFinder(superPoints)

do_selected = False
if do_selected:
    indices = [88, 89]
    featureFinder.debug_output = True
    arcs, lines = featureFinder.find_arcs_and_lines(indices)
else:
    # featureFinder.debug_output = True
    arcs, lines = featureFinder.find_arcs_and_lines()

print("\nFound ", len(arcs), " arcs and ", len(lines), " lines")

renderer = FeatureRenderer3D()
renderer.render_feature_points(featureFinder.centers, featureFinder.radii, featureFinder.test_status,
                               featureFinder.debug_status, True, True, True, False)
renderer.render_arcs(arcs)
renderer.render_lines(lines)
renderer.update()
