import Samples
from DebugDisplay3D import FeatureRenderer3D

import numpy as np
import scipy.spatial.kdtree as kd

from typing import List, Tuple

# =================================================================================================

STEP = 0.06
CLOSE_LOC = STEP / 2
CLOSE_SIZE_MIN = 0.75
CLOSE_SIZE_MAX = 1.333

NO_CLUSTER = -1


def find_clusters(super_pts: Samples.Samples) -> Tuple[List[np.ndarray], List[float], List[int]]:

    # For testing just part of the data space
    # Full range:
    # box_min = [8.2, 0.3, -1.2]
    # box_max = [11.7, 6.9, 1.3]
    # Good quadrant, trimmed
    # box_min = [10.2, 0.7, -0.7]
    # box_max = [11.7, 3.6, 1.2]
    # middle area with connected lines/arcs
    box_min = [10.7, 2.0, 0.0]
    box_max = [11.4, 2.8, 0.8]
    # small area, should have about 6 lines, 4 arcs.  Some arcs missing, others wrong.
    # box_min = [10.2, 0.7, -0.7]
    # box_max = [11.7, 1.5, 1.2]

    # For now, skip all the large radius points to focus on easier to handle small ones
    skip_big = True

    # Create list of points and radii, efficiency structure for searching points
    points = []
    radii = []
    clusters = []  # In which cluster is this point?
    for superPoint in super_pts:
        p, r = superPoint
        if all(box_min <= p) and all(p <= box_max) and (skip_big and r < 0.2):
            points.append(p)
            radii.append(r)
            clusters.append(NO_CLUSTER)
    p_tree = kd.KDTree(points)

    # Assign each point to a cluster (points that have similar locations and radii)
    cluster_index = 0
    for i, p in enumerate(points):
        found = False
        close = p_tree.query_ball_point(p, CLOSE_LOC)
        if len(close) > 1:  # Don't count this point
            rmin = radii[i] * CLOSE_SIZE_MIN
            rmax = radii[i] * CLOSE_SIZE_MAX
            for j in close:
                if j != i and clusters[j] != NO_CLUSTER and rmin <= radii[j] <= rmax:
                    clusters[i] = clusters[j]
                    found = True
                    break
        if not found:
            clusters[i] = cluster_index
            cluster_index += 1

    # Calculate the average location and radius, plus count, for each cluster
    ret_pts = []
    ret_rs = []
    ret_counts = []
    clusters = np.array(clusters)
    for i in range(cluster_index):
        indices = np.where(clusters == i)[0]
        p = points[indices[0]]
        r = radii[indices[0]]
        for j in range(1, len(indices)):
            p += points[indices[j]]
            r += radii[indices[j]]
        ret_counts.append(len(indices))
        ret_pts.append(p/len(indices))
        ret_rs.append(r/len(indices))

    return ret_pts, ret_rs, ret_counts

# =================================================================================================


if __name__ == '__main__':
    # Run the algorithm on the test data set, displaying the results
    file = 'superPoints/chunk_cheapB.pkl'
    superPoints = Samples.Samples()
    superPoints.load(file)

    pts, rs, counts = find_clusters(superPoints)
    test = []
    for index in range(len(counts)):
        counts[index] -= 1
        counts[index] = int(index/13)
        start_at = 78
        stop_before = start_at + 13
        group_size = 2
        counts[index] = int((index-start_at) / group_size) if start_at <= index < stop_before else 7
    print("found ", len(pts), " clusters")

    renderer = FeatureRenderer3D()
    renderer.render_points(pts, rs, counts, True)
    # renderer.render_points(pts[1000:1250], rs[1000:1250], counts[1000:1250], True)
    renderer.update()
