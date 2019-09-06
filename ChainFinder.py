import Samples
import ModelView
import numpy as np
from numpy.linalg import norm
import math
from typing import List, Tuple

COLOR_BLUE = [0, 0, 1]
COLOR_GREEN = [0.1, 0.9, 0.1]
COLOR_RED = [1, 0, 0]
COLOR_YELLOW = [1, 1, 0]
COLOR_CYAN = [0, 1, 1]
COLOR_MAGENTA = [1, 0, 1]
COLOR_ORANGE = [1, 0.5, 0]
COLOR_PINK = [1, 0.5, 0.6]

UNUSED = 0
START = 1
CLOSE = 2
NEAR = 3
IN_LINE = 4
LINE_END = 5
ISOLATED = 6

MIN_RADIUS_SCALE = 0.8
MAX_RADIUS_SCALE = 1.25
FARTHEST_CLOSE_POINT = 0.3
TARGET_STD_DEV = 0.025
MAX_DIST_FROM_LINE = 0.03
MAX_GAP = 0.125  # twice the apparent gap between superpoints


# Given a test point and a line defined by a point and a direction vector,l
# calculate the distance between the point and the line and the point's parametric
# position (W) along the line with the line point serving as the origin.
def distances_from_and_along_line(test_point: np.ndarray, line_point: np.ndarray, line_dir: np.ndarray) \
        -> Tuple[float, float]:
    line_point_to_test_point = test_point - line_point
    orthogonal_to_plane = np.cross(line_point_to_test_point, line_dir)
    # If test point is on the line, this yields a zero length vector
    if float(norm(orthogonal_to_plane)) == 0:
        distance = 0
    else:
        line_to_test_point = np.cross(line_dir, orthogonal_to_plane)
        line_to_test_point = line_to_test_point / norm(line_to_test_point)
        distance = float(np.dot(line_point_to_test_point, line_to_test_point))
    w = float(np.dot(line_point_to_test_point, line_dir))

    return distance, w


# Given a list of indices for 3D points, calculates the midpoint of the collection and uses
# singular value decomposition to calculate the vector through that midpoint that
# minimizes the error for the distances from the points to the line.
def midpoint_and_mean_vector(indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    pts = []
    for i in indices:
        pts.append(centers[i])
    pts_array = np.array(pts)
    midpoint = pts_array.mean(axis=0)
    tmp0, tmp1, vv = np.linalg.svd(pts_array - midpoint)
    line_dir = vv[0]
    return midpoint, line_dir


# Returns true if the provided list of point indices meets the criteria for finding a line
# within the set.  Those criteria are a sufficient number of points and large enough dimensions
# spanned by the points.
def validate_points(indices: List[int]) -> bool:
    if len(indices) <= 3:
        # error = "Too few points (" + str(len(indices)) + ") found for index " + str(test_index) + "!"
        return False  # , error
    pts = []
    for i in indices:
        pts.append(centers[i])
    spans = np.amax(pts, axis=0) - np.amin(pts, axis=0)
    if spans[0] < test_radius and spans[1] < test_radius and spans[2] < test_radius:
        # error = "Points found for index " + str(test_index) + " are too close together!"
        return False  # , error
    return True  # , ""


# Given a seed point and its radius, find indices of all points with similar radii
# that are spatially close to the seed point.  If there are less than three points,
# we will not be able to find a line (or arc) starting from this seed point.
def find_closest_similar_points(seed_pt: np.ndarray, radius_in: float) -> List[int]:
    r_min = radius_in * MIN_RADIUS_SCALE
    r_max = radius_in * MAX_RADIUS_SCALE

    indices: List[int] = []
    dists: List[float] = []
    for i, pt in enumerate(centers):
        if status[i] == UNUSED and r_min <= radii[i] <= r_max:
            vec = np.subtract(seed_pt, pt)
            dist = math.sqrt(np.dot(vec, vec))
            if dist <= FARTHEST_CLOSE_POINT:
                indices.append(i)
                dists.append(dist)
                # if status[i] != START:
                #     status[i] = CLOSE

    # print("Points close to point: ", len(indices))
    return indices


# Given a set of indices for points within which we are trying to find a line, produce a
# reasonable first estimate of that line.  The line passes through the point that
# is the mean of all points in the set and has a direction calculated using singular
# value decomposition.
def find_line_for_points(indices: List[int], radius: float)\
        -> Tuple[List[int], List[float], np.ndarray, np.ndarray]:
    midpoint = test_pt  # Works better than midpoint found below
    last_line_dir = None
    while True:
        tmp, line_dir = midpoint_and_mean_vector(indices)
        # Prevent line direction from getting reversed once we've started
        if last_line_dir is not None:
            if float(np.dot(line_dir, last_line_dir)) < 0:
                line_dir *= -1

        # Find distances from this line to each point in the set
        dists = []
        ws = []
        for i in indices:
            pt = centers[i]
            d, w = distances_from_and_along_line(pt, midpoint, line_dir)
            dists.append(d)
            ws.append(w)

        # Remove points from the set that are too far from the line, decide if we should stop
        mean_dist = np.mean(dists)
        stddev = np.std(dists)
        limit = mean_dist + stddev  # Adjust this to control deletions
        # print("  ", len(indices), line_dir, radius, mean_dist, stddev, limit)
        none_removed = True
        if mean_dist < radius * 0.1:  # Adjust this to control stopping condition
            break
        else:
            # Remove points starting from the end so indices stay the same
            for i in range(len(indices)-1, -1, -1):
                if dists[i] > limit:
                    del indices[i]
                    del ws[i]
                    none_removed = False
        last_line_dir = line_dir
        # If we didn't remove any or hit the exit criteria, bail to avoid infinite loop
        if none_removed:
            indices.clear()
        if len(indices) <= 3:
            return indices, [], np.array([0, 0, 0]), np.array([1, 0, 0])

    # Sort the points based on their parametric distance along the line
    ws, indices = (list(t) for t in zip(*sorted(zip(ws, indices))))

    # Searching out from the start, remove any that are separated by too large of a gap
    # d, w = distances_from_and_along_line(test_pt, midpoint, line_dir)
    w = 0  # If using test point, w is zero.  If using calculated midpoint, use calculated w.
    start = len(ws)-1
    for i in range(len(ws)):
        if ws[i] > w:
            start = i-1
            break
    # Delete those from the end first so start index stays the same
    last_w = ws[start]
    for i in range(start + 1, len(ws)):
        if ws[i] - last_w > MAX_GAP:
            for j in range(len(ws) - i):
                del ws[i]
                del indices[i]
            break
        last_w = ws[i]
    # Now delete those at the beginning
    last_w = ws[start]
    for i in range(start-1, -1, -1):
        if last_w - ws[i] > MAX_GAP:
            for j in range(i):
                del ws[0]
                del indices[0]
            break
        last_w = ws[i]

    # print("Points close to line: ", len(indices))
    # print("  Line from points: ", midpoint, line_dir)
    return indices, ws, midpoint, line_dir


# Determines how many of the points that lie along the line are contiguous with those that
# were initially found.  It walks from the identified previous endpoint towards the end
# of the newly found points and stops when the next point is either too far from the line
# or too far from the previous point.  Updates the line point/direction after each addition.
def walk_the_line(ws: List[float], indices: List[int],
                  midpoint: np.ndarray, line_dir: np.ndarray, start_index: int, step: int)\
                  -> Tuple[np.ndarray, np.ndarray, int]:
    last_w = ws[start_index]
    last_line_dir = line_dir
    stop = len(ws) if step > 0 else -1
    for i in range(start_index + step, stop, step):
        # Quit if this point is too far along the line from last point
        if math.fabs(ws[i] - last_w) > MAX_GAP:
            break

        # Quit if this point is too far from the updated line
        pt = centers[indices[i]]
        d, w = distances_from_and_along_line(pt, midpoint, line_dir)
        # TODO - Should this instead be half of the radius?  Or is it related to the step size?
        if d > MAX_DIST_FROM_LINE:
            break

        # Include the point in the solution set
        last_w = ws[i]
        start_index = i
        if status[indices[i]] != START:
            status[indices[i]] = IN_LINE

        # Update the midpoint and line dir
        midpoint, line_dir = midpoint_and_mean_vector(indices)
        # Prevent line direction from getting reversed once we've started
        if float(np.dot(line_dir, last_line_dir)) < 0:
            line_dir *= -1
        # print("  ", midpoint, line_dir)

    if status[indices[start_index]] != START:
        status[indices[start_index]] = LINE_END
    return midpoint, line_dir, start_index


# Finds additional points that lie along the initial estimate for the line and adds them
# to the point collection.  The line point and direction are updated as points are added
# to arrive at a better final result.  The line's radius is calculated as the average of
# the radii of all points found to be on the line.
def extend_line_by_points(indices_in: List[int], ws: List[float], radius: float,
                          midpoint: np.ndarray, line_dir: np.ndarray)\
                          -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # note first and last point currently, then clear Ws
    start_w = ws[0]
    end_w = ws[-1]
    ws = []

    # Assuming the initial line is close to what we want, find all points close to it.
    # This is only done once, so any points missed here will never be found.
    r_min = radius * MIN_RADIUS_SCALE
    r_max = radius * MAX_RADIUS_SCALE
    indices = []
    for i, pt in enumerate(centers):
        if (status[i] == UNUSED and r_min <= radii[i] <= r_max) or i in indices_in:
            d, w = distances_from_and_along_line(pt, midpoint, line_dir)
            # TODO - Should this nearness test be scaled based on distance from the midpoint?
            if d < MAX_DIST_FROM_LINE:
                ws.append(w)
                indices.append(i)

    # Sort the results by parametric distance, mark previously known points
    ws, indices = (list(t) for t in zip(*sorted(zip(ws, indices))))
    start_index = 0
    for i in range(len(indices)):
        if ws[i] > start_w:
            start_index = i
            break
    end_index = len(ws)-1
    for i in range(start_index+1, len(indices)):
        if ws[i] > end_w:
            end_index = i-1
            break

    for i in range(len(indices)):
        if status[indices[i]] != START:
            if start_index <= i <= end_index:
                status[indices[i]] = IN_LINE
            # else:
            #     status[indices[i]] = NEAR

    # Walk along the line from the previous endpoints, adding suitable points
    midpoint, line_dir, start_index = walk_the_line(ws, indices, midpoint, line_dir, start_index, -1)
    midpoint, line_dir, end_index = walk_the_line(ws, indices, midpoint, line_dir, end_index, 1)

    # Calculate an average radius for the all the points in the line
    avg_radius = 0
    for i in range(start_index, end_index+1, 1):
        avg_radius += radii[indices[i]]
    avg_radius /= (end_index - start_index + 1)

    # print("Points after extending: ", len(indices))
    # print("  Line after extending: ", midpoint, line_dir)
    return avg_radius, midpoint, line_dir, centers[indices[start_index]], centers[indices[end_index]]


'''
# The following methods are included in this submission for archival purposes.
# Other approaches work better, but these implementations might be useful someday.
# Note that they may have bugs!  Maybe that's why they didn't work as well as others...

# Find a tentative vector for the line that may be represented by a majority of the points
# in the provided set.  This involves the following steps:
#   - Calculate vectors between the starting point and each other point in the set
#   - Determine which axis is the largest in a majority of these vectors
#   - Flip vectors with negative values in the majority axis so all vectors point the same way
#   - Calculate the average vector
#   - Iteratively cull vectors until the remaining ones are very similar
#     - This is done by finding the average of the vectors and dotting each vector against the average
#     - Find the standard deviation of the dots and remove vectors whose dots are too far off
def find_common_line_by_vecs(pts: List[np.ndarray]) -> np.ndarray:
    # Calculate vectors from start point to each other point.  Identify common largest dimension.
    vecs: List[np.ndarray] = [np.array([0, 0, 0])] * (len(pts)-1)
    max_counts = [0, 0, 0]
    # skip first point, which is origin of this search
    for i in range(1, len(pts)):
        p = pts[i]
        vec = np.subtract(p, pts[i-1])
        vecs[i-1] = vec / norm(vec)
        x = math.fabs(vec[0])
        y = math.fabs(vec[1])
        z = math.fabs(vec[2])
        if x > y:
            if x > z:
                max_counts[0] += 1
            else:
                max_counts[2] += 1
        elif y > z:
            max_counts[1] += 1
        else:
            max_counts[2] += 1

    # Flip vectors that have negative values in the common largest dimension.
    max_index = max_counts.index(max(max_counts))
    vec_sum = np.array([0, 0, 0])
    for i, vec in enumerate(vecs):
        if vec[max_index] < 0:
            vecs[i] = vec * -1

    cull = True
    line = [0, 0, 0]
    while cull:
        # Find tentative line
        for vec in vecs:
            vec_sum = vec_sum + vec
        line = vec_sum / norm(vec_sum)

        # Find angles to tentative line.  Eliminate points greater than a std dev away.
        dots = []
        for vec in vecs:
            dots.append(np.dot(vec, line))
        stddev = np.std(dots)
        to_remove = []
        if stddev > TARGET_STD_DEV:
            for i in range(len(vecs)):
                if 1.0 - dots[i] > stddev:
                    to_remove.append(i)
            # remove unwanted vectors from back to front so indices remain valid
            to_remove.reverse()
            for i in to_remove:
                del vecs[i]
                del pts[i]
        cull = len(to_remove)

    print(line)
    return pts, line


# Given an estimated line and a set of points that are sorted based on their location
# along that line, produce a refined estimate for the line's equation.  Clusters of
# points that are co-located along the line are averaged and then vectors are found
# between each such averaged point.  Vectors are compared to the provided line and
# those that differ by too much are removed.  The refined line is returned.
def find_common_line_by_vectors(pts: List[np.ndarray], line_in: np.ndarray)\
        -> Tuple[bool, np.ndarray, np.ndarray]:
    # Merge clusters of points along the line
    joined_pts = []
    last_pt = pts[0]
    pts_in_group = 1
    pts_sum = [0, 0, 0] + last_pt
    i = 1
    while i < len(pts):
        vec = pts[i] - last_pt
        dist = math.sqrt(np.dot(vec, vec))
        if dist < 0.02:
            # Another point for the current cluster
            pts_sum += pts[i]
            pts_in_group += 1
        else:
            # The cluster is finished, start another one
            joined_pts.append(pts_sum / pts_in_group)
            last_pt = pts[i]
            pts_in_group = 1
            pts_sum = [0, 0, 0] + pts[i]
        i += 1

    # Calculate vectors between clustered points and keep those that match the given line.
    # TODO - Should we only discard those at the start and end (to avoid curves)?
    sum_vec = [0, 0, 0]
    first = None
    last = None
    have_some = False
    for i in range(1, len(joined_pts)):
        vec = joined_pts[i] - joined_pts[i-1]
        vec = vec / norm(vec)
        dot = np.dot(vec, line_in)
        if dot >= 0.9:
            sum_vec += vec
            if first is None:
                first = joined_pts[i-1]
            last = joined_pts[i]
            have_some = True
    if have_some:
        line = sum_vec / norm(sum_vec)
        midpoint = first + (last - first) / 2
    else:
        line = None
        midpoint = None

    print(line)
    return have_some, line, midpoint


def find_points_near_line(origin: np.ndarray, radius: float, line: np.ndarray)\
        -> Tuple[List[float], List[int]]:
    r_min = radius * MIN_RADIUS_SCALE
    r_max = radius * MAX_RADIUS_SCALE
    ws_out: List[float] = []
    indices_out: List[int] = []
    for i, pt in enumerate(centers):
        if r_min <= radii[i] <= r_max:
            vec = pt - origin
            perp1 = np.cross(vec, line)
            perp2 = np.cross(line, perp1)
            dist = np.dot(vec, perp2)
            if dist < MAX_DIST_FROM_LINE:
                ws_out.append(float(np.dot(vec, line)))
                indices_out.append(i)

    ws_out, indices_out = (list(t) for t in zip(*sorted(zip(ws_out, indices_out))))

    return ws_out, indices_out


def process_string_of_points(start_index: int, dists: List[float], indices: List[int]) -> float:
    # mid_index = indices.index(start_index)
    mid_index = int(len(indices) / 2)
    # Walk up the list from the start until a gap is too large or the end is reached
    last_dist = 0
    for i in range(mid_index+1, len(indices)):
        if dists[i] - last_dist > MAX_GAP:
            for j in range(len(indices)-i):
                del indices[i]
            break
        last_dist = dists[i]

    # Walk down the list from the start until a gap is too large or the start is reached
    # mid_index = indices.index(start_index)
    mid_index = int(len(indices) / 2)
    last_dist = 0
    for i in range(mid_index-1, -1, -1):
        if dists[i] - last_dist < -MAX_GAP:
            for j in range(i+1):
                del indices[0]
            break
        last_dist = dists[i]

    # Find average radius of remaining points
    radii_sum = 0
    for i in indices:
        radii_sum += radii[i]
    radius = radii_sum / len(indices)

    # Make endpoints have the average radius
    radii[indices[0]] = radius
    radii[indices[-1]] = radius

    # Mark status of first, start and last as endpoints, others as "in line"
    status[indices[0]] = LINE_END
    status[indices[-1]] = LINE_END
    for i in range(1, len(indices)-1):
        status[indices[i]] = START if i == mid_index else IN_LINE

    # Snap endpoints to line
    # TODO - Recalculate line?
    # centers[indices[0]] = LINE_END
    # centers[indices[-1]] = LINE_END

    return radius
'''


# Conditionally display the discovered lines and subsets of the superpoints based on their status.
def render_results(show_tubes: bool, show_ends: bool, show_unused: bool, show_diag: bool) -> None:
    # Collect all classified points and their radii
    unused = []
    unused_radii = []
    isolated = []
    isolated_radii = []
    starts = []
    start_radii = []
    inlines = []
    inline_radii = []
    endpoints = []
    endpoint_radii = []
    close = []
    close_radii = []
    near = []
    near_radii = []
    for index, center in enumerate(centers):
        if status[index] == UNUSED:
            unused.append(center)
            unused_radii.append(radii[index])
        if status[index] == ISOLATED:
            isolated.append(center)
            isolated_radii.append(radii[index])
        elif status[index] == CLOSE:
            close.append(center)
            close_radii.append(radii[index])
        elif status[index] == NEAR:
            near.append(center)
            near_radii.append(radii[index])
        elif status[index] == IN_LINE:
            inlines.append(center)
            inline_radii.append(radii[index])
        elif status[index] == START:
            starts.append(center)
            start_radii.append(radii[index])
        elif status[index] == LINE_END:
            endpoints.append(center)
            endpoint_radii.append(radii[index])

    # Draw the things that are enabled
    viewer = ModelView.ModelView()
    if show_unused:
        viewer.addPoints(unused, unused_radii, COLOR_GREEN)
        viewer.addPoints(isolated, isolated_radii, COLOR_GREEN)  # COLOR_PINK)
    if show_diag:
        viewer.addPoints(close, close_radii, COLOR_YELLOW)
        viewer.addPoints(near, near_radii, COLOR_CYAN)
        viewer.addPoints(inlines, inline_radii, COLOR_BLUE)
        if len(starts):
            viewer.addPoints(starts, start_radii, COLOR_RED)
        # TODO - Need to capture diagnostic data during loop over points, then revise these lines
        # viewer.addTube(good_midpoint - good_line, good_midpoint + good_line, 0.01, COLOR_CYAN)
        # viewer.addTube(better_midpoint - better_line, better_midpoint + better_line, 0.01, COLOR_ORANGE)
    if show_ends:
        viewer.addPoints(endpoints, endpoint_radii, COLOR_MAGENTA)
    if show_tubes:
        for i, start_pt in enumerate(start_points):
            viewer.addTube(start_pt, end_points[i], line_radii[i], COLOR_ORANGE)
    viewer.update()


# =================================================================================================

# Read test data into two arrays
file = 'superPoints/chunk_cheapB.pkl'
superPoints = Samples.Samples()
superPoints.load(file)

centers = []
radii = []
status = []
for superPoint in superPoints:
    c, r = superPoint
    centers.append(c)
    radii.append(r)
    status.append(UNUSED)

# We will collect the lines we find
start_points = []
end_points = []
line_radii = []

# Indices of interesting starting points:
#  100 short, wiggly.  good!
#  700 med,   good! stop one sooner at curves?
# 1000 long,  good! stop one sooner at curves?
# 1100 med,   good! stop one sooner at curves?
# 1600 med,   good! case with numerous parallel close points
# 2600 med,   good! another set of nearby parallel lines
# 2700 long,  varying thickness.  good test, what is right solution?  see 2698, 2690
# 2800 long,  includes tight dip.  good, but lines are slightly offset
# 2900 med,   good! through a T junction?
# others: 50, 750, 950 (fat!), 1050, 1350, 1650, 1950, 2050, 2150, 2250, 2550, 2650, 2850

#  900 curve, point is on curve, skipped
# 1400 curve, point is on curve, skipped
# 1500 long,  start point too large, skipped
# 1700 short, start point at end, skip - found with 1702
# 1900 med,   skipped

num_lines = 0
for test_index in range(len(superPoints)):
    # print(test_index)
    if status[test_index] != UNUSED:
        continue

    test_pt, test_radius = superPoints[test_index]
    status[test_index] = START

    close_indices = find_closest_similar_points(test_pt, test_radius)
    if not validate_points(close_indices):
        status[test_index] = ISOLATED
        continue

    sorted_indices, sorted_ws, good_midpoint, good_line = find_line_for_points(close_indices, test_radius)
    if not validate_points(sorted_indices):
        status[test_index] = ISOLATED
        continue

    line_radius, better_midpoint, better_line, start_point, end_point =\
        extend_line_by_points(sorted_indices, sorted_ws, test_radius, good_midpoint, good_line)
    start_points.append(start_point)
    end_points.append(end_point)
    line_radii.append(line_radius)
    num_lines += 1

print("Found ", num_lines, " lines")
render_results(True, True, False, False)
