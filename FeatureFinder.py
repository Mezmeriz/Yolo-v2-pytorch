import sys

from Status import TestStatus, DebugStatus
import Samples
from Feature import FeatureLine, FeatureArc
from Clustering import find_clusters
import DebugDisplay2D

import numpy as np
from numpy.linalg import norm
import scipy.spatial.kdtree as kd
import math

from typing import List, Tuple, Union


# =================================================================================================

class FeatureFinder:

    # TODO - These need better names, descriptions and organization.
    # These should come from the data or user options
    STEP_DISTANCE = 0.06            # in cms.  should be dependent on the data set

    # For line finding methods
    MIN_RADIUS_SCALE = 0.6
    MAX_RADIUS_SCALE = 1.4
    FARTHEST_CLOSE_POINT = 0.25     # should be relative to the start point radius
    RADIUS_SCALE_FOR_NEAR = 1.0  # 1.5
    RADIUS_SCALE_FOR_LINE = 0.5
    RADIUS_SCALE_FOR_PLANE = 0.5
    TOO_FAR_FACTOR = 0.5            # gets multiplied by standard deviation
    CLOSE_ENOUGH_FACTOR = 0.25      # gets multiplied by radius
    TARGET_STD_DEV = 0.025
    MAX_GAP = 2 * STEP_DISTANCE
    MIN_POINTS_FOR_LINE = 3
    MAX_MEAN_LINE_ERROR = 0.0075

    # For arc finding methods
    MAX_ARC_GAP = 1.25 * STEP_DISTANCE

    def __init__(self, points: Samples.Samples):
        self.centers = []
        self.radii = []
        self.test_status = []   # Which points have been tested and how they failed
        self.debug_status = []  # The role a point played in the most recent test
        self.considered = []    # The points that were considered during a test

        self.line_pts = []
        self.line_dirs = []
        self.line_colors = []
        self.debug_output = False

        # Get a clustered set of points
        pts, rs, counts = find_clusters(points)
        self.centers = pts
        self.radii = rs
        self.kdtree = kd.KDTree(self.centers)

        for _ in self.centers:
            self.test_status.append(TestStatus.UNTESTED)
            self.debug_status.append(DebugStatus.NONE)

    def num_points(self) -> int:
        return len(self.centers)

    def get_test_status(self, i: int) -> TestStatus:
        return self.test_status[i]

    def get_debug_status(self, i: int) -> DebugStatus:
        return self.debug_status[i]

    def get_considered_indices(self) -> List[int]:
        return self.considered

    def get_point(self, i: int) -> np.ndarray:
        return self.centers[i]

    def get_radius(self, i: int) -> float:
        return self.radii[i]

    # Given a test point and a line defined by a point and a direction vector,
    # calculate the distance between the point and the line and the point's parametric
    # position (W) along the line with the line point serving as the origin.
    @staticmethod
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
    def midpoint_and_mean_vector(self, indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        pts = []
        for i in indices:
            pts.append(self.centers[i])
        pts_array = np.array(pts)
        midpoint = pts_array.mean(axis=0)
        tmp0, tmp1, vv = np.linalg.svd(pts_array - midpoint)
        line_dir = vv[0]
        return midpoint, line_dir

    # Returns true if the provided list of point indices meets the criteria for finding a line
    # within the set.  Those criteria are a sufficient number of points and large enough dimensions
    # spanned by the points.
    def validate_points(self, indices: List[int], test_radius: float) -> bool:
        if len(indices) < self.MIN_POINTS_FOR_LINE:
            # error = "Too few points (" + str(len(indices)) + ") found for index " + str(test_index) + "!"
            return False  # , error
        pts = []
        for i in indices:
            pts.append(self.centers[i])
        spans = np.amax(pts, axis=0) - np.amin(pts, axis=0)
        # TODO - Don't know what a good value is here.  Test was added to avoid very short lines,
        #   but current test causes failures on lines with large radius due to small sample distance.
        #   Might need to factor in angled lines (no dimension is full size) or step size
        #   (not likely to get full sample distance because of distance between points).
        # if spans[0] < test_radius and spans[1] < test_radius and spans[2] < test_radius:
        dist = test_radius * 0.9
        if spans[0] < dist and spans[1] < dist and spans[2] < dist:
            # error = "Points found for index " + str(test_index) + " are too close together!"
            return False  # , error
        return True  # , ""

    # Given a seed point and its radius, find indices of all points with similar radii
    # that are spatially close to the seed point.  If there are less than three points,
    # we will not be able to find a line (or arc) starting from this seed point.
    def find_closest_similar_points(self, seed_index: int, max_dist: float = FARTHEST_CLOSE_POINT)\
            -> List[int]:
        r_min = self.radii[seed_index] * self.MIN_RADIUS_SCALE
        r_max = self.radii[seed_index] * self.MAX_RADIUS_SCALE

        close_pts = self.kdtree.query_ball_point(self.centers[seed_index], max_dist)
        indices: List[int] = []
        for i in close_pts:
            # We allow points at the ends of already-found features to be included in adjacent features
            if (self.test_status[i] == TestStatus.UNTESTED or
                # self.test_status[i] == TestStatus.FEATURE_END or
                self.test_status[i] == TestStatus.NOT_USED) and \
                    r_min <= self.radii[i] <= r_max:
                indices.append(i)
                if self.debug_status[i] != DebugStatus.START:
                    self.debug_status[i] = DebugStatus.CLOSE
                    self.considered.append(i)

        if self.debug_output:
            print("Points close to point: ", len(indices))
        return indices

    def calculate_plane_for_points(self, indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        pts = []
        for i in indices:
            pts.append(self.centers[i])
        pts = np.array(pts)
        points = pts.T
        center = points.mean(axis=1)
        x = points - center[:, np.newaxis]
        mat = np.dot(x, x.T)
        normal = np.linalg.svd(mat)[0][:, -1]
        return center, normal

    def points_form_a_line(self, indices: List[int]) -> bool:
        # Calculate a line that fits the points
        midpoint, line_dir = self.midpoint_and_mean_vector(indices)

        # Find the distances of the points from the line
        line_errors = []
        for i in indices:
            d, w = self.distances_from_and_along_line(self.centers[i], midpoint, line_dir)
            line_errors.append(d)

        # If all points are close to the line, return true
        mean = np.mean(line_errors)
        return mean < self.MAX_MEAN_LINE_ERROR

    # =================================================================================================
    # Find Lines

    def add_debug_line(self, pt: np.ndarray, line_dir: np.ndarray, color: int) -> None:
        self.line_pts.append(pt)
        self.line_dirs.append(line_dir)
        self.line_colors.append(color)

    # Find all distances from this line and along this line for the provided set of indices.
    def find_all_distances_from_and_along_line(self, indices: List[int], midpoint: np.ndarray, line_dir: np.ndarray)\
            -> Tuple[List[float], List[float]]:
        dists = []
        ws = []
        for i in indices:
            pt = self.centers[i]
            d, w = self.distances_from_and_along_line(pt, midpoint, line_dir)
            dists.append(d)
            ws.append(w)
        return dists, ws

    # Given a set of indices for points within which we are trying to find a line, produce a
    # reasonable first estimate of that line.  The line passes through the point that
    # is the mean of all points in the set and has a direction calculated using singular
    # value decomposition.
    def find_line_for_points(self, indices: List[int], test_index: int) \
            -> Tuple[List[int], List[float], np.ndarray, np.ndarray]:
        midpoint = self.centers[test_index]  # Works better for first pass than using midpoint found in loop
        last_line_dir = None
        indices_out = indices.copy()
        while True:
            tmp, line_dir = self.midpoint_and_mean_vector(indices_out)
            # Prevent line direction from getting reversed once we've started
            if last_line_dir is not None:
                if float(np.dot(line_dir, last_line_dir)) < 0:
                    line_dir *= -1

            # Remove points from the set that are too far from the line, decide if we should stop
            dists, ws = self.find_all_distances_from_and_along_line(indices_out, midpoint, line_dir)
            mean_dist = np.mean(dists)
            stddev = np.std(dists)
            limit = mean_dist + stddev * self.TOO_FAR_FACTOR  # Adjust this to control deletions
            if self.debug_output:
                print("  ", len(indices_out), line_dir, self.radii[test_index], mean_dist, stddev, limit)
            none_removed = True
            # if mean_dist < self.radii[test_index] * self.CLOSE_ENOUGH_FACTOR:  # stopping condition
            if mean_dist < self.MAX_MEAN_LINE_ERROR:  # stopping condition
                break
            else:
                # Remove points starting from the end so indices stay the same
                for i in range(len(indices_out)-1, -1, -1):
                    if dists[i] > limit:  # point removal condition
                        del indices_out[i]
                        del ws[i]
                        none_removed = False
            last_line_dir = line_dir
            # If we didn't remove any or hit the exit criteria, bail to avoid infinite loop
            if none_removed:
                break
            if len(indices_out) < self.MIN_POINTS_FOR_LINE:
                return indices_out, [], np.array([0, 0, 0]), np.array([1, 0, 0])

        # Sort the points based on their parametric distance along the line
        ws, indices_out = (list(t) for t in zip(*sorted(zip(ws, indices_out))))

        # Remove points that are too far (based on w) from the adjacent point.
        # Find the test point (its w will be 0) and use it as the starting location.
        w = 0
        start = len(ws)-1
        for i in range(len(ws)):
            if ws[i] > w:
                start = i-1
                break

        # Delete those above the start point first so the start index stays the same
        last_w = ws[start]
        for i in range(start + 1, len(ws)):
            if ws[i] - last_w > self.MAX_GAP:
                for j in range(len(ws) - i):
                    del ws[i]
                    del indices_out[i]
                break
            last_w = ws[i]

        # Now delete those before the start point
        last_w = ws[start]
        for i in range(start-1, -1, -1):
            if last_w - ws[i] > self.MAX_GAP:
                for j in range(i):
                    del ws[0]
                    del indices_out[0]
                break
            last_w = ws[i]

        # Test to see if removing points from the ends improves the fit.
        #   Should only remove points if there are enough left (4?) and the mean is lowered.  What about stddev?
        #   TODO - This part should probably be in its own routine.
        #     Also have to make sure next method doesn't just add the points back.
        # Recalculate line equation and point distances after latest point removals
        midpoint0, line0 = self.midpoint_and_mean_vector(indices)
        dists, ws = self.find_all_distances_from_and_along_line(indices_out, midpoint0, line0)
        mean0 = np.mean(dists)
        if len(ws) <= 3 and mean0 > self.MAX_MEAN_LINE_ERROR:  # not a good enough line
            return indices_out, [], np.array([0, 0, 0]), np.array([1, 0, 0])
        # print("X", mean0)

        # Iterate, removing the point at one end or the other if it improves things
        stop = False
        while not stop and len(indices_out) > 4:
            midpoint1, line1 = self.midpoint_and_mean_vector(indices_out[0:len(indices_out)-1])
            dists, ws = self.find_all_distances_from_and_along_line(
                indices_out[0:len(indices_out)-1], midpoint1, line1)
            mean1 = np.mean(dists)
            # print(1, mean1)

            midpoint2, line2 = self.midpoint_and_mean_vector(indices_out[1:len(indices_out)])
            dists, ws = self.find_all_distances_from_and_along_line(
                indices_out[1:len(indices_out)], midpoint2, line2)
            mean2 = np.mean(dists)
            # print(2, mean2)

            # TODO - These tests should probably require the improvement to be significant.
            if mean1 < mean2 and mean1 < mean0:
                indices_out = indices_out[0:len(indices_out)-1]
                midpoint0 = midpoint1
                line0 = line1
                mean0 = mean1
                # print(1)
            elif mean2 < mean1 and mean2 < mean0:
                indices_out = indices_out[1:len(indices_out)]
                midpoint0 = midpoint2
                line0 = line2
                mean0 = mean2
                # print(2)
            else:
                stop = True

        if self.debug_output:
            print("Points close to line: ", len(indices_out))
            print("  Line from points: ", midpoint0, line0)
        return indices_out, ws, midpoint0, line0

    # Determines how many of the points that lie along the line are contiguous with those that
    # were initially found.  It walks from the identified previous endpoint towards the end
    # of the newly found points and stops when the next point is either too far from the line
    # or too far from the previous point.
    def walk_the_line(self, ws: List[float], indices: List[int],
                      midpoint: np.ndarray, line_dir: np.ndarray, start_index: int, step: int) -> int:
        last_w = ws[start_index]
        stop = len(ws) if step > 0 else -1
        tmp_indices = indices[start_index:start_index+1].copy()
        for i in range(start_index + step, stop, step):
            # Quit if this point is too far along the line from last point
            if math.fabs(ws[i] - last_w) > self.MAX_GAP:
                break

            # Quit if this point is too far from the updated line
            pt = self.centers[indices[i]]
            d, w = self.distances_from_and_along_line(pt, midpoint, line_dir)
            # Avoid adding points that are too far from the line, which might be entering an arc.
            if d > self.STEP_DISTANCE * 0.25:
                break

            # Include the point in the solution set
            last_w = ws[i]
            start_index = i
            if indices[i] not in self.considered:
                self.considered.append(indices[i])

            # Update the temporary line definition to include the new point
            # TODO - I'm now wondering if updating the line is a good idea.  The new points might
            #   be within the tolerance, but old (already accepted) points might move outside of it.
            tmp_indices.append(indices[i])
            midpoint, line_dir = self.midpoint_and_mean_vector(tmp_indices)

        return start_index

    # Finds additional points that lie along the initial estimate for the line and adds them
    # to the point collection.  Updates the midpoint and line direction after the points
    # have been found.  The line's radius is calculated as the average of the radii of all
    # points found to be on the line.
    def extend_line_by_points(self, indices_in: List[int], ws: List[float], radius: float,
                              midpoint: np.ndarray, line_dir: np.ndarray)\
            -> Tuple[float, np.ndarray, np.ndarray, List[int], float, float]:
        # Note first and last point currently, then clear Ws
        start_w = ws[0]
        end_w = ws[-1]
        ws = []

        # Assuming the initial line is close to what we want, find all points close to it.
        # This is only done once, so any points missed here will never be found.
        r_min = radius * self.MIN_RADIUS_SCALE
        r_max = radius * self.MAX_RADIUS_SCALE
        indices = []
        for i, pt in enumerate(self.centers):
            if (self.test_status[i] == TestStatus.UNTESTED and r_min <= self.radii[i] <= r_max) or i in indices_in:
                d, w = self.distances_from_and_along_line(pt, midpoint, line_dir)
                if d < self.RADIUS_SCALE_FOR_NEAR * radius:
                    ws.append(w)
                    indices.append(i)

        if len(ws) == 0:
            return 0, np.ndarray([0, 0, 0]), np.ndarray([1, 0, 0]), [], 0, 0

        # Sort the results by parametric distance, mark previously known points
        ws, indices = (list(t) for t in zip(*sorted(zip(ws, indices))))
        start_index = 0
        for i in range(len(indices)):
            if ws[i] >= start_w:
                start_index = i
                break
        end_index = len(ws)-1
        for i in range(start_index+1, len(indices)):
            if ws[i] > end_w:
                end_index = i-1
                break

        for i in range(len(indices)):
            if self.debug_status[indices[i]] != DebugStatus.START:
                if i < start_index or i > end_index:
                    self.debug_status[indices[i]] = DebugStatus.NEAR
                    self.considered.append(i)

        # Walk along the line from the previous endpoints, adding suitable points
        start_index = self.walk_the_line(ws, indices, midpoint, line_dir, start_index, -1)
        end_index = self.walk_the_line(ws, indices, midpoint, line_dir, end_index, 1)

        # Remove points outside of accepted range
        # Delete those from the end first so start index stays the same
        for i in range(len(indices) - end_index - 1):
            del ws[end_index+1]
            del indices[end_index+1]
        # Now delete those at the beginning
        for i in range(start_index):
            del ws[0]
            del indices[0]

        # Update the midpoint and line dir
        new_midpoint, new_line_dir = self.midpoint_and_mean_vector(indices)
        # Prevent line direction from getting reversed once we've started
        if float(np.dot(new_line_dir, line_dir)) < 0:
            new_line_dir *= -1

        # Update endpoint ws after midpoint and line possibly changed
        tmp, w_start = self.distances_from_and_along_line(self.centers[indices[0]], new_midpoint, new_line_dir)
        tmp, w_end = self.distances_from_and_along_line(self.centers[indices[-1]], new_midpoint, new_line_dir)

        # Calculate an average radius for the all the points in the line
        avg_radius = 0
        for i in range(len(indices)):
            avg_radius += self.radii[indices[i]]
        avg_radius /= len(indices)

        if self.debug_output:
            print("Points after extending: ", len(indices))
            print("  Line after extending: ", new_midpoint, new_line_dir)
        return avg_radius, new_midpoint, new_line_dir, indices, w_start, w_end

    def find_line(self, test_index: int, close_indices: List[int]) -> Union[FeatureLine, None]:
        self.debug_status[test_index] = DebugStatus.START
        self.considered.append(test_index)

        sorted_indices, sorted_ws, good_midpoint, good_line = \
            self.find_line_for_points(close_indices, test_index)
        if len(sorted_ws) < self.MIN_POINTS_FOR_LINE:
            self.test_status[test_index] = TestStatus.NOT_USED
            return None
        if self.debug_output:
            self.add_debug_line(good_midpoint, good_line, 0)

        line_radius, better_midpoint, better_line, final_indices, w_start, w_end = \
            self.extend_line_by_points(sorted_indices, sorted_ws, self.radii[test_index], good_midpoint, good_line)
        if not self.validate_points(final_indices, line_radius):
            self.test_status[test_index] = TestStatus.NOT_USED
            return None
        if self.debug_output:
            self.add_debug_line(better_midpoint, better_line, 1)

        line = FeatureLine(better_midpoint, better_line, line_radius, final_indices,
                           better_midpoint + (better_line * w_start), better_midpoint + (better_line * w_end))
        for i, index in enumerate(final_indices):
            if i == 0 or i == 1 or i == len(final_indices)-2 or i == len(final_indices)-1:
                self.test_status[index] = TestStatus.FEATURE_END
            else:
                self.test_status[index] = TestStatus.FEATURE_INTERIOR

        return line

    # =================================================================================================
    # Find Arcs

    def find_plane(self, test_index: int) -> Tuple[bool, List[int], np.ndarray, np.ndarray]:
        # Find all points that are within a few steps of the test point.  Quit if there are none.
        search_radius = self.STEP_DISTANCE * 2.5
        close_indices = self.find_closest_similar_points(test_index, search_radius)
        if len(close_indices) == 0:
            return False, [], np.array([0, 0]), np.array([0, 0, 0])

        if self.points_form_a_line(close_indices):
            return False, close_indices, np.array([0, 0]), np.array([0, 0, 0])

        # This code was tuned on only one data set, which didn't have a full variety of arcs/radii.
        # Hence, this "table" is currently pretty sparse.  It contains distance for two additional
        # point gathering/testing operations that refine the plane and the point collection.
        # In the test data, most points have a radius nar 0.06, but one big pipe has radius 0.27.
        # What are typical pipe bend radii for such pipe cross-section radii?  That determines how
        # far away we should look to find points that might help define the full pipe arc.
        # TODO - Need to define some constants for these values
        test_radius = self.radii[test_index]
        if test_radius < 0.1:
            dist_step = 0.2
        else:  # mainly for radius of 0.27
            dist_step = 0.6
        # TODO - Instead of going wider now, should we find more points along the arc in find_2d_arc?
        # TODO - It now seems like going only one step wider here works better than doing more.
        distances = [dist_step]  # , 2 * dist_step]
        # Iterate over larger search areas, updating the plane equation each time
        for distance in distances:
            center, normal = self.calculate_plane_for_points(close_indices)
            # Get the points again, then discard those that are too far from the plane
            close_indices = self.find_closest_similar_points(test_index, distance)
            kept_indices = []
            for i in close_indices:
                d, w = self.distances_from_and_along_line(self.centers[i], center, normal)
                if math.fabs(w) < test_radius * self.RADIUS_SCALE_FOR_PLANE:
                    kept_indices.append(i)
            close_indices = kept_indices
        if test_index not in close_indices:
            return False, [], np.array([0, 0]), np.array([0, 0, 0])

        for i in close_indices:
            if self.debug_status[i] != DebugStatus.START:
                self.debug_status[i] = DebugStatus.NEAR

        center, normal = self.calculate_plane_for_points(close_indices)

        return True, close_indices, center, normal

    def project_points_to_2d(self, indices, midpoint, plane_eq) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Calculate 2D coordinate system (u, v) in plane
        x = math.fabs(plane_eq[0])
        y = math.fabs(plane_eq[1])
        z = math.fabs(plane_eq[2])
        if x < y and x < z:
            away_from_normal = [1, 0, 0]
        elif y < x and y < z:
            away_from_normal = [0, 1, 0]
        else:
            away_from_normal = [0, 0, 1]
        v_vec = np.cross(plane_eq, away_from_normal)
        u_vec = np.cross(v_vec, plane_eq)

        # Project given points into 2D space (origin at midpoint)
        uvs = []
        for i in indices:
            to_pt = self.centers[i] - midpoint
            uvs.append(np.array([np.dot(to_pt, u_vec), np.dot(to_pt, v_vec)]))

        return u_vec, v_vec, np.array(uvs)

    @staticmethod
    def find_circle(us: np.ndarray, vs: np.ndarray) -> Tuple[np.ndarray, float]:
        a = np.array([us, vs, np.ones(len(us))]).T
        b = us ** 2 + vs ** 2
        c = np.linalg.lstsq(a, b, rcond=None)[0]
        center = [c[0] / 2, c[1] / 2]
        radius = math.sqrt(c[2] + center[0] ** 2 + center[1] ** 2)
        return np.array(center), radius

    def order_points_around_initial_circle(self, test_i: int, indices: List[int], uvs: np.ndarray) ->\
            Tuple[List[int], List[np.ndarray], List[float], float]:

        # Find the center and radius of our initial circle estimate
        center, radius = self.find_circle(uvs[:, 0], uvs[:, 1])
        if self.debug_output:
            DebugDisplay2D.plot_points_and_arc(np.array(center), radius, uvs, "initial")

        # If the radius is too small or large for the tube radius, quit.
        # TODO - What radius is too small or too large?  Using average radius of all selected points
        avg_rad = 0
        for i in indices:
            avg_rad += self.radii[i]
        avg_rad /= len(indices)

        if radius < avg_rad:  # 1.75 * avg_rad:
            if self.debug_output:
                print("   radius too small")
            return [], [], [], 0

        if radius > 4 * avg_rad:
            if self.debug_output:
                print("   radius too large")
            return [], [], [], 0

        # Sort points going counter-clockwise around the initial circle
        thetas = []
        for i, u in enumerate(uvs[:, 0]):
            pt = np.array([u, uvs[:, 1][i]]) - center
            theta = math.atan2(pt[1], pt[0])
            # If arc of points crosses the PI/-PI barrier, fix that
            if theta < 0:
                theta += 2 * math.pi
            thetas.append(theta)
        # TODO - Can this be done with the uvs staying together as one array?
        thetas, us, vs, new_indices =\
            (list(t) for t in zip(*sorted(zip(thetas, uvs[:, 0], uvs[:, 1], indices))))

        # TODO - There must be a better "numpy" way to rebuild this
        new_uvs = []
        for i in range(len(us)):
            new_uvs.append(np.array([us[i], vs[i]]))

        # Working out from the start point, remove points after a break in the continuity
        start = new_indices.index(test_i)
        remove_these = []

        # Walk down from the start point
        last_pt = new_uvs[start]
        for i in range(start - 1, -1, -1):
            if norm(new_uvs[i] - last_pt) > self.MAX_ARC_GAP:
                remove_these.insert(0, i)
            else:
                last_pt = new_uvs[i]

        # Walk up from the start point
        last_pt = new_uvs[start]
        for i in range(start + 1, len(new_indices)):
            # TODO - This is a bit fragile, because removing one point that should have been kept
            #  almost ensures that the next bunch will be too far away and removed, too.
            if norm(new_uvs[i] - last_pt) > self.MAX_ARC_GAP:
                remove_these.append(i)
            else:
                last_pt = new_uvs[i]

        # Actually do the removing
        for i in reversed(remove_these):
            del new_uvs[i]
            del new_indices[i]
            del thetas[i]

        if self.debug_output:
            DebugDisplay2D.plot_points_and_arc(np.array(center), radius, np.array(new_uvs), "removed")

        # Now that some points have been removed, make sure we aren't left with a line.
        if self.points_form_a_line(new_indices):
            return [], [], [], 0

        if len(new_indices) < 4:
            if self.debug_output:
                print("   not enough points")
            return [], [], [], 0

        return new_indices, new_uvs, thetas, avg_rad

    def find_2d_arc(self, test_i: int, indices: List[int], uvs: List[np.ndarray],
                    thetas: List[float], tube_radius: float) -> \
            Tuple[List[int], List[np.ndarray], np.ndarray, float, float, float]:

        # Create circle estimates from subsets of the points to find the best starting location
        uvs = np.array(uvs)
        best_mean = sys.float_info.max
        bite_size = 4
        best_index = 0
        best_center = np.ndarray([0, 0])
        best_radius = 1

        for i in range(len(uvs) - bite_size + 1):
            us = np.array(uvs[i:i+bite_size, 0])
            vs = np.array(uvs[i:i+bite_size, 1])
            center, radius = self.find_circle(us, vs)

            # Calculate error
            errors = []
            for j, u in enumerate(us):
                vec = np.array([u, vs[j]]) - center
                error = math.fabs(norm(vec) - radius)
                errors.append(error)
            mean_error = np.mean(errors)
            if self.debug_output:
                print(i, mean_error, radius)

            if self.debug_output:
                print("   best group = ", str(best_index))
                DebugDisplay2D.plot_points_and_arc(center, radius, uvs, "test")

            # Keep the one with the lowest error, while radius is within a reasonable range
            # TODO - Is there a valid reason for choosing these radii?  Based on typical arc for radius?
            # TODO - Use a constant or set of them depending on min/max arc radius for classes of tube radii
            # Note - this used to start with "1.75 * avg_rad", but that was removed above and is also needed here
            if tube_radius < radius < 3.2 * tube_radius and mean_error < best_mean:
                best_center = center
                best_radius = radius
                best_mean = mean_error
                best_index = i

        if self.debug_output:
            print("   best group = ", str(best_index))
            DebugDisplay2D.plot_points_and_arc(best_center, best_radius, uvs, "solved")

        # TODO - Expand from best group to include nearby points that help?
        # TODO - Recalculate normal from remaining points?  Do it repeatedly as points are removed/added?
        # TODO - Find start and end thetas from final group of points
        #        Might not be able to do this exactly without adjacent lines
        #        If sweep of arc isn't large enough (>= 30 degrees?), bail on the arc?
        # TODO - If radius of arc is too large for radius of pipe, bail on arc?

        # TODO - This is where we need to continue looking at points beyond the four we liked best.
        #   we may want some points beyond what we've gathered.
        #   If we don't get them now, they will be used to make additional, redundant arcs.
        new_indices = []
        new_uvs = []
        for i in range(best_index, best_index + bite_size):
            new_indices.append(indices[i])
            new_uvs.append(uvs[i])
        start_theta = thetas[best_index]
        end_theta = thetas[best_index + bite_size - 1]

        # If we aren't using the start point, then don't find an arc
        if test_i not in new_indices:
            if self.debug_output:
                print("   test point not in solution")
            return [], new_uvs, best_center, best_radius, 0.0, 1.0

        # If we didn't find any arc fit...
        if len(best_center) == 0:
            if self.debug_output:
                print("   no arc found")
            return [], new_uvs, best_center, best_radius, 0.0, 1.0

        for i, index in enumerate(new_indices):
            # TODO - Need to do this for all points in the terminal groups
            # TODO - Need to record this more permanently than as debug info
            # TODO - try marking first and last TWO points as feature ends, as trying with lines
            if i == 0 or i == 1 or i == len(new_indices)-2 or i == len(new_indices)-1:
                self.test_status[index] = TestStatus.FEATURE_END
            else:
                self.test_status[index] = TestStatus.FEATURE_INTERIOR

        return new_indices, new_uvs, best_center, best_radius, start_theta, end_theta

    def find_arc(self, test_index: int, plane_indices: List[int], midpoint: np.ndarray, normal: np.ndarray)\
            -> Union[FeatureArc, None]:

        u_vec, v_vec, uvs =\
            self.project_points_to_2d(plane_indices, midpoint, normal)
        ordered_indices, ordered_uvs, thetas, radius =\
            self.order_points_around_initial_circle(test_index, plane_indices, uvs)
        if len(ordered_indices) == 0:
            return None

        final_indices, uvs, uv_center, arc_radius, start, end = \
            self.find_2d_arc(test_index, ordered_indices, ordered_uvs, thetas, radius)
        if len(final_indices) == 0:
            self.test_status[test_index] = TestStatus.NOT_USED
            return None

        # Project UV center back to 3D plane
        center = midpoint + u_vec * uv_center[0] + v_vec * uv_center[1]

        # Find average tube radius, mark used points
        tube_radius = 0
        for index in final_indices:
            tube_radius += self.radii[index]
        tube_radius /= len(final_indices)

        if self.debug_output:
            print(test_index)

        arc = FeatureArc(center, uv_center, normal, u_vec, v_vec, arc_radius, tube_radius,
                         final_indices, uvs, start, end)
        return arc

    # =================================================================================================

    def init_point_for_processing(self, test_index: int) -> bool:
        if self.test_status[test_index] != TestStatus.UNTESTED:  # \
                # and self.test_status[test_index] != TestStatus.FEATURE_END:
            return False

        for i in self.considered:
            self.debug_status[i] = DebugStatus.NONE
        self.considered.clear()
        self.debug_status[test_index] = DebugStatus.START

        if self.debug_output:
            print("point: ", test_index)

        return True

    def find_lines(self, indices: List[int] = None) -> List[FeatureLine]:
        if not indices:
            indices = range(len(self.centers))
        found_lines = []

        for test_index in indices:
            if self.init_point_for_processing(test_index):
                close_indices = self.find_closest_similar_points(test_index)
                if self.validate_points(close_indices, self.radii[test_index]):
                    line = self.find_line(test_index, close_indices)
                    if line:
                        found_lines.append(line)
                        print(test_index)
                    else:
                        self.test_status[test_index] = TestStatus.NOT_USED
                else:
                    self.test_status[test_index] = TestStatus.NOT_USED

        return found_lines

    def find_arcs(self, indices: List[int] = None) -> List[FeatureArc]:
        if not indices:
            indices = range(len(self.centers))
        found_arcs = []

        for test_index in indices:
            if not self.init_point_for_processing(test_index):
                continue

            success, on_plane_indices, midpoint, normal = self.find_plane(test_index)
            if success:
                arc = self.find_arc(test_index, on_plane_indices, midpoint, normal)
                if arc:
                    found_arcs.append(arc)
                    print(test_index)
            else:
                self.test_status[test_index] = TestStatus.NOT_USED

        return found_arcs

    def find_arcs_and_lines(self, indices: List[int] = None) -> Tuple[List[FeatureArc], List[FeatureLine]]:
        if not indices:
            indices = list(range(len(self.centers)))
        found_arcs = []
        found_lines = []

        '''
        for test_index in indices:
            if self.init_point_for_processing(test_index):
                close_indices = self.find_closest_similar_points(test_index)
                if self.validate_points(close_indices, self.radii[test_index]):
                    line = self.find_line(test_index, close_indices)
                    if line:
                        found_lines.append(line)
                    else:
                        self.test_status[test_index] = TestStatus.NOT_USED
                else:
                    self.test_status[test_index] = TestStatus.NOT_USED

        for i in indices:
            if self.test_status[i] == TestStatus.NOT_USED:
                self.test_status[i] = TestStatus.UNTESTED
        '''

        for test_index in indices:
            if not self.init_point_for_processing(test_index):
                continue

            success, on_plane_indices, midpoint, normal = self.find_plane(test_index)
            if success:
                arc = self.find_arc(test_index, on_plane_indices, midpoint, normal)
                if arc:
                    found_arcs.append(arc)
                    print(test_index)
            elif len(on_plane_indices) > 0:
                close_indices = self.find_closest_similar_points(test_index)
                if self.validate_points(close_indices, self.radii[test_index]):
                    line = self.find_line(test_index, close_indices)
                    if line:
                        found_lines.append(line)
                    else:
                        self.test_status[test_index] = TestStatus.NOT_USED
                else:
                    self.test_status[test_index] = TestStatus.NOT_USED
            else:
                self.test_status[test_index] = TestStatus.NOT_USED

        return found_arcs, found_lines
