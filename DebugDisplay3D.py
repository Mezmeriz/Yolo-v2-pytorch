from Feature import FeatureLine, FeatureArc
from Status import TestStatus, DebugStatus
import ModelView

import numpy as np

from typing import List


# =================================================================================================

class FeatureRenderer3D:

    COLOR_RED = [1, 0, 0]
    COLOR_GREEN = [0.1, 0.9, 0.1]
    COLOR_BLUE = [0, 0, 1]
    COLOR_CYAN = [0, 1, 1]
    COLOR_MAGENTA = [1, 0, 1]
    COLOR_YELLOW = [1, 1, 0]
    COLOR_ORANGE = [1, 0.5, 0]
    COLOR_PINK = [1, 0.5, 0.6]

    color_ramp = [COLOR_YELLOW, COLOR_GREEN, COLOR_CYAN, COLOR_BLUE, COLOR_MAGENTA, COLOR_PINK, COLOR_RED, COLOR_ORANGE]

    tube_colors = [COLOR_CYAN, COLOR_ORANGE, COLOR_YELLOW]

    def __init__(self):
        self.viewer = ModelView.ModelView()

    def render_points(self, points: List[np.array], radii: List[float], colors: List[int], small: bool) -> None:

        for index, point in enumerate(points):
            radius = 0.04 if small else radii[index]
            self.viewer.addPoints([point], [radius], self.color_ramp[colors[index]])

    # Conditionally display the discovered lines and subsets of the superpoints based on their status.
    def render_feature_points(self, points: List[np.array], radii: List[float],
                              test_status: List[TestStatus], debug_status: List[DebugStatus],
                              show_used: bool, show_unused: bool, show_debug: bool, small: bool) -> None:

        colors = []
        for index, point in enumerate(points):
            radius = 0.04 if small else radii[index]
            # For points on the feature, only debug-override interior points that are the start
            if show_used:
                if test_status[index] == TestStatus.FEATURE_INTERIOR:
                    if show_debug and debug_status[index] == DebugStatus.START:
                        self.viewer.addPoints([point], [radius], self.COLOR_RED)
                    else:
                        self.viewer.addPoints([point], [radius], self.COLOR_BLUE)
                elif test_status[index] == TestStatus.FEATURE_END:
                    self.viewer.addPoints([point], [radius], self.COLOR_MAGENTA)
            # For points not on the feature, override for close and near points.
            if show_unused:
                if show_debug:
                    if debug_status[index] == DebugStatus.CLOSE:
                        self.viewer.addPoints([point], [radius], self.COLOR_CYAN)
                    elif debug_status[index] == DebugStatus.NEAR:
                        self.viewer.addPoints([point], [radius], self.COLOR_PINK)
                    else:
                        self.viewer.addPoints([point], [radius], self.COLOR_GREEN)
                else:
                    if test_status[index] == TestStatus.UNTESTED:
                        self.viewer.addPoints([point], [radius], self.COLOR_GREEN)
                    elif test_status[index] == TestStatus.NOT_USED:
                        self.viewer.addPoints([point], [radius], self.COLOR_YELLOW)

    def render_debug_tubes(self, points: List[np.ndarray], dirs: List[np.ndarray], colors: List[int]):
        for i in range(len(points)):
            self.viewer.addTube(points[i] - dirs[i],
                                points[i] + dirs[i],
                                0.01,
                                self.tube_colors[colors[i]])

    def render_lines(self, line_list: List[FeatureLine]):
        for l in line_list:
            self.viewer.addTube(l.start_pt, l.end_pt, l.radius, self.COLOR_ORANGE)

    def render_arcs(self, arc_list: List[FeatureArc]):
        for a in arc_list:
            self.viewer.addTorus(a.center, a.normal, a.arc_radius, a.tube_radius, self.COLOR_ORANGE)

    def update(self):
        self.viewer.update()
