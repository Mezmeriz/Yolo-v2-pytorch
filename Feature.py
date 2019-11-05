import numpy as np
from typing import List


class FeatureLine:

    def __init__(self, o: np.ndarray, v: np.ndarray, r: float, indices: List[int], start: np.ndarray, end: np.ndarray):
        self.origin = o
        self.vector = v
        self.radius = r
        self.indices = indices
        self.start_pt = start
        self.end_pt = end


class FeatureArc:
    def __init__(self, c: np.ndarray, c_uv: np.ndarray, n: np.ndarray, u: np.ndarray, v: np.ndarray,
                 arc_r: float, tube_r: float, indices: List[int], uvs: List[np.ndarray], start: float, end: float):
        self.center = c
        self.center_uv = c_uv
        self.normal = n
        self.u_vec = u
        self.v_vec = v
        self.arc_radius = arc_r
        self.tube_radius = tube_r
        self.indices = indices
        self.uvs = np.array(uvs)
        self.start_theta = start
        self.end_theta = end
