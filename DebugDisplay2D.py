import math

from Feature import FeatureLine, FeatureArc

# from math import sin, cos
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import rcParams

from typing import List


def plot_feature_line(line: FeatureLine, centers: List[np.ndarray]) -> None:
    # create plot, specify limits to show all data
    figure = plt.figure()
    ax = figure.add_subplot(111, projection='3d')
    half_width = max(np.fabs(line.end_pt - line.start_pt)) / 2
    mean = np.array([line.start_pt, line.end_pt]).mean(axis=0)
    ax.set_xlim(mean[0] - half_width, mean[0] + half_width)
    ax.set_ylim(mean[1] - half_width, mean[1] + half_width)
    ax.set_zlim(mean[2] - half_width, mean[2] + half_width)

    # plot the points
    points = []
    for i in line.indices:
        points.append(centers[i])
    pts = np.array(points)
    ax.scatter(*pts.T)

    # plot the line
    endpoints = np.array([line.start_pt, line.end_pt])
    ax.plot(*endpoints.T)

    plt.show()


def plot_points_and_line(midpoint: np.ndarray, direction: np.ndarray, centers: List[np.ndarray]) -> None:
    # create plot, specify limits to show all data
    figure = plt.figure()
    ax = figure.add_subplot(111, projection='3d')
    pts = np.array(centers)
    span = np.max(pts, axis=0) - np.min(pts, axis=0)
    # b = np.max(points, axis=0)
    half_width = max(span) / 2
    ax.set_xlim(midpoint[0] - half_width, midpoint[0] + half_width)
    ax.set_ylim(midpoint[1] - half_width, midpoint[1] + half_width)
    ax.set_zlim(midpoint[2] - half_width, midpoint[2] + half_width)

    # plot the points
    # points = []
    # for i in line.indices:
    #     points.append(centers[i])
    # pts = np.array(points)
    ax.scatter(*pts.T)

    # plot the line
    endpoints = np.array([midpoint - span * direction, midpoint + span * direction])
    ax.plot(*endpoints.T)

    plt.show()


def plot_feature_plane(points: np.ndarray, midpoint: np.ndarray, normal: np.ndarray) -> None:
    # create plot, specify limits to show all data
    figure = plt.figure()
    ax = figure.add_subplot(111, projection='3d')
    spans = np.max(points, axis=0) - np.min(points, axis=0)
    half_width = max(spans) / 2
    mean = points.mean(axis=0)
    ax.set_xlim(mean[0] - half_width, mean[0] + half_width)
    ax.set_ylim(mean[1] - half_width, mean[1] + half_width)
    ax.set_zlim(mean[2] - half_width, mean[2] + half_width)

    # plot the points
    ax.scatter(*points.T)

    # plot the plane normal
    endpoints = np.array([midpoint - normal / 2, midpoint + normal / 2])
    ax.plot(*endpoints.T)

    '''
    # plot a representation of a plane
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    X, Y = np.meshgrid(np.arange(xlim[0], xlim[1]), np.arange(ylim[0], ylim[1]))
    Z = np.zeros(X.shape)
    for r in range(X.shape[0]):
        for c in range(X.shape[1]):
            Z[r, c] = 1 * X[r, c] + 1 * Y[r, c] + 1
    ax.plot_wireframe(X, Y, Z, color='k')
    '''

    plt.show()


def plot_feature_arc(arc: FeatureArc) -> None:
    # create plot, specify limits to show all data
    rcParams['figure.figsize'] = 8, 8
    figure = plt.figure()
    ax = figure.add_subplot(111)

    # To fill display with points...
    mins = np.min(arc.uvs, axis=0)
    maxs = np.max(arc.uvs, axis=0)
    spans = maxs - mins
    half_width = 1.2 * max(spans) / 2
    mean = mins + spans / 2
    ax.set_xlim(mean[0] - half_width, mean[0] + half_width)
    ax.set_ylim(mean[1] - half_width, mean[1] + half_width)

    # To fill display with circle...
    # radius = arc.radius * 1.2
    # ax.set_xlim(arc.center_uv[0] - radius, arc.center_uv[0] + radius)
    # ax.set_ylim(arc.center_uv[1] - radius, arc.center_uv[1] + radius)

    thetas = []
    for uv in arc.uvs:
        pt = uv - arc.center_uv
        thetas.append(math.atan2(pt[1], pt[0]))

    # plot the points
    ax.scatter(*arc.uvs.T, c=thetas, cmap=plt.get_cmap("rainbow"))

    # plot the circle
    t = np.linspace(0, 2 * np.pi, 100)
    xs = np.cos(t) * arc.arc_radius + arc.center_uv[0]
    ys = np.sin(t) * arc.arc_radius + arc.center_uv[1]
    ax.plot(xs, ys)

    plt.show()


def plot_points_and_arc(center: np.ndarray, radius: float, uvs: np.ndarray, title: str) -> None:
    # create plot, specify limits to show all data
    rcParams['figure.figsize'] = 8, 8
    figure = plt.figure()
    ax = figure.add_subplot(111)
    ax.set_title(title)

    # To fill display with points...
    mins = np.min(uvs, axis=0)
    maxs = np.max(uvs, axis=0)
    spans = maxs - mins
    half_width = 1.2 * max(spans) / 2
    mean = mins + spans / 2
    ax.set_xlim(mean[0] - half_width, mean[0] + half_width)
    ax.set_ylim(mean[1] - half_width, mean[1] + half_width)

    # To fill display with circle...
    # radius = arc.radius * 1.2
    # ax.set_xlim(arc.center_uv[0] - radius, arc.center_uv[0] + radius)
    # ax.set_ylim(arc.center_uv[1] - radius, arc.center_uv[1] + radius)

    thetas = []
    for uv in uvs:
        pt = uv - center
        thetas.append(math.atan2(pt[1], pt[0]))

    # plot the points
    ax.scatter(*uvs.T, c=thetas, cmap=plt.get_cmap("rainbow"))

    # plot the circle
    t = np.linspace(0, 2 * np.pi, 100)
    xs = np.cos(t) * radius + center[0]
    ys = np.sin(t) * radius + center[1]
    ax.plot(xs, ys)

    plt.show()
