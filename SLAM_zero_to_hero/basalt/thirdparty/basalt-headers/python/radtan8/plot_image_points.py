#
# BSD 3-Clause License
#
# This file is part of the Basalt project.
# https://gitlab.com/VladyslavUsenko/basalt.git
#
# Copyright (c) 2022, Collabora Ltd.
# All rights reserved.
#
# Author: Mateo de Mayo <mateo.demayo@collabora.com>
#

"""
Code for ploting things in image coordinates.
"""

import numpy as np
from numpy import pi, sin, cos
import matplotlib.pyplot as plt
from basalt_radtan import project


def plot_circle_at_z1(radius_m, color=None, label=None):
    if radius_m <= 0:
        return
    alphas = np.linspace(0, 2 * pi, 100)
    circle_3d = np.array([cos(alphas), sin(alphas)]) * radius_m
    us, vs = project(*circle_3d, 1)
    plt.plot(us, vs, "-", color=color, label=label)
    plt.legend()


def plot(good_points, failed_points, invalid_points, radiuses=(), names=()):
    # img = imread(f"{os.path.dirname(__file__)}/res/example-frame.png")
    # plt.imshow(img, cmap="gray")

    limits = np.array([(0, 0), (640, 0), (640, 480), (0, 480), (0, 0)])
    lxs, lys = limits.T.copy()
    plt.plot(lxs, lys, "-r")

    if len(good_points) != 0:
        good_points = np.array(good_points)
        gxs, gys = good_points.T.copy()
        plt.plot(gxs, gys, ".b", markersize=1)
    else:
        print("No good_points found")

    if len(failed_points) != 0:
        failed_points = np.array(failed_points)
        bxs, bys = failed_points.T.copy()
        plt.plot(bxs, bys, ".r", markersize=10)
    else:
        print("No failed_points found")

    if len(invalid_points) != 0:
        invalid_points = np.array(invalid_points)
        ixs, iys = invalid_points.T.copy()
        plt.plot(ixs, iys, ".m", markersize=1)
    else:
        print("No invalid_points found")

    for radius, name in zip(radiuses, names):
        plot_circle_at_z1(radius, label=name)

    plt.show()
