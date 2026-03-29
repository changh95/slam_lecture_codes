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
Radtan8 model implemented with opencv calls
"""
from math import sqrt

import cv2
import numpy as np

from calibration import FX, FY, CX, CY, K1, K2, K3, K4, K5, K6, P1, P2

array = lambda arr: np.array(arr, dtype=np.float64)
cameraMatrix = array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
distCoeffs = array([K1, K2, P1, P2, K3, K4, K5, K6])


def project(x, y, z):
    point = array([[[x, y, z]]])
    uv, _ = cv2.projectPoints(point, np.zeros(3), np.zeros(3), cameraMatrix, distCoeffs)
    return uv.flatten()


def project_jacobian_xyz(x, y, z):
    point = array([[[x, y, z]]])
    _uv, jac = cv2.projectPoints(
        point, np.zeros(3), np.zeros(3), cameraMatrix, distCoeffs
    )
    return jac[:, 3:6].flatten()


def project_jacobian_params(x, y, z):
    point = array([[[x, y, z]]])
    _uv, jac = cv2.projectPoints(
        point, np.zeros(3), np.zeros(3), cameraMatrix, distCoeffs
    )
    return jac[:, 6:].flatten()


def unproject(u, v, N):
    uv = array([[[u, v]]])
    # criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TermCriteria_EPS, N, eps)
    criteria = (cv2.TERM_CRITERIA_COUNT, N, 0)
    p3d = cv2.undistortPointsIter(uv, cameraMatrix, distCoeffs, None, None, criteria)
    mx, my = p3d.flatten()
    norm = sqrt(1 + mx**2 + my**2)

    return mx / norm, my / norm, 1 / norm
