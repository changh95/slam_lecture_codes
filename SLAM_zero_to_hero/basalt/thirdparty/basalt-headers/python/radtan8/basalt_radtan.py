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
radtan8 project unproject model manually implemented
"""

from math import sqrt
import numpy as np
from numpy.linalg import inv
from calibration import FX, FY, CX, CY, K1, K2, K3, K4, K5, K6, P1, P2


def project(x, y, z):
    xp = x / z
    yp = y / z
    r2 = xp * xp + yp * yp
    cdist = (1 + r2 * (K1 + r2 * (K2 + r2 * K3))) / (
        1 + r2 * (K4 + r2 * (K5 + r2 * K6))
    )
    deltaX = 2 * P1 * xp * yp + P2 * (r2 + 2 * xp * xp)
    deltaY = 2 * P2 * xp * yp + P1 * (r2 + 2 * yp * yp)
    xpp = xp * cdist + deltaX
    ypp = yp * cdist + deltaY
    u = FX * xpp + CX
    v = FY * ypp + CY
    return u, v


def project_jacobian_xyz(x, y, z):
    v0 = P1 * y
    v1 = P2 * x
    v3 = x * x
    v4 = y * y
    v5 = v3 + v4
    v7 = z * z
    v6 = v7 * v7
    v2 = v6 * v7
    v8 = K5 * v7
    v9 = K6 * v5
    v10 = K4 * v6 + v5 * (v8 + v9)
    v11 = v10 * v5 + v2
    v12 = v11 * v11
    v13 = 2 * v12
    v14 = K2 * v7
    v15 = K3 * v5
    v16 = K1 * v6 + v5 * (v14 + v15)
    v17 = v16 * v5 + v2
    v18 = v17 * z * (v10 + v5 * (v8 + 2 * v9))
    v19 = 2 * v18
    v20 = v16 + v5 * (v14 + 2 * v15)
    v21 = 2 * v20
    v22 = v11 * z
    v23 = 1 / v7
    v24 = 1 / v12
    v25 = FX * v24
    v26 = v23 * v25
    v27 = P2 * y
    v28 = x * y
    v29 = 2 * v12 * (P1 * x + v27) - 2 * v18 * v28 + 2 * v20 * v22 * v28
    v30 = 1 / (v7 * z)
    v31 = 2 * x
    v32 = v22 * (v17 + v21 * v5)
    v33 = FY * v24
    v34 = v23 * v33

    du_dx = v26 * (v13 * (v0 + 3 * v1) - v19 * v3 + v22 * (v17 + v21 * v3))
    du_dy = v26 * v29
    du_dz = (
        -v25 * v30 * (v13 * (P2 * (3 * v3 + v4) + v0 * v31) - v18 * v31 * v5 + v32 * x)
    )
    dv_dx = v29 * v34
    dv_dy = v34 * (v13 * (3 * v0 + v1) - v19 * v4 + v22 * (v17 + v21 * v4))
    dv_dz = (
        -v30 * v33 * (v13 * (P1 * (v3 + 3 * v4) + v27 * v31) - v19 * v5 * y + v32 * y)
    )

    return np.array([du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz])


def project_jacobian_params(x, y, z):

    w1 = x * x
    w2 = y * y
    w3 = w1 + w2
    w5 = z * z
    w4 = w5 * w5
    w0 = w4 * w5
    w6 = w0 + w3 * (K1 * w4 + w3 * (K2 * w5 + K3 * w3))
    w7 = w6 * z
    w8 = w7 * x
    w9 = 2 * x * y
    w10 = 3 * w1 + w2
    w11 = w0 + w3 * (K4 * w4 + w3 * (K5 * w5 + K6 * w3))
    w12 = 1 / w5
    w13 = 1 / w11
    w14 = w12 * w13
    w15 = w3 * z * w5
    w16 = FX * x
    w17 = w13 * w16
    w18 = w3 * w3
    w19 = w18 * z
    w20 = FX * w12
    w21 = w3 * w18 / z
    w22 = w13 * w13
    w23 = w22 * w6
    w24 = w16 * w23
    w25 = w18 * w22
    w26 = w7 * y
    w27 = w1 + 3 * w2
    w28 = FY * y
    w29 = w13 * w28
    w30 = FY * w12
    w31 = w23 * w28

    du_fx = w14 * (w11 * (P1 * w9 + P2 * w10) + w8)
    du_fy = 0
    du_cx = 1
    du_cy = 0
    du_k1 = w15 * w17
    du_k2 = w17 * w19
    du_p1 = w20 * w9
    du_p2 = w10 * w20
    du_k3 = w17 * w21
    du_k4 = -w15 * w24
    du_k5 = -FX * w25 * w8
    du_k6 = -w21 * w24
    dv_fx = 0
    dv_fy = w14 * (w11 * (P1 * w27 + P2 * w9) + w26)
    dv_cx = 0
    dv_cy = 1
    dv_k1 = w15 * w29
    dv_k2 = w19 * w29
    dv_p1 = w27 * w30
    dv_p2 = w30 * w9
    dv_k3 = w21 * w29
    dv_k4 = -w15 * w31
    dv_k5 = -FY * w25 * w26
    dv_k6 = -w21 * w31

    return np.array(
        [
            du_fx,
            du_fy,
            du_cx,
            du_cy,
            du_k1,
            du_k2,
            du_p1,
            du_p2,
            du_k3,
            du_k4,
            du_k5,
            du_k6,
            dv_fx,
            dv_fy,
            dv_cx,
            dv_cy,
            dv_k1,
            dv_k2,
            dv_p1,
            dv_p2,
            dv_k3,
            dv_k4,
            dv_k5,
            dv_k6,
        ]
    )


def unproject_jacobi(u, v, N):
    x0 = (u - CX) / FX
    y0 = (v - CY) / FY

    x = x0
    y = y0
    for _ in range(N):
        r2 = x * x + y * y
        icdist = (1 + r2 * (K4 + r2 * (K5 + r2 * K6))) / (
            1 + r2 * (K1 + r2 * (K2 + r2 * K3))
        )
        assert icdist >= 0, f"{icdist=} < 0"
        delta_x = 2 * P1 * x * y + P2 * (r2 + 2 * x * x)
        delta_y = 2 * P2 * x * y + P1 * (r2 + 2 * y * y)
        x = (x0 - delta_x) * icdist
        y = (y0 - delta_y) * icdist

    norm = sqrt(1 + x * x + y * y)
    x = x / norm
    y = y / norm
    z = 1 / norm
    return x, y, z


def distort(xp, yp):
    r2 = xp * xp + yp * yp
    cdist = (1 + r2 * (K1 + r2 * (K2 + r2 * K3))) / (
        1 + r2 * (K4 + r2 * (K5 + r2 * K6))
    )
    deltaX = 2 * P1 * xp * yp + P2 * (r2 + 2 * xp * xp)
    deltaY = 2 * P2 * xp * yp + P1 * (r2 + 2 * yp * yp)
    xpp = xp * cdist + deltaX
    ypp = yp * cdist + deltaY
    return np.array([xpp, ypp])


def distort_jacobian_xpyp(xp, yp):
    v0 = xp**2
    v1 = yp**2
    v2 = v0 + v1
    v3 = K6 * v2
    v4 = K4 + v2 * (K5 + v3)
    v5 = v2 * v4 + 1
    v6 = v5**2
    v7 = 1 / v6
    v8 = P1 * yp
    v9 = P2 * xp
    v10 = 2 * v6
    v11 = K3 * v2
    v12 = K1 + v2 * (K2 + v11)
    v13 = v12 * v2 + 1
    v14 = v13 * (v2 * (K5 + 2 * v3) + v4)
    v15 = 2 * v14
    v16 = v12 + v2 * (K2 + 2 * v11)
    v17 = 2 * v16
    v18 = xp * yp
    v19 = 2 * v7 * (-v14 * v18 + v16 * v18 * v5 + v6 * (P1 * xp + P2 * yp))

    dxpp_dxp = v7 * (-v0 * v15 + v10 * (v8 + 3 * v9) + v5 * (v0 * v17 + v13))
    dxpp_dyp = v19
    dypp_dxp = v19
    dypp_dyp = v7 * (-v1 * v15 + v10 * (3 * v8 + v9) + v5 * (v1 * v17 + v13))

    return np.array([[dxpp_dxp, dxpp_dyp], [dypp_dxp, dypp_dyp]])


def unproject_newton(u, v, N, eps=None):  # pylint: disable=unused-argument
    p_dist = np.array([(u - CX) / FX, (v - CY) / FY])
    p_undist = p_dist.copy()
    for _ in range(N):
        f = distort(*p_undist)
        J = distort_jacobian_xpyp(*p_undist)
        e = f - p_dist
        p_undist -= inv(J) @ e
        # if norm(e) < eps:
        #     break

    x, y = p_undist
    length = sqrt(1 + x * x + y * y)
    x = x / length
    y = y / length
    z = 1 / length
    return x, y, z


unproject = unproject_newton
