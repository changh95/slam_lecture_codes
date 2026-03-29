#!/usr/bin/env python3
#
# BSD 3-Clause License
#
# This file is part of the Basalt project.
# https://gitlab.com/VladyslavUsenko/basalt.git
#
# Copyright (c) 2023, Collabora Ltd.
# All rights reserved.
#
# Author: Mateo de Mayo <mateo.demayo@collabora.com>
#

"""
This file tries to numerically fit a kannala-brandt4 camera model into a
radial-tangential8 model by projecting a simulated 3D grid. Parameters of the
inpout kb4 model are below.
"""

import numpy as np
from scipy.optimize import minimize
from numpy import sqrt, arctan2

# Define the projection function for Kannala-Brandt model
def kb4_projection(params, points):
    fx, fy, cx, cy, k1, k2, k3, k4 = params
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Compute the distorted image coordinates
    r2 = x**2 + y**2
    r = sqrt(r2)
    t = arctan2(r, z)
    d = t + k1 * t**3 + k2 * t**5 + k3 * t**7 + k4 * t**9
    u = fx * d * (x / r) + cx
    v = fy * d * (y / r) + cy

    return np.column_stack((u, v))


def rt8_projection(params, points):
    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 = params
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    xp = x / z
    yp = y / z
    r2 = xp * xp + yp * yp
    cdist = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (
        1 + r2 * (k4 + r2 * (k5 + r2 * k6))
    )
    deltaX = 2 * p1 * xp * yp + p2 * (r2 + 2 * xp * xp)
    deltaY = 2 * p2 * xp * yp + p1 * (r2 + 2 * yp * yp)
    xpp = xp * cdist + deltaX
    ypp = yp * cdist + deltaY
    u = fx * xpp + cx
    v = fy * ypp + cy
    return np.column_stack((u, v))


# Define the error function for optimization
def error_function(rt8_params, kb4_projections, points_3d, fx, fy, cx, cy):
    rt8_params = np.concatenate(([fx, fy, cx, cy], rt8_params))
    rt8_projections = rt8_projection(rt8_params, points_3d)
    error = np.linalg.norm(kb4_projections - rt8_projections)
    print(error)
    return error


# Initial parameters for Kannala-Brandt model
fx = 421.23561386503078
fy = 421.4668948188076
cx = 467.746800268133
cy = 484.4807401358835
k1 = 0.19017898170828449
k2 = 0.0546719732148914
k3 = -0.2491419411107316
k4 = 0.10310283081793621
initial_params = [fx, fy, cx, cy, k1, k2, k3, k4]  # Replace with your values

# Simulated 3D points
xs = 5  # Span of X values in meters (+-)
ys = 5  # Span of X values in meters (+-)
zs = 2  # Span of X values in meters (+)
point_count = 1000
np.random.seed(0)
points_3d = (np.random.rand(point_count, 3) - (0.5, 0.5, 0)) * [xs * 2, ys * 2, zs]

# Simulated 2D points (corresponding to the above 3D points)
kb4_projections = kb4_projection(initial_params, points_3d)

# Initial guess for the pinhole radial-tangential parameters
initial_guess = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Optimize the pinhole radial-tangential parameters using scipy
result = minimize(
    error_function,
    initial_guess,
    args=(kb4_projections, points_3d, fx, fy, cx, cy),
    method="powell",
    options={"maxfev": 100000, "maxiter": 100000},
)
# Nelder-Mead (ERROR 9909): [-0.0009629836688293419, 0.0004938171615563396, 9.738350693900328e-08, -9.316121850967052e-08, 1.6983755696426284e-09, -0.005425949815434813, 0.008733454263167094, 7.668313326355561e-07]
# powell      (ERROR 3554): [0.00033910941623906436, 0.0016526977059368436, 2.0382438109564118e-07, -1.9941820902364245e-07, 4.7690647490781044e-08, 0.08285582144993399, 0.007074083130056834, 1.2021299767958244e-05]
# COBYLA      (ERROR 3e16): [0.0010986328124991318, -1.2342032253716455e-08, 1.299082237616408e-14, -3.742288356480218e-15, -9.999999924791063e-05, 1.001903125, 1.6792722237236482e-18, 5.09882003371107e-17]


# Extract the optimized parameters
optimized_params = [fx, fy, cx, cy] + list(result.x)

print(result)
print("Optimized Parameters:")
print("fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6")
print(optimized_params)
