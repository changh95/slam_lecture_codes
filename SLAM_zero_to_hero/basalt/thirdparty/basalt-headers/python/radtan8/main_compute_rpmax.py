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
Computes rpmax and shows graphs related to it
"""

from math import sqrt, inf
from sympy import symbols, lambdify, diff
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from numpy import sign
from numpy.linalg import inv
from scipy.optimize import OptimizeResult
from calibration import CALIB, METRIC_RADIUS, WIDTH, HEIGHT, FX, FY, CX, CY

from basalt_radtan import unproject_newton

arr = lambda xs: np.array(xs, dtype=np.float64)

fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6, x, y, z = symbols(
    "fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6 x y z"
)

xp = x / z
yp = y / z
rp2 = xp * xp + yp * yp
cdist = (1 + rp2 * (k1 + rp2 * (k2 + rp2 * k3))) / (
    1 + rp2 * (k4 + rp2 * (k5 + rp2 * k6))
)
deltaX = 2 * p1 * xp * yp + p2 * (rp2 + 2 * xp * xp)
deltaY = 2 * p2 * xp * yp + p1 * (rp2 + 2 * yp * yp)
xpp = xp * cdist + deltaX
ypp = yp * cdist + deltaY
u = fx * xpp + cx
v = fy * ypp + cy

rpp2 = xpp**2 + ypp**2

rpp2_from_xy = rpp2.subs({**CALIB, "z": 1})
rpp2_from_xy_f = lambdify([x, y], rpp2_from_xy)  # TODO: duplicated def
rp2_from_xy = rp2.subs({**CALIB, "z": 1})
rp2_from_xy_f = lambdify([x, y], rp2_from_xy)

drpp2_dx_from_xy = diff(rpp2_from_xy, x)
drpp2_dx_from_xy_f = lambdify([x, y], drpp2_dx_from_xy)
drpp2_dy_from_xy = diff(rpp2_from_xy, y)
drpp2_dy_from_xy_f = lambdify([x, y], drpp2_dy_from_xy)
J_rpp2_from_xy_f = lambda xy: arr([drpp2_dx_from_xy_f(*xy), drpp2_dy_from_xy_f(*xy)])
J_error_from_xy_f = lambda xy: -J_rpp2_from_xy_f(xy)


def get_corner_with_highest_rpp2():
    ul = (0, 0)
    ur = (WIDTH, 0)
    bl = (0, HEIGHT)
    br = (WIDTH, HEIGHT)
    uvs = [ul, ur, bl, br]

    selected_corner_xy = (0, 0)
    selected_rpp2 = -inf
    for u, v in uvs:
        x, y, z = unproject_newton(u, v, 15, None)
        x, y, z = arr([x, y, z]) / z  # To plane z=1
        corner_rpp2 = rpp2_from_xy_f(x, y)
        if corner_rpp2 > selected_rpp2:
            selected_corner_xy = (x, y)
            selected_rpp2 = corner_rpp2
    return selected_corner_xy, selected_rpp2


corner, corner_rpp2 = get_corner_with_highest_rpp2()
rpp2_bound = 2 * corner_rpp2
error = lambda xy: rpp2_bound - rpp2_from_xy_f(*xy)


def numeric_J_f_from_xy(f):
    def _numeric_J_f_from_xy(xy):
        rel_step = np.finfo(xy.dtype).eps ** (1 / 2)  # sqrt(1.1920929e-07) for float32
        h = rel_step * sign(xy) * np.maximum(1, abs(xy))
        # h = array([0.01, 0.01])
        # TODO: ((xy + h) - xy) could have a zero if xy is too big, check that?
        f_xy = f(xy)
        df_dx = (f(xy + [h[0], 0]) - f_xy) / h[0]
        df_dy = (f(xy + [0, h[1]]) - f_xy) / h[1]
        return arr([[df_dx, df_dy]])

    return _numeric_J_f_from_xy


def my_least_squares_old(f, x0, max_nfev, jac, ftol=1e-8):
    # NOTE: Gauss newton but doesnt work, goes astray
    x = arr(x0)
    fx = f(x)
    for i in range(1, max_nfev + 1):
        J = jac(x)
        JtJ_inv = 1 / (J[0] ** 2 + J[1] ** 2)
        JtJ_inv_other = inv(np.atleast_2d(J.T @ J))
        assert JtJ_inv == JtJ_inv_other

        x = x - JtJ_inv * J * fx

        old_fx = fx
        fx = f(x)
        if abs(old_fx - fx) < ftol * old_fx:
            break

    return OptimizeResult(cost=fx, x=x, nfev=i)


def my_least_squares(f, x0, max_nfev, jac, ftol=1e-8):
    # NOTE: Gauss newton but doesnt work, inv fails, detects singular matrix
    x = x0
    fx = f(x)
    for i in range(1, max_nfev + 1):
        J = jac(x)
        J_pseudoinverse = inv(J.T @ J) @ J.T
        x = x.reshape(2, 1)  # x as column vector
        x = x - J_pseudoinverse * fx
        x = x.reshape(2)

        old_fx = fx
        fx = f(x)
        if abs(old_fx - fx) < ftol * old_fx:
            break

    return OptimizeResult(cost=fx, x=x, nfev=i)


def get_maxrp2_corner():
    ul = (0, 0)
    ur = (2 * CX, 0)
    bl = (0, 2 * CY)
    br = (2 * CX, 2 * CY)
    uvs = [ul, ur, bl, br]

    selected_corner_xy = arr([0, 0])
    selected_rp2 = -inf
    for u, v in uvs:
        x, y, z = unproject_newton(u, v, 15, None)
        x, y, z = arr([x, y, z]) / z  # To plane z=1
        corner_rp2 = rp2_from_xy_f(x, y)
        if corner_rp2 > selected_rp2:
            selected_corner_xy = arr([x, y])
            selected_rp2 = corner_rp2
    return selected_corner_xy, selected_rp2


def gradient_ascent(f, x0, max_nfev, jac, ftol=1e-4, in_domain_bounds=lambda x: True):
    x = x0
    fx = f(x)
    for i in range(1, max_nfev + 1):
        n = 0.1
        x = x + n * jac(x).reshape(2)

        old_fx = fx
        fx = f(x)
        if abs(old_fx - fx) < ftol * old_fx:
            break

        if not in_domain_bounds(x):
            return OptimizeResult(cost=inf, x=arr([inf, inf]), nfev=i)

    return OptimizeResult(cost=fx, x=x, nfev=i)


def plot_rpp2_from_xy(border=10, points=(), names=()):
    # RPP2 from XY
    xs = np.linspace(-border, border, 100)
    ys = np.linspace(-border, border, 100)
    xs, ys = np.meshgrid(xs, ys)
    rpp2s = rpp2_from_xy_f(xs, ys)
    _, ax3 = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax3.plot_surface(xs, ys, rpp2s, cmap=cm.PiYG, label="rpp2(x, y)")

    # HACK: see stackoverflow.com/q/54994600
    surf._edgecolors2d = surf._edgecolor3d  # pylint: disable=protected-access
    surf._facecolors2d = surf._facecolor3d  # pylint: disable=protected-access

    # RPP2 of arbitrary points
    for point, name in zip(points, names):
        ax3.plot(point[0], point[1], rpp2_from_xy_f(*point), "o", zorder=3, label=name)

    ax3.legend()


def plot_cdist_rpp2_from_x(border=10, points=(), names=()):
    _, ax2 = plt.subplots()

    # CDIST from X
    cdist_from_x = cdist.subs({**CALIB, "z": 1, "y": 0})
    cdist_from_x_f = lambdify(x, cdist_from_x)
    xs = np.linspace(0, border, 100)
    ax2.plot(xs, cdist_from_x_f(xs), label="cdist(x)")

    # RPP2 from X
    rpp2_from_x = rpp2.subs({**CALIB, "z": 1, "y": 0})
    rpp2_from_x_function = lambdify(x, rpp2_from_x)
    xs = np.linspace(0, border, 100)
    ax2.plot(xs, rpp2_from_x_function(xs), label="rpp2(x)")

    # RPP2/CDIST of arbitrary X points
    for point, name in zip(points, names):
        ax2.plot(point, cdist_from_x_f(point), "o", label=f"cdist(X={name})")
        ax2.plot(point, rpp2_from_xy_f(point, 0), "o", label=f"rpp2(X={name})")

    ax2.legend()


def compute_rpmax(plot=False):
    MAX_ITERS = 1000
    NUDGE = 0.1
    UNPROJECT_ITERS = 15
    CORNER_BOUND_SCALE = 1.5
    RPMAX_SCALE = 0.85

    # guess = corner
    X, Y, Z = unproject_newton(NUDGE * FX + CX, NUDGE * FY + CY, UNPROJECT_ITERS, None)
    X, Y, Z = arr([X, Y, Z]) / Z  # To plane Z=1
    guess = arr([X, Y])

    # Compute rpmax with different types of scipy.optimize.least_squares like:
    # result = least_squares(error, guess, max_nfev=MAX_ITERS)
    # result = least_squares(error, guess, max_nfev=MAX_ITERS, jac=J_error_from_xy_f)
    # result = least_squares(error, guess, max_nfev=MAX_ITERS, jac=numeric_J_f_from_xy(error))

    # Or do it our own way with gradient ascent:
    rpp2_from_xy_ff = lambda xy: rpp2_from_xy_f(*xy)

    # Compute corner with farthest unprojection to get our gradient ascent domain bounds
    _, corners_maxrp2 = get_maxrp2_corner()
    in_domain_bounds = (
        lambda xy: rp2_from_xy_f(*xy) < CORNER_BOUND_SCALE * corners_maxrp2
    )

    # Finally, compute point with rpmax
    result = gradient_ascent(
        rpp2_from_xy_ff,
        guess,
        max_nfev=MAX_ITERS,
        jac=numeric_J_f_from_xy(rpp2_from_xy_ff),
        in_domain_bounds=in_domain_bounds,
    )

    # Decide rpmax from optimization result
    print(result)
    optimization_diverged = result.cost == inf
    if not optimization_diverged:
        unscaled_rpmax = sqrt(rp2_from_xy_f(*result.x))
        rpmax = unscaled_rpmax * RPMAX_SCALE
        print("Optimization success")
        print(f"{rpmax=} {unscaled_rpmax=} {RPMAX_SCALE=} {METRIC_RADIUS=}")
        print(f"{METRIC_RADIUS / unscaled_rpmax=} {result.x=} {result.cost}")
    else:
        unscaled_rpmax = 0
        rpmax = 0
        print("Optimization diverged")
        print(
            f"The entire domain of this calibration is considered biyective ({METRIC_RADIUS=})"
        )

    # Plot
    if plot:
        B = 10
        plot_rpp2_from_xy(border=B, points=[result.x], names=["rpmax"])
        plot_cdist_rpp2_from_x(border=B, points=[result.x[0]], names=[result.x[0]])
        plt.show()

    return rpmax, unscaled_rpmax


if __name__ == "__main__":
    compute_rpmax(True)
