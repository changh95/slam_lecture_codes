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
Computes jacobian expressions for the radtan8 camera model.
"""
from itertools import count
from sympy import symbols, diff, simplify, cse
from sympy.printing import cxxcode
from sympy.codegen.rewriting import create_expand_pow_optimization

fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 = symbols(
    "fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6"
)


def distort_symbolic(xp, yp):
    r2 = xp * xp + yp * yp
    cdist = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (
        1 + r2 * (k4 + r2 * (k5 + r2 * k6))
    )
    deltaX = 2 * p1 * xp * yp + p2 * (r2 + 2 * xp * xp)
    deltaY = 2 * p2 * xp * yp + p1 * (r2 + 2 * yp * yp)
    xpp = xp * cdist + deltaX
    ypp = yp * cdist + deltaY
    return xpp, ypp


def project_symbolic(x, y, z):
    xp = x / z
    yp = y / z
    xpp, ypp = distort_symbolic(xp, yp)
    u = fx * xpp + cx
    v = fy * ypp + cy

    return u, v


def cse_as_cxx(cse_tuple, expr_names):
    expand_pow = create_expand_pow_optimization(20)
    betterformat = lambda e: cxxcode(expand_pow(e))

    subexprs, rhss = cse_tuple

    res = ""
    for var, expr in subexprs:
        res += f"const Scalar {var} = {betterformat(expr)};\n"
    res += "\n"

    for lhs, rhs in zip(expr_names, rhss):
        res += f"const Scalar {lhs} = {betterformat(rhs)};\n"

    return res


def distort_jacobian_to_xpyp():
    # Requires uncommenting line above for xp, yp
    var_names = (symbols(f"v{i}") for i in count())
    xp, yp = symbols("xp yp")
    xpp, ypp = distort_symbolic(xp, yp)

    dxpp_dxp = simplify(diff(xpp, xp))
    dxpp_dyp = simplify(diff(xpp, yp))
    dypp_dxp = simplify(diff(ypp, xp))
    dypp_dyp = simplify(diff(ypp, yp))
    cse_tuple = cse([dxpp_dxp, dxpp_dyp, dypp_dxp, dypp_dyp], var_names)
    names = ["dxpp_dxp", "dxpp_dyp", "dypp_dxp", "dypp_dyp"]
    return cse_tuple, names


def project_jacobian_to_3dpoint():
    var_names = (symbols(f"v{i}") for i in count())
    x, y, z = symbols("x y z")
    u, v = project_symbolic(x, y, z)

    du_dx = simplify(diff(u, x))
    du_dy = simplify(diff(u, y))
    du_dz = simplify(diff(u, z))
    dv_dx = simplify(diff(v, x))
    dv_dy = simplify(diff(v, y))
    dv_dz = simplify(diff(v, z))
    cse_tuple = cse([du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz], var_names)
    names = ["du_dx", "du_dy", "du_dz", "dv_dx", "dv_dy", "dv_dz"]
    return cse_tuple, names


def project_jacobian_to_intrinsics():
    var_names = (symbols(f"w{i}") for i in count())
    x, y, z = symbols("x y z")
    u, v = project_symbolic(x, y, z)

    du_fx = simplify(diff(u, fx))
    du_fy = simplify(diff(u, fy))
    du_cx = simplify(diff(u, cx))
    du_cy = simplify(diff(u, cy))
    du_k1 = simplify(diff(u, k1))
    du_k2 = simplify(diff(u, k2))
    du_p1 = simplify(diff(u, p1))
    du_p2 = simplify(diff(u, p2))
    du_k3 = simplify(diff(u, k3))
    du_k4 = simplify(diff(u, k4))
    du_k5 = simplify(diff(u, k5))
    du_k6 = simplify(diff(u, k6))

    dv_fx = simplify(diff(v, fx))
    dv_fy = simplify(diff(v, fy))
    dv_cx = simplify(diff(v, cx))
    dv_cy = simplify(diff(v, cy))
    dv_k1 = simplify(diff(v, k1))
    dv_k2 = simplify(diff(v, k2))
    dv_p1 = simplify(diff(v, p1))
    dv_p2 = simplify(diff(v, p2))
    dv_k3 = simplify(diff(v, k3))
    dv_k4 = simplify(diff(v, k4))
    dv_k5 = simplify(diff(v, k5))
    dv_k6 = simplify(diff(v, k6))

    # fmt:off
    cse_tuple = cse(
        [
            du_fx, du_fy, du_cx, du_cy, du_k1, du_k2, du_p1, du_p2,
            du_k3, du_k4, du_k5, du_k6,
            dv_fx, dv_fy, dv_cx, dv_cy, dv_k1, dv_k2, dv_p1, dv_p2,
            dv_k3, dv_k4, dv_k5, dv_k6
        ],
        var_names
    )
    names = [
        "du_fx", "du_fy", "du_cx", "du_cy", "du_k1", "du_k2", "du_p1", "du_p2",
        "du_k3", "du_k4", "du_k5", "du_k6",
        "dv_fx", "dv_fy", "dv_cx", "dv_cy", "dv_k1", "dv_k2", "dv_p1", "dv_p2",
        "dv_k3", "dv_k4", "dv_k5", "dv_k6"
    ]
    # fmt:on

    return cse_tuple, names


def main():
    print("[Project jacobians of 2D point w.r.t 3D point]")
    res = project_jacobian_to_3dpoint()
    print(cse_as_cxx(*res))

    print("[Project jacobians of 2D point w.r.t intrinsics]")
    res = project_jacobian_to_intrinsics()
    print(cse_as_cxx(*res))

    print("[Distort jacobians of distorted 2D point from undistorted 2D point]")
    res = distort_jacobian_to_xpyp()
    print(cse_as_cxx(*res))

    print("[Unproject jacobians of 3D point w.r.t 2D point]")
    print("Not implemented yet")

    print("[Unproject jacobians of 3D point w.r.t intrinsics]")
    print("Not implemented yet")


if __name__ == "__main__":
    main()
