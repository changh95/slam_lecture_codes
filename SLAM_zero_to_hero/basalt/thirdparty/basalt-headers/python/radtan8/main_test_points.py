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
Different tests on the camera models applied on a grid of points.
By default it plots good/bad/invalid points and the different rpmax radiuses.
"""
import itertools as it

import numpy as np
from numpy.linalg import norm

import basalt_radtan as bs_rt
import opencv_radtan as cv_rt
from plot_image_points import plot
from main_compute_rpmax import compute_rpmax
from calibration import METRIC_RADIUS

EPS = 1e-5

BAD = 0
GOOD = 1
INVALID = 2


def test_point_inyective(x, y, z, rpmax, mut, N=None, eps=None):
    if z <= EPS:
        return INVALID

    name = mut.__name__
    nx, ny, nz = np.array([x, y, z]) / norm(np.array([x, y, z]))
    print(f"\nPoint [{x}, {y}, {z}] (unit [{nx}, {ny}, {nz}]):")

    mu, mv = mut.project(x, y, z)
    print(f"{name} project [{x}, {y}, {z}] -> [{mu}, {mv}]")

    rp2 = (x / z) ** 2 + (y / z) ** 2
    if rpmax > 0 and rp2 > rpmax**2:
        return INVALID

    mx, my, mz = mut.unproject(mu, mv, N=N, eps=eps)
    print(f"{name} unproject [{mu}, {mv}] -> [{mx}, {my}, {mz}]")

    up_result = "GOOD" if np.isclose((nx, ny, nz), (mx, my, mz)).all() else "BAD"
    print(
        f"{name} unproject-projection {up_result} [{nx}, {ny}, {nz}] -> [{mx}, {my}, {mz}]"
    )

    nmu, nmv = mut.project(mx, my, mz)
    pu_result = "GOOD" if np.isclose((mu, mv), (nmu, nmv)).all() else "BAD"
    print(f"{name} project-unprojection {pu_result} [{mu}, {mv}] -> [{nmu}, {nmv}]")

    return up_result == "GOOD"  # Unproject project, same as basalt test
    # return pu_result == "GOOD" # Project unproject, this one works


# fmt: off
def test_point_matches_opencv(x, y, z, asserts_enabled=True, N=None):
    if z < 1e-5:
        return

    nx, ny, nz = np.array([x, y, z]) / norm(np.array([x, y, z]))
    print(f"\nPoint [{x}, {y}, {z}] (unit [{nx}, {ny}, {nz}]):")

    # 1. Same project
    bu, bv = bs_rt.project(x, y, z)
    cu, cv = cv_rt.project(x, y, z)
    print(f"basalt project [{x}, {y}, {z}] -> [{bu}, {bv}]")
    print(f"opencv project [{x}, {y}, {z}] -> [{cu}, {cv}]")
    assert np.isclose((bu, bv), (cu, cv)).all() or not asserts_enabled

    # 2. Same unproject
    bx, by, bz = bs_rt.unproject(bu, bv, N, None)
    cx, cy, cz = cv_rt.unproject(cu, cv, N)
    print(f"basalt unproject [{bu}, {bv}] -> [{bx}, {by}, {bz}]")
    print(f"opencv unproject [{cu}, {cv}] -> [{cx}, {cy}, {cz}]")
    assert np.isclose((bx, by, bz), (cx, cy, cz)).all() or not asserts_enabled

    # 3. Same project jacobians w.r.t 3D point
    b_du_dx, b_du_dy, b_du_dz, b_dv_dx, b_dv_fdy, b_dv_dz = bs_rt.project_jacobian_xyz(x, y, z)
    c_du_dx, c_du_dy, c_du_dz, c_dv_dx, c_dv_fdy, c_dv_dz = cv_rt.project_jacobian_xyz(x, y, z)
    print(f"basalt project_jacobian_xyz [{x}, {y}, {z}] -> [{b_du_dx}, {b_du_dy}, {b_du_dz}, {b_dv_dx}, {b_dv_fdy}, {b_dv_dz}]")
    print(f"opencv project_jacobian_xyz [{x}, {y}, {z}] -> [{c_du_dx}, {c_du_dy}, {c_du_dz}, {c_dv_dx}, {c_dv_fdy}, {c_dv_dz}]")
    assert np.isclose((b_du_dx, b_du_dy, b_du_dz, b_dv_dx, b_dv_fdy, b_dv_dz), (c_du_dx, c_du_dy, c_du_dz, c_dv_dx, c_dv_fdy, c_dv_dz)).all() or not asserts_enabled

    # 4. Same project jacobians w.r.t. intrinsic parameters
    b_du_fx, b_du_fy, b_du_cx, b_du_cy, b_du_k1, b_du_k2, b_du_p1, b_du_p2, b_du_k3, b_du_k4, b_du_k5, b_du_k6, b_dv_fx, b_dv_fy, b_dv_cx, b_dv_cy, b_dv_k1, b_dv_k2, b_dv_p1, b_dv_p2, b_dv_k3, b_dv_k4, b_dv_k5, b_dv_k6 = bs_rt.project_jacobian_params(x, y, z)
    c_du_fx, c_du_fy, c_du_cx, c_du_cy, c_du_k1, c_du_k2, c_du_p1, c_du_p2, c_du_k3, c_du_k4, c_du_k5, c_du_k6, c_dv_fx, c_dv_fy, c_dv_cx, c_dv_cy, c_dv_k1, c_dv_k2, c_dv_p1, c_dv_p2, c_dv_k3, c_dv_k4, c_dv_k5, c_dv_k6 = cv_rt.project_jacobian_params(x, y, z)
    print(f"basalt project_jacobian_params [{x}, {y}, {z}] -> [{b_du_fx}, {b_du_fy}, {b_du_cx}, {b_du_cy}, {b_du_k1}, {b_du_k2}, {b_du_p1}, {b_du_p2}, {b_du_k3}, {b_du_k4}, {b_du_k5}, {b_du_k6}, {b_dv_fx}, {b_dv_fy}, {b_dv_cx}, {b_dv_cy}, {b_dv_k1}, {b_dv_k2}, {b_dv_p1}, {b_dv_p2}, {b_dv_k3}, {b_dv_k4}, {b_dv_k5}, {b_dv_k6}]")
    print(f"opencv project_jacobian_params [{x}, {y}, {z}] -> [{c_du_fx}, {c_du_fy}, {c_du_cx}, {c_du_cy}, {c_du_k1}, {c_du_k2}, {c_du_p1}, {c_du_p2}, {c_du_k3}, {c_du_k4}, {c_du_k5}, {c_du_k6}, {c_dv_fx}, {c_dv_fy}, {c_dv_cx}, {c_dv_cy}, {c_dv_k1}, {c_dv_k2}, {c_dv_p1}, {c_dv_p2}, {c_dv_k3}, {c_dv_k4}, {c_dv_k5}, {c_dv_k6}]")
    assert np.isclose((b_du_fx, b_du_fy, b_du_cx, b_du_cy, b_du_k1, b_du_k2, b_du_p1, b_du_p2, b_du_k3, b_du_k4, b_du_k5, b_du_k6, b_dv_fx, b_dv_fy, b_dv_cx, b_dv_cy, b_dv_k1, b_dv_k2, b_dv_p1, b_dv_p2, b_dv_k3, b_dv_k4, b_dv_k5, b_dv_k6), (c_du_fx, c_du_fy, c_du_cx, c_du_cy, c_du_k1, c_du_k2, c_du_p1, c_du_p2, c_du_k3, c_du_k4, c_du_k5, c_du_k6, c_dv_fx, c_dv_fy, c_dv_cx, c_dv_cy, c_dv_k1, c_dv_k2, c_dv_p1, c_dv_p2, c_dv_k3, c_dv_k4, c_dv_k5, c_dv_k6)).all() or not asserts_enabled

    # 5. Check unproject "inverted" project (usually fails, so no assert)
    b_up_result = "GOOD" if np.isclose((nx, ny, nz), (bx, by, bz)).all() else "BAD"
    c_up_result = "GOOD" if np.isclose((nx, ny, nz), (cx, cy, cz)).all() else "BAD"
    print(f"basalt unproject-projection {b_up_result} [{nx}, {ny}, {nz}] -> [{bx}, {by}, {bz}]")
    print(f"opencv unproject-projection {c_up_result} [{nx}, {ny}, {nz}] -> [{cx}, {cy}, {cz}]")
    # assert np.isclose((nx, ny, nz), (bx, by, bz)).all() # OpenCV can fail this

    # 6. Check project "inverts" unproject (usually fails, so no assert)
    nbu, nbv = bs_rt.project(bx, by, bz)
    ncu, ncv = cv_rt.project(cx, cy, cz)
    b_pu_result = "GOOD" if np.isclose((bu, bv), (nbu, nbv)).all() else "BAD"
    c_pu_result = "GOOD" if np.isclose((cu, cv), (ncu, ncv)).all() else "BAD"
    print(f"basalt project-unprojection {b_pu_result} [{bu}, {bv}] -> [{nbu}, {nbv}]")
    print(f"opencv project-unprojection {c_pu_result} [{cu}, {cv}] -> [{ncu}, {ncv}]")
    # assert np.isclose((bu, bv), (nbu, nbv)).all() # OpenCV can fail this
# fmt: on


def main():
    # Test points
    xs = np.linspace(-10, 10, num=21)
    ys = np.linspace(-10, 10, num=21)
    zs = np.linspace(0, 5, num=6)

    # TODO: Use the xs/ys/zs from above (the same from basalt tests)
    # My points
    r = 10
    xs = np.linspace(-r, r, 100)
    ys = np.linspace(-r, r, 100)
    zs = np.array([1])

    good_points = []
    bad_points = []
    invalid_points = []
    mut = bs_rt  # module under test
    rpmax, unscaled_rpmax = compute_rpmax()

    for x, y, z in it.product(xs, ys, zs):
        ret = test_point_inyective(x, y, z, rpmax, mut, N=20, eps=1e-5)
        if ret == GOOD:
            good_points.append(bs_rt.project(x, y, z))
        elif ret == BAD:
            bad_points.append(bs_rt.project(x, y, z))
        elif ret == INVALID:
            invalid_points.append(bs_rt.project(x, y, z))
    print(f"{len(good_points)=}, {len(bad_points)=}, {len(invalid_points)=}")
    metric_radius = METRIC_RADIUS if METRIC_RADIUS > 0 else 0
    plot(
        good_points,
        bad_points,
        invalid_points,
        [metric_radius, rpmax, unscaled_rpmax],
        ["metric_radius", "rpmax", "unscaled_rpmax"],
    )


if __name__ == "__main__":
    main()
