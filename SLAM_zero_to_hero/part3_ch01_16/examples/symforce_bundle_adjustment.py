#!/usr/bin/env python3
"""SymForce Tutorial: Bundle Adjustment with the BAL dataset

Loads a BAL problem, builds the BAL reprojection residual symbolically
(angle-axis rotation + translation, with the dataset's f, k1, k2), and optimizes
camera poses and 3D points with SymForce's Levenberg-Marquardt optimizer.

Every iteration is streamed live to a rerun viewer (landmark cloud, camera
centres, squared error and pixel RMSE plots) - see examples/rerun_viz.py. The
run also prints its numbers, so it is useful with no viewer at all.

Deliberate choices, shared with the g2o / GTSAM / Ceres chapters so the four
numbers are directly comparable:
  * Intrinsics (f, k1, k2) are read from the dataset and held FIXED - the
    calib{c} keys are never listed in optimized_keys. Only the 6-DoF camera
    poses and the 3D points are optimized.
  * Camera 0's pose is fixed (also by omission from optimized_keys). That
    removes 6 of the 7 gauge degrees of freedom; overall scale stays free
    because the BAL projection is scale invariant, and LM damping handles that
    remaining direction. No point is fixed - that would over-constrain.
  * No robust loss. SymForce ships one (symforce.opt.noise_models.
    BarronNoiseModel, which covers Huber/Cauchy/Geman-McClure), but it is
    deliberately off here so the reported error IS the minimized objective.

The full BAL problem is used (no subsampling). SymForce builds one Python factor
per observation, so constructing the full Trafalgar problem (36455 observations)
takes a while before the solver runs. Pass a different BAL file as the first
argument if desired.
"""
import sys
import time
from pathlib import Path

import symforce

symforce.set_epsilon_to_symbol()

import numpy as np
import symforce.symbolic as sf
from symforce.opt.factor import Factor
from symforce.opt.optimizer import Optimizer
from symforce.values import Values

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rerun_viz import BUNDLE_ADJUSTMENT_RECORDING, Viz  # noqa: E402

MAX_ITERATIONS = 30


def reprojection_residual(
    cam_pose: sf.Pose3, calib: sf.V3, point: sf.V3, pixel: sf.V2, epsilon: sf.Scalar
) -> sf.V2:
    """BAL reprojection: p' = f (1 + k1 r^2 + k2 r^4) (-P_cam / P_cam.z).

    ``calib`` is an input like ``pixel``, not an optimized variable.
    """
    p_cam = cam_pose * point  # world-to-camera: R * X + t
    x = -p_cam[0] / (p_cam[2] - epsilon)
    y = -p_cam[1] / (p_cam[2] - epsilon)
    r2 = x * x + y * y
    f, k1, k2 = calib[0], calib[1], calib[2]
    dist = 1 + k1 * r2 + k2 * r2 * r2
    return sf.V2(f * dist * x, f * dist * y) - pixel


def load_bal(path):
    with open(path) as fh:
        it = iter(fh.read().split())
    n_cam, n_pt, n_obs = int(next(it)), int(next(it)), int(next(it))
    obs = [(int(next(it)), int(next(it)), float(next(it)), float(next(it)))
           for _ in range(n_obs)]
    cams = [[float(next(it)) for _ in range(9)] for _ in range(n_cam)]
    pts = [[float(next(it)) for _ in range(3)] for _ in range(n_pt)]
    return n_cam, n_pt, obs, cams, pts


def reprojection_metrics(rot, trans, calib, points, cam_idx, pt_idx, pixels):
    """The metrics every chapter reports, from the same formula.

    sq_error = sum over observations of |projected - measured|^2 (raw), and
    rmse_px  = sqrt(sq_error / num_observations)  <- per observation, NOT per
    residual component. Computed here in numpy rather than read off the
    optimizer, because SymForce reports 0.5 * sum of squares while the other
    libraries use other conventions.
    """
    p_cam = np.einsum("oij,oj->oi", rot[cam_idx], points[pt_idx]) + trans[cam_idx]
    xy = -p_cam[:, :2] / p_cam[:, 2:3]
    r2 = np.sum(xy * xy, axis=1)
    f, k1, k2 = calib[cam_idx, 0], calib[cam_idx, 1], calib[cam_idx, 2]
    scale = f * (1.0 + k1 * r2 + k2 * r2 * r2)
    residual = scale[:, None] * xy - pixels
    sq_error = float(np.sum(residual * residual))
    return sq_error, float(np.sqrt(sq_error / len(pixels)))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "problem-21-11315-pre.txt"
    if not Path(path).is_file():
        sys.exit(f"BAL problem file not found: {path}\n"
                 f"The Dockerfile downloads it into /workspace/part3_ch01_16/run; "
                 f"otherwise fetch it from "
                 f"https://grail.cs.washington.edu/projects/bal/")

    print("=== SymForce Tutorial: Bundle Adjustment (BAL) ===\n")
    n_cam, n_pt, obs, cams, pts = load_bal(path)

    used_c = list(range(n_cam))
    used_p = sorted({o[1] for o in obs})
    pt_row = {p: k for k, p in enumerate(used_p)}
    cam_idx = np.array([o[0] for o in obs], dtype=int)
    pt_idx = np.array([pt_row[o[1]] for o in obs], dtype=int)
    pixels = np.array([[o[2], o[3]] for o in obs], dtype=float)
    calib = np.array([cams[c][6:9] for c in used_c], dtype=float)

    print(f"Problem: {n_cam} cameras, {len(used_p)} points, {len(obs)} observations "
          f"(full, no subsampling)")
    print("Intrinsics (f, k1, k2) are fixed; camera 0's pose is fixed; no robust loss.")
    print("Building factors (one per observation; this can take a while)...")
    t0 = time.time()

    eps = sf.numeric_epsilon
    values = Values()
    values["epsilon"] = eps
    for c in used_c:
        rot = sf.Rot3.from_tangent(sf.V3(*cams[c][0:3]), epsilon=eps)
        values[f"cam{c}"] = sf.Pose3(R=rot, t=sf.V3(*cams[c][3:6]))
        values[f"calib{c}"] = sf.V3(*cams[c][6:9])
    for p in used_p:
        values[f"pt{p}"] = sf.V3(*pts[p])
    for i, (c, p, u, v) in enumerate(obs):
        values[f"px{i}"] = sf.V2(u, v)

    factors = [
        Factor(residual=reprojection_residual,
               keys=[f"cam{c}", f"calib{c}", f"pt{p}", f"px{i}", "epsilon"])
        for i, (c, p, u, v) in enumerate(obs)
    ]
    print(f"Built {len(factors)} factors in {time.time() - t0:.1f} s")

    # Fixed by omission: calib{c} for every camera, and cam0's pose (the gauge
    # anchor). Everything listed here is what the solver is allowed to move.
    anchor = used_c[0]
    optimized = ([f"cam{c}" for c in used_c if c != anchor]
                 + [f"pt{p}" for p in used_p])

    def state(v):
        """Rotations, translations, camera centres and points of one iteration.

        Camera centres are C = -R^T t (``pose.inverse().position()``) because the
        BAL poses are world-to-camera; logging t instead would scatter the
        cameras away from the point cloud.
        """
        poses = [values[f"cam{anchor}"] if c == anchor else v[f"cam{c}"]
                 for c in used_c]
        # reshape: the fixed camera is still a symbolic sf.Pose3, whose 3x3
        # rotation flattens to 9 elements under np.array, while the optimized
        # ones are numeric sym.Pose3 and come out as (3, 3) already.
        rot = np.array([np.array(p.rotation().to_rotation_matrix(),
                                 dtype=float).reshape(3, 3)
                        for p in poses])
        trans = np.array([np.array(p.position(), dtype=float).flatten()
                          for p in poses])
        centers = np.array([np.array(p.inverse().position(), dtype=float).flatten()
                            for p in poses])
        points = np.array([np.array(v[f"pt{p}"], dtype=float).flatten()
                           for p in used_p])
        return rot, trans, centers, points

    initial_points = np.array([pts[p] for p in used_p], dtype=float)
    viz = Viz(BUNDLE_ADJUSTMENT_RECORDING, "symforce")
    viz.ba_setup(initial_points)

    print(f"Optimizing (max {MAX_ITERATIONS} LM iterations; the first run compiles "
          f"the symbolic residual)...")
    t0 = time.time()
    # debug_stats=True keeps the Values of every iteration so each step can be
    # streamed; verbose=False silences SymForce's own per-iteration LM log.
    params = Optimizer.Params(
        iterations=MAX_ITERATIONS,
        debug_stats=True,
        verbose=False,
        early_exit_min_reduction=1e-6,
    )
    optimizer = Optimizer(factors=factors, optimized_keys=optimized, params=params)
    result = optimizer.optimize(values)
    print(f"Optimized in {time.time() - t0:.1f} s, status {result.status.name}")

    # result.iterations[0] is the initial state, not a step (SymForce numbers it
    # -1); it is streamed as frame 0 so the timeline matches the sibling
    # chapters. SymForce has no per-iteration callback, so the frames are
    # replayed from the stats debug_stats kept rather than logged during the run.
    # The fixed keys (calib*, cam0) are absent from the per-iteration Values -
    # state() fills cam0 in from the constant input.
    sq_first = rmse_first = None
    for step, it in enumerate(result.iterations):
        v = optimizer.load_iteration_values(it.values)
        rot, trans, centers, points = state(v)
        sq_error, rmse_px = reprojection_metrics(
            rot, trans, calib, points, cam_idx, pt_idx, pixels
        )
        viz.ba_iteration(step, points, centers, sq_error, rmse_px)
        if sq_first is None:
            sq_first, rmse_first = sq_error, rmse_px
        sq_last, rmse_last = sq_error, rmse_px

    print(f"\nIterations: {len(result.iterations) - 1} (max {MAX_ITERATIONS})")
    print(f"Initial: sq_error {sq_first:.2f}  RMSE {rmse_first:.3f} px")
    print(f"Final:   sq_error {sq_last:.2f}  RMSE {rmse_last:.3f} px")
    print(f"Reduction: {(1 - sq_last / sq_first) * 100:.2f}% of squared error, "
          f"{(1 - rmse_last / rmse_first) * 100:.2f}% of pixel RMSE")


if __name__ == "__main__":
    main()
