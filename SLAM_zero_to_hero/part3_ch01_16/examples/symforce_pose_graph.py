#!/usr/bin/env python3
"""SymForce Tutorial: 2D Pose-Graph Optimization (PGO)

A robot drives a square loop and returns to its start. We define a between
residual on sf.Pose2, build odometry + one loop closure from ground truth, and
optimize from a noisy initial estimate with Levenberg-Marquardt.

Every iteration is streamed live to a rerun viewer (poses, headings, loop
closure, chi-squared) - see examples/rerun_viz.py. The run also prints its
numbers, so it is useful with no viewer at all.

Shared problem definition (identical in the g2o / GTSAM / Ceres chapters):
  5 ground-truth poses (0,0,0) (1,0,0) (1,1,pi/2) (0,1,pi) (0,0,-pi/2),
  odometry edges 0-1, 1-2, 2-3, 3-4 plus the 4-0 loop closure,
  measurements are the exact relative transforms of ground truth (deliberately
  noise-free: only the initial estimate is perturbed, so the optimum is exactly
  ground truth and any residual error is the solver's),
  initial estimate: seed 7, sigma_xy = 0.15, sigma_theta = 0.08 on poses 1..4
  (pose 0 starts exactly at ground truth), noise model
  sigma = (0.1, 0.1, 0.05) i.e. information diag(100, 100, 400),
  LM, max 30 iterations.

The perturbation of the initial estimate is the one thing that is not
bit-identical to the C++ chapters: numpy's default_rng(7) is a different
generator from std::mt19937(7), so the initial chi-squared is 17.218417 here
against 15.9973 there. The optimum is ground truth in both cases, because the
measurements are noise-free - see the README section "What is identical to the
C++ chapters, and what is not".
"""
import sys
from pathlib import Path

import symforce

symforce.set_epsilon_to_symbol()

import numpy as np
import symforce.symbolic as sf
from symforce.opt.factor import Factor
from symforce.opt.optimizer import Optimizer
from symforce.values import Values

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rerun_viz import LOOP, ODOMETRY, POSE_GRAPH_RECORDING, Viz  # noqa: E402

GT_XYT = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, np.pi / 2),
    (0.0, 1.0, np.pi),
    (0.0, 0.0, -np.pi / 2),
]
EDGES = [(0, 1, ODOMETRY), (1, 2, ODOMETRY), (2, 3, ODOMETRY), (3, 4, ODOMETRY),
         (4, 0, LOOP)]
SIGMA_XY, SIGMA_THETA = 0.1, 0.05  # noise model, i.e. information diag(100,100,400)
INIT_SIGMA_XY, INIT_SIGMA_THETA = 0.15, 0.08
MAX_ITERATIONS = 30


def between_residual(
    a: sf.Pose2, b: sf.Pose2, meas: sf.Pose2, weight: sf.V3, epsilon: sf.Scalar
) -> sf.V3:
    """Whitened residual of measured^{-1} * (a^{-1} * b).

    ``weight`` is 1/sigma per tangent component - SymForce has no separate
    "noise model" object, so the information matrix lives in the residual. Note
    sf.Pose2.to_tangent() orders the tangent as (theta, x, y): rotation first,
    the opposite of the (x, y, theta) order the other chapters' SE(2) types use.
    """
    predicted = a.inverse() * b
    tangent = sf.V3((meas.inverse() * predicted).to_tangent(epsilon=epsilon))
    return sf.V3(
        weight[0] * tangent[0], weight[1] * tangent[1], weight[2] * tangent[2]
    )


def pose2(x, y, th):
    return sf.Pose2(R=sf.Rot2.from_angle(th), t=sf.V2(x, y))


def xyt(p, eps):
    """(x, y, theta) of an sf.Pose2."""
    return (
        float(p.t[0]),
        float(p.t[1]),
        float(p.rotation().to_tangent(epsilon=eps)[0]),
    )


def wrap(angle):
    """Wrap to (-pi, pi]."""
    return -(np.remainder(-angle + np.pi, 2 * np.pi) - np.pi)


def relative(pi, pj):
    """Relative pose pi^{-1} * pj as (dx, dy, dtheta), plain numpy."""
    c, s = np.cos(pi[2]), np.sin(pi[2])
    dx, dy = pj[0] - pi[0], pj[1] - pi[1]
    return (c * dx + s * dy, -s * dx + c * dy, wrap(pj[2] - pi[2]))


def chi_squared(poses):
    """Shared chi-squared: per edge, the (x, y, theta) residual between the
    measured and the current relative pose, weighted by diag(100, 100, 400).

    Computed here rather than taken from the optimizer, because SymForce reports
    0.5 * sum of squares and the four chapters must plot the same number.
    """
    info = np.array([1.0 / SIGMA_XY**2, 1.0 / SIGMA_XY**2, 1.0 / SIGMA_THETA**2])
    chi2 = 0.0
    for i, j, _ in EDGES:
        meas = relative(GT_XYT[i], GT_XYT[j])  # measurements come from GT
        cur = relative(poses[i], poses[j])
        r = np.array([cur[0] - meas[0], cur[1] - meas[1], wrap(cur[2] - meas[2])])
        chi2 += float(np.dot(info * r, r))
    return chi2


def main():
    print("=== SymForce Tutorial: 2D Pose-Graph Optimization ===\n")

    eps = sf.numeric_epsilon
    gt = [pose2(*p) for p in GT_XYT]
    n = len(gt)

    # Perturb poses 1..4 only. Pose 0 is the gauge anchor and stays exactly at
    # ground truth in the solver state and in everything logged.
    rng = np.random.default_rng(7)
    init_xyt = [GT_XYT[0]]
    for x, y, th in GT_XYT[1:]:
        init_xyt.append((
            x + rng.normal(0, INIT_SIGMA_XY),
            y + rng.normal(0, INIT_SIGMA_XY),
            th + rng.normal(0, INIT_SIGMA_THETA),
        ))

    viz = Viz(POSE_GRAPH_RECORDING, "symforce")
    viz.pose_graph_setup(GT_XYT, init_xyt, EDGES)

    values = Values()
    values["epsilon"] = eps
    values["weight"] = sf.V3(1.0 / SIGMA_THETA, 1.0 / SIGMA_XY, 1.0 / SIGMA_XY)
    for i in range(n):
        values[f"pose{i}"] = pose2(*init_xyt[i])
    for k, (i, j, _) in enumerate(EDGES):
        values[f"meas{k}"] = gt[i].inverse() * gt[j]

    factors = [
        Factor(
            residual=between_residual,
            keys=[f"pose{i}", f"pose{j}", f"meas{k}", "weight", "epsilon"],
        )
        for k, (i, j, _) in enumerate(EDGES)
    ]

    # Gauge freedom: SymForce anchors a pose by simply leaving its key out of
    # optimized_keys - no prior factor, no "fix" call. g2o uses
    # setFixed(true), Ceres SetParameterBlockConstant, GTSAM a tight prior.
    optimized = [f"pose{i}" for i in range(1, n)]
    params = Optimizer.Params(
        iterations=MAX_ITERATIONS,
        debug_stats=True,
        verbose=False,
        early_exit_min_reduction=1e-6,
    )
    optimizer = Optimizer(factors=factors, optimized_keys=optimized, params=params)
    print("Optimizing (the first run compiles the symbolic residual)...")
    result = optimizer.optimize(values)

    # result.iterations[0] is the initial state, not a step (SymForce numbers it
    # -1); it is streamed as frame 0 so the timeline matches the sibling
    # chapters. The anchored pose 0 is constant, so it is absent from the
    # per-iteration Values.
    for step, it in enumerate(result.iterations):
        v = optimizer.load_iteration_values(it.values)
        poses = [GT_XYT[0]] + [xyt(v[f"pose{i}"], eps) for i in range(1, n)]
        viz.pose_graph_iteration(step, poses, chi_squared(poses), EDGES)

    ov = result.optimized_values
    opt_xyt = [GT_XYT[0]] + [xyt(ov[f"pose{i}"], eps) for i in range(1, n)]

    print(f"\nSolver: Levenberg-Marquardt, {len(result.iterations) - 1} iterations "
          f"(max {MAX_ITERATIONS}), status {result.status.name}")
    print(f"Initial chi2: {chi_squared(init_xyt):.6f}")
    print(f"Final chi2:   {chi_squared(opt_xyt):.3e}   (measurements are exact, so "
          f"the optimum is ground truth)")

    print("\nPose | ground truth        | initial             | optimized           | error")
    print("-" * 85)
    for i in range(n):
        gx, gy, gth = GT_XYT[i]
        ix, iy, ith = init_xyt[i]
        ox, oy, oth = opt_xyt[i]
        err = np.hypot(ox - gx, oy - gy)
        print(f"  x{i} | ({gx:5.2f},{gy:5.2f},{gth:5.2f}) | "
              f"({ix:5.2f},{iy:5.2f},{ith:5.2f}) | "
              f"({ox:5.2f},{oy:5.2f},{oth:5.2f}) | {err:.4f}")

    init_rmse = np.sqrt(np.mean([
        (i[0] - g[0]) ** 2 + (i[1] - g[1]) ** 2 for i, g in zip(init_xyt, GT_XYT)
    ]))
    opt_rmse = np.sqrt(np.mean([
        (o[0] - g[0]) ** 2 + (o[1] - g[1]) ** 2 for o, g in zip(opt_xyt, GT_XYT)
    ]))
    print(f"\nPosition RMSE vs ground truth: {init_rmse:.4f} m -> {opt_rmse:.2e} m")


if __name__ == "__main__":
    main()
