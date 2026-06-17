#!/usr/bin/env python3
"""SymForce Tutorial: 2D Pose-Graph Optimization (PGO)

A robot drives a square loop. We define prior and between residuals on sf.Pose2,
build consistent odometry + one loop closure from ground truth, and optimize from
a noisy initial estimate. Dumps `pose_graph.txt` for viz/plot_pose_graph.py.
"""
import symforce

symforce.set_epsilon_to_symbol()

import numpy as np
import symforce.symbolic as sf
from symforce.opt.factor import Factor
from symforce.opt.optimizer import Optimizer
from symforce.values import Values


def prior_residual(pose: sf.Pose2, prior: sf.Pose2, epsilon: sf.Scalar) -> sf.V3:
    return sf.V3((prior.inverse() * pose).to_tangent(epsilon=epsilon))


def between_residual(
    a: sf.Pose2, b: sf.Pose2, meas: sf.Pose2, epsilon: sf.Scalar
) -> sf.V3:
    """Residual of measured^{-1} * (a^{-1} * b)."""
    predicted = a.inverse() * b
    return sf.V3((meas.inverse() * predicted).to_tangent(epsilon=epsilon))


def pose2(x, y, th):
    return sf.Pose2(R=sf.Rot2.from_angle(th), t=sf.V2(x, y))


def xyt(p, eps):
    return float(p.t[0]), float(p.t[1]), float(p.rotation().to_tangent(epsilon=eps)[0])


def main():
    print("=== SymForce Tutorial: 2D Pose-Graph Optimization ===\n")

    gt_xyt = [(0, 0, 0), (1, 0, 0), (1, 1, np.pi / 2), (0, 1, np.pi), (0, 0, -np.pi / 2)]
    gt = [pose2(*p) for p in gt_xyt]
    edges = [(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 4, 0), (4, 0, 1)]  # type 1 = loop
    N = len(gt)
    eps = sf.numeric_epsilon

    rng = np.random.default_rng(7)
    init_xyt, init = [], []
    for i, (x, y, th) in enumerate(gt_xyt):
        if i == 0:
            ix, iy, ith = x, y, th
        else:
            ix = x + rng.normal(0, 0.15)
            iy = y + rng.normal(0, 0.15)
            ith = th + rng.normal(0, 0.08)
        init_xyt.append((ix, iy, ith))
        init.append(pose2(ix, iy, ith))

    values = Values()
    values["epsilon"] = eps
    for i in range(N):
        values[f"pose{i}"] = init[i]
    values["prior0"] = gt[0]
    for k, (i, j, _) in enumerate(edges):
        values[f"meas{k}"] = gt[i].inverse() * gt[j]

    factors = [Factor(residual=prior_residual, keys=["pose0", "prior0", "epsilon"])]
    for k, (i, j, _) in enumerate(edges):
        factors.append(
            Factor(residual=between_residual,
                   keys=[f"pose{i}", f"pose{j}", f"meas{k}", "epsilon"])
        )

    optimizer = Optimizer(factors=factors, optimized_keys=[f"pose{i}" for i in range(N)])
    result = optimizer.optimize(values)
    ov = result.optimized_values
    print(f"Initial error: {result.iterations[0].new_error:.6f}  "
          f"Final error: {result.error():.6f}")

    opt_xyt = [xyt(ov[f"pose{i}"], eps) for i in range(N)]
    print("\nPose | ground truth        | optimized           | error")
    print("-" * 63)
    for i in range(N):
        gx, gy, gth = gt_xyt[i]
        ox, oy, oth = opt_xyt[i]
        err = np.hypot(ox - gx, oy - gy)
        print(f"  x{i} | ({gx:5.2f},{gy:5.2f},{gth:5.2f}) | "
              f"({ox:5.2f},{oy:5.2f},{oth:5.2f}) | {err:.4f}")

    with open("pose_graph.txt", "w") as f:
        f.write(f"nodes {N}\n")
        for i in range(N):
            g = gt_xyt[i]
            ii = init_xyt[i]
            o = opt_xyt[i]
            f.write(f"{i} {g[0]} {g[1]} {g[2]} {ii[0]} {ii[1]} {ii[2]} "
                    f"{o[0]} {o[1]} {o[2]}\n")
        f.write(f"edges {len(edges)}\n")
        for i, j, t in edges:
            f.write(f"{i} {j} {t}\n")
    print("\nWrote pose_graph.txt -> visualize with viz/plot_pose_graph.py")


if __name__ == "__main__":
    main()
