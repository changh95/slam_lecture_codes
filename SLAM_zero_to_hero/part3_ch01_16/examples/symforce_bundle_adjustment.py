#!/usr/bin/env python3
"""SymForce Tutorial: Bundle Adjustment with the BAL dataset

Loads a BAL problem, builds a symbolic reprojection residual (the BAL camera
model: angle-axis rotation + translation + f, k1, k2), and jointly optimizes
camera poses and 3D points with SymForce's optimizer. Dumps
`bundle_adjustment.txt` for viz/show_bundle_adjustment.py (rerun 3D).

Because the SymForce optimizer builds one Python factor per observation, the
full Trafalgar problem (problem-21-11315, 36455 observations) is subsampled to keep the demo
fast: by default the first MAX_CAM cameras and up to MAX_OBS of their
observations. Override:  symforce_bundle_adjustment.py <file> <MAX_CAM> <MAX_OBS>
"""
import sys

import symforce

symforce.set_epsilon_to_symbol()

import numpy as np
import symforce.symbolic as sf
from symforce.opt.factor import Factor
from symforce.opt.optimizer import Optimizer
from symforce.values import Values


def reprojection_residual(
    cam_pose: sf.Pose3, calib: sf.V3, point: sf.V3, pixel: sf.V2, epsilon: sf.Scalar
) -> sf.V2:
    """BAL reprojection: p' = f (1 + k1 r^2 + k2 r^4) (-P_cam / P_cam.z)."""
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
    nC, nP, nO = int(next(it)), int(next(it)), int(next(it))
    obs = [(int(next(it)), int(next(it)), float(next(it)), float(next(it)))
           for _ in range(nO)]
    cams = [[float(next(it)) for _ in range(9)] for _ in range(nC)]
    pts = [[float(next(it)) for _ in range(3)] for _ in range(nP)]
    return nC, nP, obs, cams, pts


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "problem-21-11315-pre.txt"
    MAX_CAM = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    MAX_OBS = int(sys.argv[3]) if len(sys.argv) > 3 else 4000

    print("=== SymForce Tutorial: Bundle Adjustment (BAL) ===\n")
    nC, nP, obs, cams, pts = load_bal(path)

    sub = [o for o in obs if o[0] < MAX_CAM][:MAX_OBS]
    used_c = sorted({o[0] for o in sub})
    used_p = sorted({o[1] for o in sub})
    print(f"Full problem : {nC} cameras, {nP} points, {len(obs)} observations")
    print(f"Subsampled   : {len(used_c)} cameras, {len(used_p)} points, {len(sub)} observations")

    eps = sf.numeric_epsilon
    values = Values()
    values["epsilon"] = eps

    init_center = {}
    for c in used_c:
        aa = np.array(cams[c][0:3])
        t = np.array(cams[c][3:6])
        R = sf.Rot3.from_tangent(sf.V3(*aa), epsilon=eps)
        pose = sf.Pose3(R=R, t=sf.V3(*t))
        values[f"cam{c}"] = pose
        values[f"calib{c}"] = sf.V3(cams[c][6], cams[c][7], cams[c][8])
        init_center[c] = np.array(pose.inverse().position()).flatten()

    init_pt = {}
    for p in used_p:
        values[f"pt{p}"] = sf.V3(*pts[p])
        init_pt[p] = np.array(pts[p])

    for i, (c, p, u, v) in enumerate(sub):
        values[f"px{i}"] = sf.V2(u, v)

    factors = [
        Factor(residual=reprojection_residual,
               keys=[f"cam{c}", f"calib{c}", f"pt{p}", f"px{i}", "epsilon"])
        for i, (c, p, u, v) in enumerate(sub)
    ]

    anchor = used_c[0]  # fix first camera to remove gauge freedom
    optimized = [f"cam{c}" for c in used_c if c != anchor] + [f"pt{p}" for p in used_p]

    print("Optimizing (first run compiles the residual; please wait)...")
    # debug_stats=True keeps the Values at every iteration (for the animation);
    # verbose=False silences the per-iteration LM logging.
    params = Optimizer.Params(debug_stats=True, verbose=False)
    optimizer = Optimizer(factors=factors, optimized_keys=optimized, params=params)
    result = optimizer.optimize(values)

    n_res = 2 * len(sub)
    init_err = result.iterations[0].new_error
    final_err = result.error()
    print(f"Initial error: {init_err:.2f}  (RMSE {np.sqrt(2 * init_err / n_res):.3f} px)")
    print(f"Final error:   {final_err:.2f}  (RMSE {np.sqrt(2 * final_err / n_res):.3f} px)")
    print(f"Improvement: {(1 - final_err / init_err) * 100:.2f}%")

    # Capture the structure at every optimizer iteration. SymForce keeps the
    # Values for each iteration in result.iterations (index 0 = initial state).
    # The anchored camera is constant, so it is not in the per-iteration Values;
    # fall back to its fixed center for those.
    anchor_center = np.array(values[f"cam{anchor}"].inverse().position()).flatten()

    def cam_center(v, c):
        if c == anchor:
            return anchor_center
        return np.array(v[f"cam{c}"].inverse().position()).flatten()

    frames = []
    for it in result.iterations:
        v = optimizer.load_iteration_values(it.values)  # values_t -> keyed Values
        pts_f = [np.array(v[f"pt{p}"]).flatten() for p in used_p]
        cams_f = [cam_center(v, c) for c in used_c]
        frames.append((pts_f, cams_f))

    with open("bundle_adjustment.txt", "w") as f:
        f.write(f"points {len(used_p)}\n")
        f.write(f"cameras {len(used_c)}\n")
        f.write(f"steps {len(frames)}\n")
        for k, (pts_f, cams_f) in enumerate(frames):
            f.write(f"step {k}\n")
            for p in pts_f:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
            for c in cams_f:
                f.write(f"{c[0]} {c[1]} {c[2]}\n")
    print(f"\nWrote bundle_adjustment.txt ({len(frames)} iterations) -> "
          f"visualize with viz/show_bundle_adjustment.py")


if __name__ == "__main__":
    main()
