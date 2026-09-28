#!/usr/bin/env python3
"""Score a DSP-SLAM KITTI trajectory against KITTI ground truth.

    python3 eval_kitti.py CameraTrajectory.txt 07.txt [--plot out.png]

Both files are KITTI format (12 numbers per line, one line per frame). The
estimate is aligned to the ground truth with a closed-form SE(3) Umeyama fit
(stereo, so no scale), then the RMS absolute translation error is reported,
together with path length and the objects found in MapObjects.txt.
"""
import argparse
import os

import numpy as np


def load_kitti(path):
    m = np.loadtxt(path).reshape(-1, 3, 4)
    return m[:, :, 3]


def umeyama_se3(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    cov = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, _, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    return R, mu_d - R @ mu_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("estimate")
    ap.add_argument("groundtruth")
    ap.add_argument("--plot", default=None)
    a = ap.parse_args()

    est, gt = load_kitti(a.estimate), load_kitti(a.groundtruth)
    n = min(len(est), len(gt))
    est, gt = est[:n], gt[:n]
    R, t = umeyama_se3(est, gt)
    est_al = est @ R.T + t
    err = np.linalg.norm(est_al - gt, axis=1)
    path = np.linalg.norm(np.diff(gt, axis=0), axis=1).sum()
    path_est = np.linalg.norm(np.diff(est, axis=0), axis=1).sum()
    print(f"poses            : {n}")
    print(f"GT path length   : {path:.1f} m   (estimate {path_est:.1f} m)")
    print(f"ATE RMSE (SE3)   : {np.sqrt((err ** 2).mean()):.3f} m")
    print(f"ATE mean / max   : {err.mean():.3f} / {err.max():.3f} m")
    print(f"ATE / path       : {100 * np.sqrt((err ** 2).mean()) / path:.2f} %")

    obj_file = os.path.join(os.path.dirname(a.estimate), "MapObjects.txt")
    if os.path.exists(obj_file):
        with open(obj_file) as f:
            # 3 lines per object: id, 3x4 Sim(3) pose, 64-d DeepSDF code
            print(f"map objects      : {sum(1 for l in f if l.strip()) // 3}")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(gt[:, 0], gt[:, 2], "k-", lw=2, label="KITTI GT")
        ax.plot(est_al[:, 0], est_al[:, 2], "r-", lw=1.2, label="DSP-SLAM (SE3-aligned)")
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        ax.set_title(f"KITTI 07 - ATE RMSE {np.sqrt((err ** 2).mean()):.2f} m over {path:.0f} m")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=120)
        print(f"plot             : {a.plot}")


if __name__ == "__main__":
    main()
