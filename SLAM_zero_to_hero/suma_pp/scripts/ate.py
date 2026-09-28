#!/usr/bin/env python3
"""Compare a SuMa++ KITTI-format pose file with the KITTI ground truth.

    python3 scripts/ate.py results/kitti00/poses.txt ~/data/kitti_vo_slam/extracted/dataset/poses/00.txt

Both files are 3x4 row-major poses in the KITTI left-camera frame, one line
per scan, starting at identity. The estimate may be shorter than the GT (a
partial run); only the first N GT poses are used. Prints path lengths, the
unaligned per-frame translation error (both start at identity, so no
alignment is applied), and the error after a rigid Umeyama/Horn alignment.
"""
import sys

import numpy as np


def load(path):
    m = np.loadtxt(path, dtype=np.float64)
    return m.reshape(-1, 3, 4)


def path_length(t):
    return float(np.linalg.norm(np.diff(t, axis=0), axis=1).sum())


def align_rigid(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    return (R @ src.T).T + (mu_d - R @ mu_s)


def stats(e):
    return f"mean {e.mean():.3f} / rmse {np.sqrt((e ** 2).mean()):.3f} / max {e.max():.3f} m"


def main():
    est, gt = load(sys.argv[1]), load(sys.argv[2])
    n = len(est)
    gt = gt[:n]
    te, tg = est[:, :, 3], gt[:, :, 3]
    print(f"poses: {n}")
    print(f"path length: est {path_length(te):.2f} m, GT {path_length(tg):.2f} m")
    print("ATE unaligned:", stats(np.linalg.norm(te - tg, axis=1)))
    print("ATE rigid-aligned:", stats(np.linalg.norm(align_rigid(te, tg) - tg, axis=1)))
    print(f"end-point error: {np.linalg.norm(te[-1] - tg[-1]):.3f} m")


if __name__ == "__main__":
    main()
