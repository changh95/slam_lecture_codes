#!/usr/bin/env python3
"""ATE of a TUM-format trajectory against TUM ground truth (numpy + matplotlib only).

    python3 scripts/eval_ate.py <groundtruth.txt> <CameraTrajectory.txt> [--plot out.png] [--sim3]

Poses are matched by nearest timestamp (max 20 ms apart, as in TUM's associate.py),
then aligned with Umeyama: SE(3) by default (RGB-D / stereo are metric), Sim(3) with
--sim3 (monocular is scale-free). Prints RMS/mean/median/max ATE and path lengths.
"""
import argparse

import numpy as np


def load_tum(path):
    rows = [l.split() for l in open(path) if l.strip() and not l.startswith("#")]
    a = np.array(rows, dtype=float)
    return a[:, 0], a[:, 1:4]


def associate(t_gt, t_est, max_dt=0.02):
    idx = np.searchsorted(t_gt, t_est)
    pairs = []
    for i, t in enumerate(t_est):
        cands = [j for j in (idx[i] - 1, idx[i]) if 0 <= j < len(t_gt)]
        j = min(cands, key=lambda j: abs(t_gt[j] - t))
        if abs(t_gt[j] - t) < max_dt:
            pairs.append((j, i))
    return np.array(pairs)


def umeyama(src, dst, with_scale):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / xs.var(0).sum() if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def path_length(p):
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gt")
    ap.add_argument("est")
    ap.add_argument("--sim3", action="store_true")
    ap.add_argument("--plot")
    a = ap.parse_args()

    t_gt, p_gt = load_tum(a.gt)
    t_est, p_est = load_tum(a.est)
    m = associate(t_gt, t_est)
    g, e = p_gt[m[:, 0]], p_est[m[:, 1]]
    s, R, t = umeyama(e, g, a.sim3)
    e_al = (s * (R @ e.T)).T + t
    err = np.linalg.norm(e_al - g, axis=1)

    print(f"poses: {len(t_est)} estimated, {len(t_gt)} GT, {len(m)} matched")
    print(f"alignment: {'Sim(3)' if a.sim3 else 'SE(3)'}" + (f", scale {s:.4f}" if a.sim3 else ""))
    print(f"ATE RMS {np.sqrt((err ** 2).mean()):.4f} m | mean {err.mean():.4f} | "
          f"median {np.median(err):.4f} | max {err.max():.4f}")
    print(f"path length: estimated {path_length(p_est) * s:.3f} m, GT (matched span) {path_length(g):.3f} m")
    print(f"duration: {t_est[-1] - t_est[0]:.2f} s")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
        ax.plot(g[:, 0], g[:, 1], "k--", lw=1.2, label="ground truth")
        ax.plot(e_al[:, 0], e_al[:, 1], color="#1f77b4", lw=1.5, label="ORB-SLAM2 (aligned)")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", "datalim")
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_title(f"ATE RMS {np.sqrt((err ** 2).mean()) * 100:.1f} cm ({len(m)} poses)")
        fig.tight_layout()
        fig.savefig(a.plot)
        print(f"plot: {a.plot}")


if __name__ == "__main__":
    main()
