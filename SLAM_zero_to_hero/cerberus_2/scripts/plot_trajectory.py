#!/usr/bin/env python3
"""Plot the trajectory CSVs Cerberus 2.0 writes, and print drift numbers.

Every estimator variant appends one CSV under `output_path` with a fixed 10-column
layout (src/fusion/VILOFusion.cpp, VILOLoop):

    time, px, py, pz, roll, pitch, yaw, vx, vy, vz

and the *file name* says which variant produced it, because parameters.cpp
derives it from kf_type / vilo_fusion_type:

    vilo-m-<dataset>.csv    stereo VIO + multi-IMU leg-odometry velocity  <- Cerberus 2.0
    vilo-s-<dataset>.csv    same, single-IMU proprioceptive odometry
    vilo-tm-{n,y}-<ds>.csv  tightly-coupled leg factor, kinematics off/on
    vio-<dataset>.csv       stereo VIO only, legs unused
    mipo-<dataset>.csv      multi-IMU proprioceptive odometry only (no camera)
    sipo-<dataset>.csv      single-IMU proprioceptive odometry only
    gt-<dataset>.csv        mocap, indoor sequences only -- empty outdoors

No ground truth is plotted for the outdoor sequences: what ships next to those
bags is a MATLAB Mobile .mat holding `timetable` objects (MCOS), which needs
MATLAB and upstream's script/matlab/mobile_gps_process/ to become a trajectory.
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# One stable colour per variant, so figures from different runs are comparable.
STYLE = {
    "vilo-m": ("#1a9850", "VILO-M (stereo + multi-IMU leg odom)"),
    "vilo-s": ("#66bd63", "VILO-S (stereo + single-IMU leg odom)"),
    "vilo-tm-n": ("#006837", "VILO-TM (tight leg factor, kin. fixed)"),
    "vilo-tm-y": ("#31a354", "VILO-TM (tight leg factor, kin. estimated)"),
    "vio": ("#3288bd", "VIO (stereo + body IMU only)"),
    "mipo": ("#f46d43", "MIPO (multi-IMU proprioception only)"),
    "sipo": ("#d73027", "SIPO (single-IMU proprioception only)"),
    "gt": ("#000000", "ground truth (mocap)"),
}


def variant_of(path):
    """vilo-m-mill19_trail.csv -> vilo-m. Longest known prefix wins."""
    base = os.path.basename(path)[:-4] if path.endswith(".csv") else os.path.basename(path)
    for key in sorted(STYLE, key=len, reverse=True):
        if base.startswith(key + "-") or base == key:
            return key
    return base


def load(path):
    a = np.loadtxt(path, delimiter=",", ndmin=2)
    if a.size == 0:
        return None
    # The estimator starts publishing (0,0,0) before its first state update, and
    # those rows would anchor every plot to the origin and inflate path length.
    # Drop the leading all-zero position block only.
    nz = np.flatnonzero(np.any(a[:, 1:4] != 0.0, axis=1))
    return a[nz[0]:] if len(nz) else None


def path_length(xyz):
    return float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))


def max_jump(xyz):
    """Largest single-sample position step, and how many exceed 25 cm.

    A trotting Go1 logged at ~15 Hz moves at most ~7 cm between samples, so this
    separates smooth drift from the visible discontinuities you get when the
    sliding window re-initialises or the solver returns garbage. A healthy run has
    one ~0.3 m step, at initialisation, and no others; a run that jumps on screen
    has dozens.
    """
    d = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    if not len(d):
        return 0.0, 0
    return float(d.max()), int((d > 0.25).sum())


# A trotting Go1 is commanded at 0.6-0.8 m/s and tops out around 1.5 m/s, so a
# mean speed above this can only be a diverged estimate. Diverged runs still get
# drawn (dashed) because seeing VIO leave the map is the point of the ablation,
# but they must not set the axis limits or a 600 km excursion compresses every
# real trajectory into a single pixel.
DIVERGED_MEAN_SPEED = 3.0  # m/s


def is_diverged(a, length):
    span = a[:, 0].ptp()
    return span > 1.0 and length / span > DIVERGED_MEAN_SPEED


def umeyama_rt(src, dst):
    """Rigid transform (R, t) minimising ||R*src + t - dst||, no scale.

    Both trajectories are metric, so scale is deliberately not solved for --
    fitting it would hide exactly the drift we want to measure. The estimator's
    "world" frame starts at the robot's initial pose while mocap uses the room
    origin, so some rigid alignment is unavoidable before an error means anything.
    """
    mu_s, mu_d = src.mean(0), dst.mean(0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, mu_d - R @ mu_s


def ate(est, gt):
    """Absolute trajectory error after rigid alignment.

    est/gt are (t, x, y, z) arrays. GT is sampled at whatever rate the mocap ran,
    so the estimate is linearly interpolated onto the GT timestamps over their
    overlapping interval.
    """
    lo = max(est[0, 0], gt[0, 0])
    hi = min(est[-1, 0], gt[-1, 0])
    g = gt[(gt[:, 0] >= lo) & (gt[:, 0] <= hi)]
    if len(g) < 20:
        return None
    e = np.column_stack([np.interp(g[:, 0], est[:, 0], est[:, k]) for k in (1, 2, 3)])
    R, t = umeyama_rt(e, g[:, 1:4])
    err = np.linalg.norm((R @ e.T).T + t - g[:, 1:4], axis=1)
    gt_len = path_length(g[:, 1:4])
    return dict(n=len(g), rmse=float(np.sqrt((err ** 2).mean())), max=float(err.max()),
                gt_path=gt_len, pct=100.0 * float(np.sqrt((err ** 2).mean())) / gt_len if gt_len else float("nan"),
                aligned=(R @ e.T).T + t, gt=g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", help="trajectory CSVs written by cerberus2_main")
    ap.add_argument("--out", default="trajectory.png")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    runs, gt = [], None
    for p in args.csv:
        a = load(p)
        if a is None:
            print(f"[plot] {os.path.basename(p)}: empty, skipped")
            continue
        if variant_of(p) == "gt":
            gt = a
            continue
        runs.append((variant_of(p), a))
    if not runs:
        sys.exit("[plot] every CSV was empty")

    title = args.title
    if title is None:
        base = os.path.basename(args.csv[0])[:-4]
        v = variant_of(args.csv[0])
        title = base[len(v) + 1:] if base.startswith(v + "-") else base

    fig = plt.figure(figsize=(15, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1])
    ax_xy, ax_z, ax_v = (fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2]))

    print(f"{'variant':<12}{'poses':>8}{'path [m]':>11}{'span xy [m]':>13}{'end-start [m]':>15}"
          f"{'z rng [m]':>11}{'max step':>10}{'>25cm':>7}  state")
    good_xy, good_z, n_diverged = [], [], 0
    for key, a in runs:
        colour, label = STYLE.get(key, ("#777777", key))
        t = a[:, 0] - a[0, 0]
        xyz = a[:, 1:4]
        L = path_length(xyz)
        closure = float(np.linalg.norm(xyz[-1] - xyz[0]))
        span = float(max(xyz[:, 0].ptp(), xyz[:, 1].ptp()))
        bad = is_diverged(a, L)
        style = dict(color=colour, lw=1.2, ls="--", alpha=0.55) if bad else dict(color=colour, lw=1.6)
        if bad:
            label += "  [DIVERGED]"

        ax_xy.plot(xyz[:, 0], xyz[:, 1], label=label, **style)
        ax_xy.plot(xyz[0, 0], xyz[0, 1], "o", color=colour, ms=7, mfc="white", mew=1.8)
        ax_xy.plot(xyz[-1, 0], xyz[-1, 1], "s", color=colour, ms=7)
        ax_z.plot(t, xyz[:, 2], **style)
        ax_v.plot(t, np.linalg.norm(a[:, 7:10], axis=1), **{**style, "lw": 1.0})
        if bad:
            n_diverged += 1
        else:
            good_xy.append(xyz[:, :2])
            good_z.append(xyz[:, 2])

        mj, njump = max_jump(xyz)
        print(f"{key:<12}{len(a):>8}{L:>11.1f}{span:>13.1f}{closure:>15.2f}{xyz[:, 2].ptp():>11.2f}"
              f"{mj:>10.2f}{njump:>7d}  {'DIVERGED' if bad else 'ok'}")

    ax_xy.set_title(f"{title} — top-down (x/y)")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.axis("equal")
    ax_xy.grid(alpha=0.3)
    ax_xy.legend(fontsize=8, loc="best")

    ax_z.set_title("height")
    ax_z.set_xlabel("t [s]")
    ax_z.set_ylabel("z [m]")
    ax_z.grid(alpha=0.3)

    ax_v.set_title("body speed")
    ax_v.set_xlabel("t [s]")
    ax_v.set_ylabel("|v| [m/s]")
    ax_v.grid(alpha=0.3)

    # Mocap ground truth, when the sequence has any (indoor only). Each estimate
    # is rigidly aligned to it and reported as ATE; the plotted GT is the raw
    # mocap track, and the aligned estimate is drawn faintly on top of it.
    if gt is not None:
        colour, label = STYLE["gt"]
        ax_xy.plot(gt[:, 1], gt[:, 2], color=colour, lw=2.2, alpha=0.55, label=label, zorder=0)
        ax_z.plot(gt[:, 0] - gt[0, 0], gt[:, 3], color=colour, lw=2.0, alpha=0.55, zorder=0)
        good_xy.append(gt[:, 1:3])
        good_z.append(gt[:, 3])
        print()
        print(f"{'variant':<12}{'ATE rmse [m]':>14}{'ATE max [m]':>13}{'gt path [m]':>13}{'rmse/path':>11}")
        for key, a in runs:
            m = ate(a, gt)
            if m is None:
                print(f"{key:<12}   no timestamp overlap with ground truth")
                continue
            print(f"{key:<12}{m['rmse']:>14.3f}{m['max']:>13.3f}{m['gt_path']:>13.1f}{m['pct']:>10.2f}%")
            ax_xy.plot(m["aligned"][:, 0], m["aligned"][:, 1], color=STYLE.get(key, ("#777", ""))[0],
                       lw=1.0, ls=":", alpha=0.9)
        ax_xy.legend(fontsize=8, loc="best")

    # Frame on the runs that stayed sane, with 10 % padding. Done after
    # axis("equal") so the equal aspect is preserved inside these limits.
    if good_xy and n_diverged:
        P = np.vstack(good_xy)
        for lim, col in ((ax_xy.set_xlim, 0), (ax_xy.set_ylim, 1)):
            lo, hi = P[:, col].min(), P[:, col].max()
            pad = max(0.1 * (hi - lo), 1.0)
            lim(lo - pad, hi + pad)
        Z = np.concatenate(good_z)
        pad = max(0.1 * (Z.max() - Z.min()), 0.5)
        ax_z.set_ylim(Z.min() - pad, Z.max() + pad)
        ax_v.set_ylim(0, 2.0)

    fig.savefig(args.out, dpi=130)
    print(f"[plot] wrote {args.out}")


if __name__ == "__main__":
    main()
