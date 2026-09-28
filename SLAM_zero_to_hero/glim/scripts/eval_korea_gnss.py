#!/usr/bin/env python3
"""ATE of a GLIM dump against the Korea_drive GNSS track (/surf/oxts/gnss/fix).

Reads NavSatFix straight from the bag's sqlite3 file (no ROS needed), converts
WGS84 -> ECEF -> ENU about the first fix, associates each GLIM pose with the
GNSS fix nearest in time, and aligns with a rigid Kabsch fit (rotation +
translation, scale fixed at 1). Reports 3D and horizontal (2D) ATE and writes a
top-down plot next to the dump.

usage: eval_korea_gnss.py <dump_dir> <bag_dir> [out.png]
"""
import glob
import os
import sqlite3
import struct
import sys

import numpy as np


def load_gnss(bag_dir, cache):
    if os.path.exists(cache):
        return np.load(cache)
    db = glob.glob(os.path.join(bag_dir, "*.db3"))[0]
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    (tid,) = con.execute("select id from topics where name='/surf/oxts/gnss/fix'").fetchone()
    rows = []
    for (data,) in con.execute("select data from messages where topic_id=?", (tid,)):
        b = data[4:]  # skip CDR encapsulation header
        sec, nsec, n = struct.unpack_from("<iII", b, 0)
        o = (12 + n + 3) // 4 * 4
        status = struct.unpack_from("<b", b, o)[0]
        o = (o + 4 + 7) // 8 * 8
        lat, lon, alt = struct.unpack_from("<3d", b, o)
        if status >= 0:
            rows.append((sec + nsec * 1e-9, lat, lon, alt))
    arr = np.array(rows)
    np.save(cache, arr)
    return arr


def lla_to_enu(lla):
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    lat, lon, h = np.radians(lla[:, 0]), np.radians(lla[:, 1]), lla[:, 2]
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    ecef = np.stack([(N + h) * np.cos(lat) * np.cos(lon),
                     (N + h) * np.cos(lat) * np.sin(lon),
                     (N * (1 - e2) + h) * np.sin(lat)], axis=1)
    la, lo = lat[0], lon[0]
    R = np.array([[-np.sin(lo), np.cos(lo), 0],
                  [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                  [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])
    return (ecef - ecef[0]) @ R.T


def kabsch(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - ms).T @ (dst - md))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, md - R @ ms


def path_len(p):
    return np.linalg.norm(np.diff(p, axis=0), axis=1).sum()


def main():
    dump, bag = sys.argv[1], sys.argv[2]
    out_png = sys.argv[3] if len(sys.argv) > 3 else os.path.join(dump, "trajectory_vs_gnss.png")
    gnss = load_gnss(bag, os.path.join(os.path.dirname(os.path.abspath(dump)), "korea_gnss_fix.npy"))
    g_t, g_xyz = gnss[:, 0], lla_to_enu(gnss[:, 1:4])

    for name in ("odom_lidar.txt", "traj_lidar.txt"):
        tr = np.loadtxt(os.path.join(dump, name))
        t, xyz = tr[:, 0], tr[:, 1:4]
        idx = np.clip(np.searchsorted(g_t, t), 1, len(g_t) - 1)
        idx -= (t - g_t[idx - 1]) < (g_t[idx] - t)
        ok = np.abs(g_t[idx] - t) < 0.05
        est, ref = xyz[ok], g_xyz[idx[ok]]
        R, tv = kabsch(est, ref)
        al = est @ R.T + tv
        e3 = np.linalg.norm(al - ref, axis=1)
        e2 = np.linalg.norm((al - ref)[:, :2], axis=1)
        print(f"{name}: poses {len(xyz)} (matched {ok.sum()}), duration {t[-1] - t[0]:.1f} s")
        print(f"  path {path_len(xyz):.1f} m 3D / {path_len(xyz[:, :2]):.1f} m 2D | "
              f"GNSS {path_len(ref):.1f} m 3D / {path_len(ref[:, :2]):.1f} m 2D")
        print(f"  ATE rmse {np.sqrt((e3 ** 2).mean()):.2f} m 3D, {np.sqrt((e2 ** 2).mean()):.2f} m 2D, "
              f"max {e3.max():.1f} m | start-end gap SLAM {np.linalg.norm(xyz[-1] - xyz[0]):.2f} m, "
              f"GNSS {np.linalg.norm(ref[-1] - ref[0]):.2f} m | NaN {np.isnan(xyz).sum()}")
        if name == "traj_lidar.txt":
            best = (al, ref, np.sqrt((e2 ** 2).mean()))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        al, ref, r2 = best
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(ref[:, 0], ref[:, 1], color="0.6", lw=3, label="GNSS (OXTS)")
        ax.plot(al[:, 0], al[:, 1], color="tab:blue", lw=1.2, label=f"GLIM traj_lidar (ATE 2D {r2:.2f} m)")
        ax.plot(*ref[0, :2], "go", label="start")
        ax.set_aspect("equal")
        ax.set_xlabel("east [m]")
        ax.set_ylabel("north [m]")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_title("GLIM on Korea_drive vs GNSS (rigid-aligned)")
        fig.tight_layout()
        fig.savefig(out_png, dpi=110)
        print("wrote", out_png)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
