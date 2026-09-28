#!/usr/bin/env python3
"""Top-down overview of a MASt3R-SLAM run: the saved reconstruction (.ply) and the
keyframe trajectory, Sim(3)-aligned to TUM ground truth with evo (same alignment as
`evo_ape -as`).

usage: plot_overview.py <groundtruth.txt> <seq>.txt <seq>.ply <out.png>
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from evo.core import sync
from evo.tools import file_interface
from plyfile import PlyData

gt_file, est_file, ply_file, out_png = sys.argv[1:5]

ref = file_interface.read_tum_trajectory_file(gt_file)
est = file_interface.read_tum_trajectory_file(est_file)
ref, est = sync.associate_trajectories(ref, est, max_diff=0.02)
r, t, s = est.align(ref, correct_scale=True)

v = PlyData.read(ply_file)["vertex"].data
xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1) / 255.0
rng = np.random.default_rng(0)
keep = rng.choice(len(xyz), size=min(len(xyz), 400_000), replace=False)
xyz, rgb = xyz[keep], rgb[keep]
xyz = s * xyz @ r.T + t  # same Sim(3) as the trajectory

# drop the floor/ceiling extremes so the top view shows furniture and walls
z_lo, z_hi = np.percentile(xyz[:, 2], [2, 98])
m = (xyz[:, 2] > z_lo) & (xyz[:, 2] < z_hi)
order = np.argsort(xyz[m, 2])  # draw higher points last

p_ref, p_est = ref.positions_xyz, est.positions_xyz
err = np.linalg.norm(p_ref - p_est, axis=1)
fig, ax = plt.subplots(figsize=(9, 8), dpi=130)
ax.scatter(xyz[m][order, 0], xyz[m][order, 1], c=rgb[m][order], s=0.3, linewidths=0)
ax.plot(p_ref[:, 0], p_ref[:, 1], "-", color="0.15", lw=1.5, label="ground truth (keyframe stamps)")
ax.plot(p_est[:, 0], p_est[:, 1], "o-", color="#d62728", ms=3, lw=1.2,
        label=f"MASt3R-SLAM keyframes (n={len(p_est)})")
ax.set_aspect("equal")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title(f"TUM fr1_room — reconstruction + keyframes, Sim(3)-aligned\n"
             f"ATE RMSE {np.sqrt(np.mean(err ** 2)):.3f} m, scale {s:.3f}")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(out_png)
print(f"wrote {out_png}: {len(p_est)} keyframes, RMSE {np.sqrt(np.mean(err ** 2)):.4f} m, scale {s:.4f}")
