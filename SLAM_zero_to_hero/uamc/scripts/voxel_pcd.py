#!/usr/bin/env python3
"""Voxel-downsample FAST-LIVO2's all_raw_points.pcd into an x y z rgb map.

One point per voxel, position and colour averaged. A LiDAR-visual-inertial map keeps its
camera colours; a LiDAR-inertial map (intensity only) is coloured by height instead.

FAST-LIVO2's own pcl::VoxelGrid pass silently returns the input unchanged on a scene this
large ("Integer indices would overflow" at 0.15 m over ~500 m), so do it here with 64-bit keys.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pcd_io import DT, height_rgb, read_pcd, write_pcd  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--leaf", type=float, default=0.1)
a = ap.parse_args()

p = read_pcd(a.src)
xyz = np.stack([p["x"], p["y"], p["z"]], 1).astype(np.float64)
ok = np.isfinite(xyz).all(1)
p, xyz = p[ok], xyz[ok]
k = np.floor(xyz / a.leaf).astype(np.int64)
k -= k.min(0)
dims = k.max(0) + 1
key = (k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2]
_, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
m = len(cnt)
out = np.zeros(m, dtype=DT)
for c in "xyz":
    out[c] = np.bincount(inv, weights=xyz[:, "xyz".index(c)], minlength=m) / cnt

if "rgb" in p.dtype.names:
    chan = [(p["rgb"] >> s) & 0xFF for s in (16, 8, 0)]
    r, g, b = (np.round(np.bincount(inv, weights=ch, minlength=m) / cnt).astype(np.uint32)
               for ch in chan)
    how = "camera colours"
else:
    # Height colour ramp (blue -> cyan -> green -> yellow -> red). The range is set by the
    # 20th/99.5th percentiles: ~6-10 % of COEX points are specular ghosts tens to hundreds
    # of metres below the floor, and they would otherwise squash the ramp.
    lo, hi = np.percentile(out["z"], [20, 99.5])
    r, g, b = height_rgb(out["z"], lo, hi)
    how = f"height colours, z {lo:.1f}..{hi:.1f} m"
out["rgb"] = (r << 16) | (g << 8) | b
write_pcd(a.dst, out)
print(f"voxel {a.leaf} m: {len(p)} -> {m} points ({how}) -> {a.dst}")
