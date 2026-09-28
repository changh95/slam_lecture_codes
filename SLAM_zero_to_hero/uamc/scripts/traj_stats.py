#!/usr/bin/env python3
"""Summarise a FAST-LIVO2 TUM trajectory: pose count, duration, path length, start-end gap."""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("traj")
ap.add_argument("--wall", type=float, default=None, help="wall-clock seconds of bag playback")
a = ap.parse_args()

d = np.loadtxt(a.traj, ndmin=2)
t, p = d[:, 0], d[:, 1:4]
step = np.linalg.norm(np.diff(p, axis=0), axis=1)
length = step.sum()
gap = np.linalg.norm(p[-1] - p[0])
print(f"poses          {len(d)}")
print(f"duration       {t[-1] - t[0]:.1f} s ({len(d) / max(t[-1] - t[0], 1e-9):.2f} Hz)")
print(f"path length    {length:.1f} m")
print(f"extent xyz     {np.ptp(p[:, 0]):.1f} x {np.ptp(p[:, 1]):.1f} x {np.ptp(p[:, 2]):.1f} m")
print(f"start-end gap  {gap:.2f} m ({100 * gap / max(length, 1e-9):.2f} % of path)")
print(f"max step       {step.max():.2f} m")
if a.wall:
    print(f"wall clock     {a.wall:.0f} s")
