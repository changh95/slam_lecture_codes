#!/usr/bin/env python3
"""Render the final MonoGS Gaussian map from estimated keyframe poses.

Writes a grid (top row: input frame, bottom row: rendered Gaussians) and a
3D trajectory plot (estimate vs ground truth, keyframes only).

    python3 /opt/scripts/render_views.py <run_dir> [--n 4] [--out <dir>]

<run_dir> is one MonoGS results folder, e.g.
results/datasets_tum/2026-09-28-00-30-00 (it holds config.yml, plot/ and
point_cloud/).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "/MonoGS")
os.chdir("/MonoGS")

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from munch import munchify  # noqa: E402

from gaussian_splatting.gaussian_renderer import render  # noqa: E402
from gaussian_splatting.scene.gaussian_model import GaussianModel  # noqa: E402
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2  # noqa: E402
from utils.camera_utils import Camera  # noqa: E402
from utils.dataset import load_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--n", type=int, default=4, help="number of keyframes to render")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run = os.path.abspath(a.run_dir)
    out = a.out or run

    cfg = yaml.full_load(open(os.path.join(run, "config.yml")))
    trj = json.load(open(os.path.join(run, "plot", "trj_final.json")))
    ply = os.path.join(run, "point_cloud", "final_after_opt", "point_cloud.ply")
    if not os.path.exists(ply):
        ply = os.path.join(run, "point_cloud", "final", "point_cloud.ply")

    mp = munchify(cfg["model_params"])
    g = GaussianModel(mp.sh_degree, config=cfg)
    g.load_ply(ply)
    print(f"[render_views] {g.get_xyz.shape[0]} Gaussians from {ply}")
    ds = load_dataset(mp, mp.source_path, config=cfg)
    proj = getProjectionMatrix2(0.01, 100.0, ds.cx, ds.cy, ds.fx, ds.fy, ds.width, ds.height)
    proj = proj.transpose(0, 1).cuda()
    pipe = munchify(cfg["pipeline_params"])
    bg = torch.zeros(3, device="cuda")

    ids = trj["trj_id"]
    pick = np.linspace(0, len(ids) - 1, a.n).round().astype(int)
    tops, bots = [], []
    for p in pick:
        idx = ids[p]
        cam = Camera.init_from_dataset(ds, idx, proj)
        w2c = np.linalg.inv(np.array(trj["trj_est"][p]))
        cam.update_RT(torch.tensor(w2c[:3, :3]).float(), torch.tensor(w2c[:3, 3]).float())
        with torch.no_grad():
            img = render(cam, g, pipe, bg)["render"].clamp(0, 1)
        r = np.ascontiguousarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        gt = np.ascontiguousarray((cam.original_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        for im, txt in ((gt, f"input #{idx}"), (r, f"render #{idx}")):
            cv2.putText(im, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        tops.append(gt)
        bots.append(r)
    grid = np.vstack([np.hstack(tops), np.hstack(bots)])
    grid = cv2.resize(grid, (grid.shape[1] // 2, grid.shape[0] // 2), interpolation=cv2.INTER_AREA)
    f1 = os.path.join(out, "render_grid.png")
    cv2.imwrite(f1, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    est = np.array([np.array(T)[:3, 3] for T in trj["trj_est"]])
    gtp = np.array([np.array(T)[:3, 3] for T in trj["trj_gt"]])
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*gtp.T, "--", color="gray", label="ground truth")
    ax.plot(*est.T, color="tab:blue", label="MonoGS (keyframes)")
    ax.scatter(*est[0], color="green", s=30, label="start")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.legend(); ax.set_title(f"{len(ids)} keyframes (raw, not aligned)")
    f2 = os.path.join(out, "trajectory_3d.png")
    fig.savefig(f2, dpi=110, bbox_inches="tight")
    seg = np.linalg.norm(np.diff(est, axis=0), axis=1).sum()
    print(f"[render_views] keyframes={len(ids)} kf path length={seg:.3f} m")
    print(f"[render_views] wrote {f1} and {f2}")


if __name__ == "__main__":
    main()
