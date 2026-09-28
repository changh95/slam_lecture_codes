#!/usr/bin/env python3
"""Headless text query over a fused ConceptFusion map.

Same scoring as upstream examples/demo_text_query.py (cosine similarity between the
OpenCLIP text embedding and every map point's fused embedding, min-max normalised,
thresholded, jet colour map blended 50/50 with the RGB), but without the input()
loop and the blocking Open3D window, so it can run in a container:

  * <out>/map_rgb.png, <out>/query_<text>.png         Open3D renders (needs an X server, e.g. xvfb-run)
  * <out>/overview.png                                 the renders side by side
  * <out>/query_2d_<text>.png                          the same query on one frame's pixel-aligned features
  * <out>/query_summary.json                           per-query statistics
  * <work>/map_rgb.ply, <work>/query_<text>.ply       coloured point clouds
  * <work>/concept_fusion.rrd                          rerun recording (map, heatmaps, GT trajectory)
  * streams the same recording live to a rerun viewer on the host if one listens on --rerun-url
"""
import argparse
import json
import os
import socket
import time
from pathlib import Path
from urllib.parse import urlparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open_clip
import torch
from gradslam.structures.pointclouds import Pointclouds


def slug(text):
    return "".join(c if c.isalnum() else "_" for c in text.strip().lower())


def rerun_reachable(url, timeout=0.5):
    u = urlparse(url.replace("rerun+", ""))
    try:
        with socket.create_connection((u.hostname, u.port or 9876), timeout=timeout):
            return True
    except OSError:
        return False


def load_gt_trajectory(data_dir, sequence, end, stride):
    """Camera centres in the map frame (same pose chain gradslam used for fusion)."""
    from gradslam.datasets import ICLDataset, load_dataset_config

    cfg = load_dataset_config("/opt/concept-fusion/examples/dataconfigs/icl.yaml")
    ds = ICLDataset(cfg, data_dir, sequence, start=0, end=end, stride=1,
                    desired_height=120, desired_width=160)
    poses = ds.transformed_poses.numpy()
    return poses[:, :3, 3], poses[::stride, :3, 3]


def render(pcd, path, width=1280, height=960, traj=None):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(pcd)
    if traj is not None:
        vis.add_geometry(traj)
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([1.0, 1.0, 1.0])
    ctr = vis.get_view_control()
    # Map frame = first ICL camera (x right, y up, z backwards because fy < 0 in icl.yaml).
    # Look down at the room from above and behind the start pose.
    ctr.set_lookat(pcd.get_axis_aligned_bounding_box().get_center())
    ctr.set_front([0.0, 0.75, 0.66])
    ctr.set_up([0.0, 0.66, -0.75])
    ctr.set_zoom(0.55)
    vis.poll_events()
    vis.update_renderer()
    img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()
    plt.imsave(path, np.clip(img, 0, 1))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-path", required=True, help="dir written by run_feature_fusion_and_save_map.py")
    ap.add_argument("--queries", nargs="+", default=["sofa", "table"])
    ap.add_argument("--out-dir", required=True, help="PNGs and query_summary.json")
    ap.add_argument("--work-dir", default=None, help="PLYs and the .rrd (default: --out-dir)")
    ap.add_argument("--similarity-thresh", type=float, default=0.6)
    ap.add_argument("--feat-dir", default=None, help="saved-feat dir, for the 2D per-frame query figure")
    ap.add_argument("--data-dir", default=None, help="dataset root, for the GT trajectory and the 2D figure")
    ap.add_argument("--sequence", default="living_room_traj2_frei_png")
    ap.add_argument("--frame-end", type=int, default=880)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--frame-2d", type=int, default=0, help="frame index for the 2D figure")
    ap.add_argument("--rerun-url", default="rerun+http://127.0.0.1:9876/proxy")
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--no-render", action="store_true", help="skip the Open3D PNG renders")
    args = ap.parse_args()

    res = Path(args.out_dir)
    out = Path(args.work_dir or args.out_dir)
    res.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    torch.autograd.set_grad_enabled(False)

    pc = Pointclouds.load_pointcloud_from_h5(args.load_path)
    pts = pc.points_padded[0].float().numpy()
    rgb = pc.colors_padded[0].float().numpy()
    rgb = rgb / 255.0 if rgb.max() > 1.0 else rgb
    emb = pc.embeddings_padded[0].float().cuda()
    emb = torch.nn.functional.normalize(emb, dim=-1)
    ext = pts.max(0) - pts.min(0)
    print(f"Map: {pts.shape[0]} points, embedding dim {emb.shape[1]}, extent {ext.round(2).tolist()} m")

    t0 = time.time()
    model, _, _ = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
    model = model.cuda().eval()
    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    print(f"OpenCLIP ViT-H-14 loaded in {time.time() - t0:.1f} s")

    cmap = matplotlib.colormaps["jet"]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    o3d.io.write_point_cloud(str(out / "map_rgb.ply"), pcd)

    traj_ls = None
    traj_all = None
    if args.data_dir:
        traj_all, _ = load_gt_trajectory(args.data_dir, args.sequence, args.frame_end, args.stride)
        traj_ls = o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(traj_all.astype(np.float64)),
            o3d.utility.Vector2iVector(np.stack([np.arange(len(traj_all) - 1), np.arange(1, len(traj_all))], 1)))
        traj_ls.paint_uniform_color([0.0, 0.0, 0.0])
        steps = np.linalg.norm(np.diff(traj_all, axis=0), axis=1).sum()
        print(f"GT trajectory: {len(traj_all)} poses, path length {steps:.2f} m")

    import rerun as rr

    rr.init("concept_fusion", recording_id="concept_fusion")
    sinks = [rr.FileSink(str(out / "concept_fusion.rrd"))]
    if not args.no_stream and rerun_reachable(args.rerun_url):
        sinks.append(rr.GrpcSink(args.rerun_url))
        print(f"Streaming to rerun viewer at {args.rerun_url}")
    rr.set_sinks(*sinks)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
    rr.log("world/map_rgb", rr.Points3D(pts, colors=rgb, radii=0.01), static=True)
    if traj_all is not None:
        rr.log("world/gt_trajectory", rr.LineStrips3D([traj_all], colors=[[0, 0, 0]], radii=0.01), static=True)

    # One 3D view per layer, all looking down into the room from the same eye.
    import rerun.blueprint as rrb

    c = pts.mean(0)
    eye = rrb.EyeControls3D(kind=rrb.Eye3DKind.Orbital,
                            position=c + np.array([0.0, 0.75, -0.66]) * 0.9 * float(np.linalg.norm(ext)),
                            look_target=c, eye_up=[0.0, 1.0, 0.0])
    layers = [("RGB map", "/world/map_rgb")] + [(f'"{q}"', f"/world/query/{slug(q)}") for q in args.queries]
    rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(*[
        rrb.Spatial3DView(name=n, origin="/world", contents=[e, "/world/gt_trajectory"], eye_controls=eye)
        for n, e in layers]), collapse_panels=True))

    renders = {}
    if not args.no_render:
        renders["RGB map"] = render(pcd, res / "map_rgb.png", traj=traj_ls)

    summary = {"points": int(pts.shape[0]), "queries": {}}
    for q in args.queries:
        text = tokenizer([q]).cuda()
        tfeat = torch.nn.functional.normalize(model.encode_text(text).float(), dim=-1)
        sim = (emb @ tfeat[0])  # cosine, [-1, 1]
        rel = (sim - sim.min()) / (sim.max() - sim.min() + 1e-12)
        rel = rel.cpu().numpy()
        hot = rel >= args.similarity_thresh
        heat = rel.copy()
        heat[~hot] = 0.0
        colors = 0.5 * rgb + 0.5 * cmap(heat)[:, :3]
        qp = o3d.geometry.PointCloud(pcd.points)
        qp.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        o3d.io.write_point_cloud(str(out / f"query_{slug(q)}.ply"), qp)
        top = np.argsort(-rel)[:2000]
        centroid = pts[top].mean(0)
        summary["queries"][q] = {
            "cos_min": float(sim.min()), "cos_max": float(sim.max()),
            "points_above_thresh": int(hot.sum()), "fraction_above_thresh": float(hot.mean()),
            "top2000_centroid": centroid.round(3).tolist(),
        }
        print(f"Query '{q}': cos [{sim.min():.3f}, {sim.max():.3f}], "
              f"{hot.sum()} pts ({100 * hot.mean():.1f} %) >= {args.similarity_thresh} rel, "
              f"top-2000 centroid {centroid.round(2).tolist()}")
        rr.log(f"world/query/{slug(q)}", rr.Points3D(pts, colors=colors, radii=0.01), static=True)
        if not args.no_render:
            renders[f'"{q}"'] = render(qp, res / f"query_{slug(q)}.png", traj=traj_ls)

        # 2D: the same text query against one frame's pixel-aligned feature map
        if args.feat_dir and args.data_dir:
            fid = args.frame_2d
            f2d = torch.load(os.path.join(args.feat_dir, f"{fid}.pt")).float().cuda()  # H, W, D
            s2d = (torch.nn.functional.normalize(f2d, dim=-1) @ tfeat[0]).cpu().numpy()
            img = plt.imread(os.path.join(args.data_dir, args.sequence, "rgb", f"{fid}.png"))
            fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
            ax[0].imshow(img)
            ax[0].set_title(f"frame {fid}")
            m = ax[1].imshow(s2d, cmap="jet")
            ax[1].set_title(f'cosine sim to "{q}"')
            for a in ax:
                a.axis("off")
            fig.colorbar(m, ax=ax[1], fraction=0.035)
            fig.tight_layout()
            fig.savefig(res / f"query_2d_{slug(q)}.png", dpi=110)
            plt.close(fig)

    if renders:
        n = len(renders)
        fig, ax = plt.subplots(1, n, figsize=(6 * n, 4.8))
        for a, (title, im) in zip(np.atleast_1d(ax), renders.items()):
            a.imshow(np.clip(im, 0, 1))
            a.set_title(title, fontsize=16)
            a.axis("off")
        fig.tight_layout()
        fig.savefig(res / "overview.png", dpi=90)
        plt.close(fig)

    with open(res / "query_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote PNGs + query_summary.json to {res}, PLYs + concept_fusion.rrd to {out}")


if __name__ == "__main__":
    main()
