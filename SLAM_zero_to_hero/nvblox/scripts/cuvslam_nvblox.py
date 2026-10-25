#!/usr/bin/env python3
"""Online cuVSLAM + nvblox on Humanoid Everyday episodes: track, fuse, stream, frame by frame.

For every frame of an episode:

  1. cuVSLAM, RGB-D mode, tracks the colour camera. It needs depth registered
     to the image it tracks, so the raw depth is warped into the colour camera
     and the grid of holes that warp leaves is filled (see fill_warp_holes).
  2. nvblox (through nvblox_torch) integrates the *raw* depth in the depth
     camera's own frame -- no warp needed there -- and the colour image in the
     colour camera's, both at the pose cuVSLAM just returned.
  3. The pose, the image with the tracked features and, every few frames, the
     colour mesh are streamed to a rerun viewer.

The world frame is gravity-aligned with the floor at z = 0, anchored by the
robot IMU on the first frame exactly as he_to_nvblox.py does for depth ICP.

    cuvslam_nvblox.py <episode_dir> [<episode_dir> ...] --out /results/cuvslam

Several episodes run one after another, each with its own tracker, its own map
and its own rerun recording -- they are separate recordings with unrelated
coordinate frames, so there is nothing to join. Per episode the output folder
gets traj_cuvslam.tum, traj_odom.tum, mesh.ply and stats.json.
"""

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

# nvblox logs every hash resize and layer allocation through glog at INFO; keep warnings.
os.environ.setdefault("GLOG_minloglevel", "1")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from he_to_nvblox import (DEPTH_H, DEPTH_W, align_depth_to_colour,  # noqa: E402
                          gravity_aligned_first_pose, load_calib, matrix_from_quat_wxyz, odom_pose,
                          read_depth, read_records, transform_from_dict, write_tum)

DEFAULT_CALIB = Path(__file__).resolve().parent.parent / "config" / "humanoid_everyday_d435.json"


# --------------------------------------------------------------------------- #
# depth for the tracker
# --------------------------------------------------------------------------- #
def fill_warp_holes(depth_mm: np.ndarray, passes: int = 2) -> np.ndarray:
    """Fill the empty rows and columns the depth->colour warp leaves behind.

    The depth camera is ~79 deg wide and the colour camera ~56 deg, so forward
    projection magnifies the depth grid by ~1.57 and leaves every second or third
    row and column empty -- about 60 % of interior pixels are zero.
    he_to_nvblox.fill_pinholes only fills pixels with five valid neighbours,
    which a missing *row* never has. cuVSLAM reads depth around each feature,
    so on the raw warp it sees roughly half the true distance and its whole
    trajectory comes out at ~0.45x scale. Each pass sets a zero pixel to the
    mean of its non-zero 3x3 neighbours (when it has at least two); real
    occlusion shadows are wider than two passes and stay empty.
    """
    d = depth_mm.astype(np.float32)
    for _ in range(passes):
        nb = np.stack([np.roll(np.roll(d, dv, 0), du, 1)
                       for dv in (-1, 0, 1) for du in (-1, 0, 1) if dv or du])
        cnt = (nb > 0).sum(0)
        mean = nb.sum(0) / np.maximum(cnt, 1)
        hole = (d == 0) & (cnt >= 2)
        d[hole] = mean[hole]
    return np.rint(d).astype(np.uint16)


# --------------------------------------------------------------------------- #
# poses
# --------------------------------------------------------------------------- #
def matrix_from_quat_xyzw(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def pose_to_matrix(pose) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = matrix_from_quat_xyzw(pose.rotation)
    T[:3, 3] = pose.translation
    return T


def gravity_error_deg(T_w_c: np.ndarray, rec: dict, R_pelvis_colour: np.ndarray) -> float:
    """Angle between "down" as cuVSLAM has the camera and "down" as the robot IMU has it.

    The world frame starts gravity-aligned, so a perfect tracker keeps the two
    equal for the whole episode; the IMU's roll and pitch are gravity-referenced
    and do not drift. Yaw is unobservable this way, which is why it is a check on
    rotation drift only -- but rotation drift is how these maps fail.
    """
    down = np.array([0.0, 0.0, -1.0])
    g_vslam = T_w_c[:3, :3].T @ down
    g_imu = (matrix_from_quat_wxyz(rec["states"]["imu"]["quaternion"]) @ R_pelvis_colour).T @ down
    return float(np.degrees(np.arccos(np.clip(g_vslam @ g_imu, -1.0, 1.0))))


def verdict(max_gravity_err: float) -> str:
    if max_gravity_err < 5.0:
        return "ok"
    return "marginal" if max_gravity_err < 15.0 else "drifted"


def umeyama_se3(src: np.ndarray, dst: np.ndarray):
    """R, t minimising |R src + t - dst|. No scale: RGB-D odometry is metric."""
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - ms).T @ (dst - md))
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ S @ U.T
    return R, md - R @ ms


def path_length(p: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


# --------------------------------------------------------------------------- #
# rerun
# --------------------------------------------------------------------------- #
def viewer_reachable(url: str, timeout: float = 0.5) -> bool:
    """rerun's connect_grpc never fails on a missing viewer, it just retries, so probe first."""
    u = urlparse(url.replace("rerun+", "", 1))
    try:
        with socket.create_connection((u.hostname, u.port or 9876), timeout=timeout):
            return True
    except OSError:
        return False


class Stream:
    """Live rerun output for one episode. All methods are no-ops when disabled."""

    def __init__(self, args, name: str, colour_intr: dict):
        self.on = False
        self.images = not args.no_stream_images
        if args.no_stream and not args.save_rrd:
            return
        import rerun as rr
        import rerun.blueprint as rrb
        self.rr = rr
        rr.init("humanoid_cuvslam_nvblox", recording_id=name, spawn=False)
        if args.save_rrd:
            rr.save(str(args.save_rrd_dir / f"{name}.rrd"))
        elif viewer_reachable(args.stream_url):
            rr.connect_grpc(args.stream_url)
        else:
            print(f"[stream] no rerun viewer at {args.stream_url}; running without "
                  "streaming (start `rerun` on the host, or pass --no-stream)", flush=True)
            return
        self.on = True
        rr.send_blueprint(rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(rrb.Spatial2DView(name="colour + cuVSLAM features",
                                               origin="world/camera/image"),
                             rrb.TimeSeriesView(name="features / gravity error (deg)", origin="plots")),
                rrb.Spatial3DView(name="nvblox mesh + trajectory", origin="world"),
                column_shares=[1, 2]),
            rrb.TimePanel(state="collapsed")), make_active=True)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log("world/camera/image", rr.Pinhole(
            focal_length=[colour_intr["fx"], colour_intr["fy"]],
            principal_point=[colour_intr["cx"], colour_intr["cy"]],
            resolution=[DEPTH_W, DEPTH_H], image_plane_distance=0.2), static=True)
        self.track_colours = {}

    def frame(self, i, T_w_c, rgb, observations, traj, odom_traj, grav_err):
        if not self.on:
            return
        rr = self.rr
        rr.set_time("frame", sequence=i)
        rr.log("world/camera", rr.Transform3D(translation=T_w_c[:3, 3], mat3x3=T_w_c[:3, :3]))
        rr.log("world/trajectory/cuvslam", rr.LineStrips3D([traj], colors=[[255, 80, 40]], radii=0.006))
        if odom_traj is not None:
            rr.log("world/trajectory/leg_odometry",
                   rr.LineStrips3D([odom_traj], colors=[[60, 160, 255]], radii=0.004))
        rr.log("plots/features", rr.Scalars(len(observations)))
        rr.log("plots/gravity_error_deg", rr.Scalars(grav_err))
        if self.images:
            rr.log("world/camera/image", rr.Image(rgb).compress(jpeg_quality=80))
            if observations:
                for o in observations:
                    if o.id not in self.track_colours:
                        self.track_colours[o.id] = np.random.randint(64, 256, 3)
                rr.log("world/camera/image/features", rr.Points2D(
                    [[o.u, o.v] for o in observations], radii=3,
                    colors=[self.track_colours[o.id] for o in observations]))

    def mesh(self, mesh):
        if not self.on:
            return
        v = mesh.vertices().cpu().numpy()
        if len(v) == 0:
            return
        self.rr.log("world/mesh", self.rr.Mesh3D(
            vertex_positions=v,
            triangle_indices=mesh.triangles().cpu().numpy(),
            vertex_colors=mesh.vertex_colors().cpu().numpy()))


# --------------------------------------------------------------------------- #
# one episode
# --------------------------------------------------------------------------- #
def run_episode(episode: Path, out: Path, args) -> dict:
    import cuvslam
    import torch
    from nvblox_torch.mapper import Mapper
    from nvblox_torch.mapper_params import MapperParams, ProjectiveIntegratorParams
    from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType
    from nvblox_torch.sensor import Sensor

    name = f"{episode.parent.name}/{episode.name}"
    out.mkdir(parents=True, exist_ok=True)
    calib = load_calib(args.calib, args.robot)
    depth_intr, colour_intr = calib["depth_intrinsics"], calib["colour_intrinsics"]
    T_colour_depth = transform_from_dict(calib["depth_to_colour"])
    T_pelvis_colour = transform_from_dict(calib["pelvis_to_colour_camera"])

    recs = read_records(episode)[args.start:]
    if args.frames > 0:
        recs = recs[:args.frames]
    print(f"[{name}] {len(recs)} frames", flush=True)

    # cuVSLAM tracks the colour camera; depth is registered to it (camera 0).
    cam = cuvslam.Camera(size=(DEPTH_W, DEPTH_H),
                         principal=(colour_intr["cx"], colour_intr["cy"]),
                         focal=(colour_intr["fx"], colour_intr["fy"]))
    odom_cfg = cuvslam.Tracker.OdometryConfig(
        odometry_mode=cuvslam.Tracker.OdometryMode.RGBD,
        async_sba=False,  # deterministic: the same episode gives the same trajectory
        enable_observations_export=True,
        rgbd_settings=cuvslam.Tracker.OdometryRGBDSettings(depth_scale_factor=1000.0,
                                                           depth_camera_id=0))
    tracker = cuvslam.Tracker(cuvslam.Rig([cam]), odom_cfg,
                              cuvslam.Tracker.SlamConfig() if args.slam else None)

    params = MapperParams()
    pi = ProjectiveIntegratorParams()
    pi.projective_integrator_max_integration_distance_m = args.max_integration
    params.set_projective_integrator_params(pi)
    mapper = Mapper(voxel_sizes_m=args.voxel, integrator_types=ProjectiveIntegratorType.TSDF,
                    mapper_parameters=params)
    depth_sensor = Sensor.from_camera(depth_intr["fx"], depth_intr["fy"], depth_intr["cx"],
                                      depth_intr["cy"], DEPTH_W, DEPTH_H)
    colour_sensor = Sensor.from_camera(colour_intr["fx"], colour_intr["fy"], colour_intr["cx"],
                                       colour_intr["cy"], DEPTH_W, DEPTH_H)

    stream = Stream(args, name.replace("/", "_"), colour_intr)

    T0 = None                       # world <- cuVSLAM start frame (gravity alignment)
    T_odom0_inv = None
    times, poses, odom, n_obs, grav = [], [], [], [], []
    lost = 0
    t_start = time.time()
    for i, rec in enumerate(recs):
        rgb = np.array(Image.open(episode / rec["image"]).convert("RGB"))  # writable, for torch
        raw = read_depth(episode, rec).copy()
        raw[raw > args.depth_max * 1000] = 0
        depth_c = fill_warp_holes(align_depth_to_colour(raw, depth_intr, colour_intr,
                                                         T_colour_depth))
        if T0 is None:
            T0 = gravity_aligned_first_pose(rec, T_pelvis_colour, depth_c, colour_intr,
                                            args.depth_max)

        est, slam_pose = tracker.track(int(round(rec["time"] * 1e9)),
                                       images=[rgb], depths=[depth_c])
        pose = slam_pose if (args.slam and slam_pose is not None) else (
            est.world_from_rig.pose if est.world_from_rig is not None else None)
        if pose is None:
            lost += 1
            continue
        T_w_c = T0 @ pose_to_matrix(pose)

        # The legged odometry, expressed in the same world: anchored to start where
        # cuVSLAM starts, so the two trajectories can be overlaid without fitting.
        T_odom = odom_pose(rec) @ T_pelvis_colour
        if T_odom0_inv is None:
            T_odom0_inv = np.linalg.inv(T_odom)
        odom.append((T0 @ T_odom0_inv @ T_odom)[:3, 3])

        depth_t = torch.from_numpy(raw.astype(np.float32) / 1000.0).cuda()
        mapper.add_depth_frame(depth_t, torch.from_numpy(T_w_c @ T_colour_depth).float(),
                               depth_sensor)
        mapper.add_color_frame(torch.from_numpy(rgb).cuda(),
                               torch.from_numpy(T_w_c).float(), colour_sensor)

        times.append(rec["time"])
        poses.append(T_w_c)
        grav.append(gravity_error_deg(T_w_c, rec, T_pelvis_colour[:3, :3]))
        obs = tracker.get_last_observations(0)
        n_obs.append(len(obs))
        stream.frame(i, T_w_c, rgb, obs, np.array([T[:3, 3] for T in poses]), np.array(odom),
                     grav[-1])
        if stream.on and i % args.mesh_every == 0:
            mapper.update_color_mesh()
            stream.mesh(mapper.get_color_mesh())
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(recs)}  features {len(obs)}  "
                  f"{(i+1)/(time.time()-t_start):.1f} fps", flush=True)
    wall = time.time() - t_start

    mapper.update_color_mesh()
    mesh = mapper.get_color_mesh()
    stream.mesh(mesh)
    mesh.save(str(out / "mesh.ply"))
    write_tum(out / "traj_cuvslam.tum", times, poses)
    odom_T = []
    for p in odom:
        T = np.eye(4)
        T[:3, 3] = p
        odom_T.append(T)
    write_tum(out / "traj_odom.tum", times, odom_T)

    p = np.array([T[:3, 3] for T in poses]) if poses else np.zeros((0, 3))
    q = np.array(odom) if odom else np.zeros((0, 3))
    stats = {
        "episode": name, "frames": len(recs), "tracked": len(poses),
        "tracked_pct": round(100.0 * len(poses) / max(len(recs), 1), 1),
        "lost_frames": lost, "fps": round(len(recs) / wall, 1),
        "features_median": int(np.median(n_obs)) if n_obs else 0,
        "path_m": round(path_length(p), 3), "net_m": round(float(np.linalg.norm(p[-1] - p[0])), 3) if len(p) else 0.0,
        "odom_path_m": round(path_length(q), 3),
        "odom_net_m": round(float(np.linalg.norm(q[-1] - q[0])), 3) if len(q) else 0.0,
        "mesh_vertices": int(mesh.vertices().shape[0]), "slam": bool(args.slam),
    }
    if grav:
        stats["gravity_err_median_deg"] = round(float(np.median(grav)), 2)
        stats["gravity_err_max_deg"] = round(float(np.max(grav)), 2)
        stats["verdict"] = verdict(stats["gravity_err_max_deg"])
    if len(p) > 2:
        R, t = umeyama_se3(p, q)
        err = np.linalg.norm((R @ p.T).T + t - q, axis=1)
        stats["ate_vs_odom_rmse_m"] = round(float(np.sqrt((err ** 2).mean())), 3)
        stats["ate_vs_odom_max_m"] = round(float(err.max()), 3)
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"[{name}] {stats.get('verdict', '?')}: gravity error max "
          f"{stats.get('gravity_err_max_deg', float('nan'))} deg, tracked {stats['tracked_pct']} %, path {stats['path_m']} m "
          f"(odom {stats['odom_path_m']} m), ATE vs odom "
          f"{stats.get('ate_vs_odom_rmse_m', float('nan'))*100:.1f} cm, {stats['fps']} fps", flush=True)
    del mapper, tracker
    torch.cuda.empty_cache()
    return stats


# --------------------------------------------------------------------------- #
def write_summary(rows: list, path: Path) -> None:
    icon = {"ok": "✅ ok", "marginal": "⚠️ marginal", "drifted": "❌ drifted"}
    lines = ["| episode | frames | verdict | gravity err median / max (°) | path (m) | "
             "leg odom path (m) | net (m) | ATE vs leg odom (cm) | fps |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in rows:
        if "error" in s:
            lines.append(f"| {s['episode']} | — | ❌ {s['error']} | | | | | | |")
            continue
        ate = s.get("ate_vs_odom_rmse_m")
        lines.append(f"| {s['episode']} | {s['frames']} | {icon.get(s.get('verdict'), '?')} | "
                     f"{s.get('gravity_err_median_deg', '—')} / {s.get('gravity_err_max_deg', '—')} | "
                     f"{s['path_m']} | {s['odom_path_m']} | {s['net_m']} | "
                     f"{'—' if ate is None else f'{ate*100:.1f}'} | {s['fps']} |")
    n_ok = sum(s.get("verdict") == "ok" for s in rows)
    lines.append(f"\n{n_ok}/{len(rows)} episodes ok (gravity error < 5° throughout).")
    path.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episodes", type=Path, nargs="+", help="episode_N directories")
    ap.add_argument("--out", type=Path, default=Path("/results/cuvslam"))
    ap.add_argument("--robot", default="g1", choices=("g1", "h1"))
    ap.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    ap.add_argument("--slam", action="store_true",
                    help="enable cuVSLAM's SLAM layer (loop closure) and map with its poses")
    ap.add_argument("--voxel", type=float, default=0.015, help="TSDF voxel size, metres")
    ap.add_argument("--max-integration", type=float, default=5.0, help="metres")
    ap.add_argument("--depth-max", type=float, default=6.0, help="metres")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=0, help="0 = whole episode")
    ap.add_argument("--mesh-every", type=int, default=15,
                    help="stream the mesh every N frames (meshing costs ~10 ms)")
    ap.add_argument("--stream-url", default="rerun+http://127.0.0.1:9876/proxy")
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--no-stream-images", action="store_true")
    ap.add_argument("--save-rrd", action="store_true",
                    help="write <out>/rrd/<episode>.rrd instead of streaming")
    args = ap.parse_args()
    args.save_rrd_dir = args.out / "rrd"
    if args.save_rrd:
        args.save_rrd_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ep in args.episodes:
        ep = ep.resolve()
        out = args.out / ep.parent.name / ep.name
        try:
            rows.append(run_episode(ep, out, args))
        except Exception as e:  # keep going: one bad episode must not stop the sweep
            print(f"[{ep.parent.name}/{ep.name}] FAILED: {e!r}", flush=True)
            rows.append({"episode": f"{ep.parent.name}/{ep.name}", "error": type(e).__name__})
    write_summary(rows, args.out / "summary.md")


if __name__ == "__main__":
    main()
