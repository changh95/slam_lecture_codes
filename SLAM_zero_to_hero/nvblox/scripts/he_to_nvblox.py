#!/usr/bin/env python3
"""Convert one Humanoid Everyday episode into the dataset layout nvblox' fuser reads.

Humanoid Everyday ships an egocentric RealSense D435 stream -- JPEG colour, raw
uint16 depth inside an lzma container -- plus the Unitree G1/H1 state at 30 Hz.
nvblox' `fuse_replica` wants a directory that looks like a NICE-SLAM Replica
sequence:

    <out>/cam_params.json          w, h, fx, fy, cx, cy, scale
    <out>/seq/traj.txt             one 4x4 row-major T_world_camera per line
    <out>/seq/results/frameNNNNNN.jpg
    <out>/seq/results/depthNNNNNN.png

nvblox does no tracking of its own, so the interesting part of this script is
where traj.txt comes from. Two sources:

  --poses icp    depth frame-to-model ICP (Open3D, point-to-plane). Default.
  --poses odom   the robot's own legged odometry, composed with the fixed
                 pelvis->camera extrinsic from the config file.

Either way the odometry trajectory is also written out as TUM, so the two can be
compared afterwards.
"""

import argparse
import json
import lzma
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

DEPTH_H, DEPTH_W = 480, 640


# --------------------------------------------------------------------------- #
# config / calibration
# --------------------------------------------------------------------------- #
def load_calib(path: Path, robot: str) -> dict:
    cfg = json.loads(path.read_text())
    if robot not in cfg["robots"]:
        sys.exit(f"no calibration for robot '{robot}' in {path}")
    return cfg["robots"][robot]


def transform_from_dict(d: dict) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.array(d["rotation"], dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.array(d["translation"], dtype=np.float64)
    return T


# --------------------------------------------------------------------------- #
# episode reading
# --------------------------------------------------------------------------- #
def read_records(episode: Path) -> list:
    """Per-frame state. robot_data.jsonl is the streaming form of data.json."""
    jsonl = episode / "robot_data.jsonl"
    if jsonl.exists():
        return [json.loads(line) for line in jsonl.open()]
    blob = episode / "data.json"
    if blob.exists():
        return json.loads(blob.read_text())
    sys.exit(f"{episode} has neither robot_data.jsonl nor data.json")


def read_depth(episode: Path, rec: dict) -> np.ndarray:
    """uint16 millimetres. The .npy.lzma files hold a bare buffer, no npy header."""
    raw = lzma.decompress((episode / rec["depth"]).read_bytes())
    return np.frombuffer(raw, dtype=np.uint16).reshape(DEPTH_H, DEPTH_W)


def matrix_from_quat_wxyz(q) -> np.ndarray:
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def odom_pose(rec: dict) -> np.ndarray:
    """T_world_pelvis from the odometry position and its [w, x, y, z] quaternion."""
    T = np.eye(4)
    T[:3, :3] = matrix_from_quat_wxyz(rec["states"]["odometry"]["quat"])
    T[:3, 3] = rec["states"]["odometry"]["position"]
    return T


def gravity_aligned_first_pose(rec: dict, T_pelvis_cam: np.ndarray,
                               depth=None, intr=None, dmax=6.0) -> np.ndarray:
    """T_world_camera for the first frame, in a z-up world with the floor at z=0.

    ICP has no idea which way is up, so on its own it puts the map in the first
    camera's frame -- and that camera is pitched 56 deg into the floor, which
    leaves the mesh lying on its face and nvblox' ground-plane RANSAC (which
    only looks for a floor between z=-0.1 and z=0.15) with nothing to find.

    The IMU fixes the orientation: its roll and pitch are gravity-referenced.
    Take those and drop the yaw, so the world x axis is just the robot's initial
    heading. The height comes from the floor in the first depth frame, which is
    a direct measurement and beats the assumed pelvis-to-camera offset by about
    10 cm.
    """
    imu = rec["states"]["imu"]
    yaw = imu["rpy"][2]
    cy, sy = np.cos(-yaw), np.sin(-yaw)
    R_yaw_out = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    R_world_cam = R_yaw_out @ matrix_from_quat_wxyz(imu["quaternion"]) \
        @ T_pelvis_cam[:3, :3]

    height = None
    if depth is not None:
        height = floor_height_above_camera(depth, intr, R_world_cam, dmax)
    if height is None:
        height = (rec["states"]["odometry"]["position"][2] + T_pelvis_cam[2, 3])
        print("[convert] no floor plane in the first frame; falling back to the "
              f"odometry + extrinsic height ({height:.3f} m)", flush=True)

    T = np.eye(4)
    T[:3, :3] = R_world_cam
    T[2, 3] = height
    return T


def floor_height_above_camera(depth_mm, intr, R_world_cam, dmax=6.0):
    """How far the camera sits above the floor, from one depth frame.

    RANSAC every dominant plane in turn and keep the first whose normal is
    within 20 deg of world up and which lies below the camera -- otherwise a
    tabletop, which is the other big horizontal plane in these scenes, wins.
    """
    import open3d as o3d

    d = depth_mm.astype(np.float32) / 1000.0
    d[(d <= 0.2) | (d > dmax)] = 0.0
    o3d_intr = o3d.camera.PinholeCameraIntrinsic(
        DEPTH_W, DEPTH_H, intr["fx"], intr["fy"], intr["cx"], intr["cy"])
    pc = o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image(d), o3d_intr, depth_scale=1.0, depth_trunc=dmax, stride=3)
    pc = pc.voxel_down_sample(0.04)
    # Work in the world frame with the camera still at the origin, so a floor
    # plane has normal +-z and sits at negative z.
    pc.transform(np.block([[R_world_cam, np.zeros((3, 1))],
                           [np.zeros((1, 3)), np.ones((1, 1))]]))
    for _ in range(3):
        if len(pc.points) < 500:
            break
        model, inliers = pc.segment_plane(distance_threshold=0.04, ransac_n=3,
                                         num_iterations=300)
        a, b, c, dd = model
        if abs(c) > np.cos(np.radians(20)):
            # signed height of the plane at x=y=0, i.e. under the camera
            z0 = -dd / c
            if z0 < -0.3:
                return float(-z0)
        pc = pc.select_by_index(inliers, invert=True)
    return None


# --------------------------------------------------------------------------- #
# depth -> colour alignment
# --------------------------------------------------------------------------- #
def align_depth_to_colour(depth_mm, depth_intr, colour_intr, T_colour_depth):
    """Warp raw depth into the colour camera, the way librealsense' align does.

    The two D435 imagers do not share intrinsics -- depth is ~79 deg wide, colour
    ~56 deg -- and the dataset stores depth in the *depth* frame (its own example
    loader deprojects it with the depth intrinsics). nvblox' Replica loader
    assumes a single camera for both images, so one of them has to be resampled
    into the other's frame.
    """
    h, w = DEPTH_H, DEPTH_W
    z = depth_mm.astype(np.float32) / 1000.0
    v, u = np.nonzero(z > 0)
    zz = z[v, u]
    pts = np.stack([(u - depth_intr["cx"]) * zz / depth_intr["fx"],
                    (v - depth_intr["cy"]) * zz / depth_intr["fy"],
                    zz])
    pts = T_colour_depth[:3, :3] @ pts + T_colour_depth[:3, 3:4]
    pts = pts[:, pts[2] > 0]
    cu = np.rint(pts[0] / pts[2] * colour_intr["fx"] + colour_intr["cx"]).astype(np.int64)
    cv = np.rint(pts[1] / pts[2] * colour_intr["fy"] + colour_intr["cy"]).astype(np.int64)
    inside = (cu >= 0) & (cu < w) & (cv >= 0) & (cv < h)
    cu, cv, cz = cu[inside], cv[inside], pts[2][inside]

    # z-buffer: the nearest surface wins, which is what occlusion means here.
    out = np.full(h * w, np.inf, dtype=np.float32)
    np.minimum.at(out, cv * w + cu, cz)
    out[~np.isfinite(out)] = 0.0
    return np.rint(out.reshape(h, w) * 1000.0).astype(np.uint16)


def warp_colour_to_depth(colour, depth_mm, depth_intr, colour_intr, T_colour_depth):
    """Resample the colour image into the depth camera, using the depth itself.

    The mirror of align_depth_to_colour, and the one that keeps the depth
    camera's much wider field of view (79 deg against 56 deg) -- worth having,
    because coverage per frame is what fills the map in. Pixels the colour
    camera never saw stay black; nvblox has no notion of "colour unknown", so
    those parts of the mesh come out unshaded.
    """
    h, w = DEPTH_H, DEPTH_W
    z = depth_mm.astype(np.float32) / 1000.0
    v, u = np.nonzero(z > 0)
    zz = z[v, u]
    pts = np.stack([(u - depth_intr["cx"]) * zz / depth_intr["fx"],
                    (v - depth_intr["cy"]) * zz / depth_intr["fy"],
                    zz])
    pts = T_colour_depth[:3, :3] @ pts + T_colour_depth[:3, 3:4]
    ok = pts[2] > 0
    cu = np.rint(pts[0] / pts[2] * colour_intr["fx"] + colour_intr["cx"]).astype(np.int64)
    cv = np.rint(pts[1] / pts[2] * colour_intr["fy"] + colour_intr["cy"]).astype(np.int64)
    ok &= (cu >= 0) & (cu < w) & (cv >= 0) & (cv < h)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[v[ok], u[ok]] = colour[cv[ok], cu[ok]]
    return out


def fill_pinholes(depth_mm: np.ndarray) -> np.ndarray:
    """Median-fill single-pixel holes left by forward warping.

    Forward projection scatters a coarse grid into a finer one and leaves speckle
    between the samples. Only pixels with at least 5 valid 8-neighbours are
    filled, so real depth boundaries and the shadow behind an object stay empty.
    """
    nb = np.stack([np.roll(np.roll(depth_mm, dv, axis=0), du, axis=1)
                   for dv in (-1, 0, 1) for du in (-1, 0, 1)
                   if not (dv == 0 and du == 0)]).astype(np.float32)
    valid = nb > 0
    count = valid.sum(axis=0)
    nb[~valid] = np.nan
    fill = (depth_mm == 0) & (count >= 5)
    out = depth_mm.copy()
    if fill.any():
        # Take the median only over columns that have a valid neighbour, so
        # nanmedian never sees an all-NaN slice.
        med = np.nanmedian(nb[:, fill], axis=0)
        out[fill] = med.astype(np.uint16)
    return out


# --------------------------------------------------------------------------- #
# tracking
# --------------------------------------------------------------------------- #
def track_icp(depths, intr, voxel, dmax, icp_dist, T_world_cam0=None, verbose=True):
    """Depth-only frame-to-model ICP. Returns a list of T_world_camera.

    Frame-to-model rather than frame-to-frame: consecutive-frame ICP compounds
    its own error at every step and 900 frames of that pulls the map apart.
    Registering against the accumulated cloud holds the drift down to what the
    model itself has accumulated.
    """
    import open3d as o3d

    o3d_intr = o3d.camera.PinholeCameraIntrinsic(
        DEPTH_W, DEPTH_H, intr["fx"], intr["fy"], intr["cx"], intr["cy"])
    normal_param = o3d.geometry.KDTreeSearchParamHybrid(radius=4 * voxel, max_nn=30)

    def cloud(depth_mm):
        d = depth_mm.astype(np.float32) / 1000.0
        d[(d <= 0.15) | (d > dmax)] = 0.0
        pc = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(d), o3d_intr, depth_scale=1.0, depth_trunc=dmax, stride=2)
        pc = pc.voxel_down_sample(voxel)
        pc.estimate_normals(normal_param)
        return pc

    T0 = np.eye(4) if T_world_cam0 is None else np.asarray(T_world_cam0, dtype=float)
    poses = [T0]
    model = o3d.geometry.PointCloud(cloud(depths[0])).transform(T0)
    model.estimate_normals(normal_param)
    fitness = []
    t0 = time.time()
    for i in range(1, len(depths)):
        cur = cloud(depths[i])
        # Constant-velocity guess: the previous step, repeated.
        init = poses[-1] if len(poses) < 2 else poses[-1] @ np.linalg.inv(poses[-2]) @ poses[-1]
        reg = o3d.pipelines.registration.registration_icp(
            cur, model, icp_dist, init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
        T = np.array(reg.transformation)
        poses.append(T)
        fitness.append(reg.fitness)
        model += o3d.geometry.PointCloud(cur).transform(T)
        if i % 10 == 0:
            model = model.voxel_down_sample(voxel)
            model.estimate_normals(normal_param)
        if verbose and i % 100 == 0:
            print(f"    icp {i}/{len(depths)-1}  fitness={reg.fitness:.3f} "
                  f"rmse={reg.inlier_rmse*100:.1f} cm  {time.time()-t0:.0f} s", flush=True)
    if verbose:
        print(f"    icp done in {time.time()-t0:.0f} s, "
              f"median fitness {np.median(fitness):.3f}", flush=True)
    return poses


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def quat_from_matrix(R):
    """[qx, qy, qz, qw] from a rotation matrix, via the largest-diagonal branch."""
    tr = R.trace()
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                         (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2
    q = np.zeros(4)
    q[i], q[j], q[k] = 0.25 * s, (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return q


def write_tum(path: Path, times, poses):
    with path.open("w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for t, T in zip(times, poses):
            qx, qy, qz, qw = quat_from_matrix(T[:3, :3])
            x, y, z = T[:3, 3]
            f.write(f"{t:.6f} {x:.6f} {y:.6f} {z:.6f} "
                    f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode", type=Path, help="path to an episode_N directory")
    ap.add_argument("out", type=Path, help="output dataset directory")
    ap.add_argument("--robot", default="g1", choices=("g1", "h1"))
    ap.add_argument("--calib", type=Path,
                    default=Path(__file__).resolve().parent.parent / "config" /
                    "humanoid_everyday_d435.json")
    ap.add_argument("--poses", default="icp", choices=("icp", "odom"),
                    help="pose source for traj.txt (default: icp)")
    ap.add_argument("--world", default="gravity", choices=("gravity", "camera"),
                    help="world frame for --poses icp: 'gravity' anchors the "
                         "first pose with the IMU so z is up and the floor is "
                         "near z=0, 'camera' leaves the first camera at the "
                         "origin (default: gravity)")
    ap.add_argument("--frame", default="colour", choices=("colour", "depth"),
                    help="camera frame the RGB-D pair is written in "
                         "(default: colour, i.e. depth is warped into it)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=0, help="0 = to the end of the episode")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--depth-max", type=float, default=6.0,
                    help="metres; D435 stereo depth is mostly noise past this indoors")
    ap.add_argument("--icp-voxel", type=float, default=0.03)
    ap.add_argument("--icp-dist", type=float, default=0.15)
    args = ap.parse_args()

    calib = load_calib(args.calib, args.robot)
    depth_intr = calib["depth_intrinsics"]
    colour_intr = calib["colour_intrinsics"]
    T_colour_depth = transform_from_dict(calib["depth_to_colour"])
    T_pelvis_colour = transform_from_dict(calib["pelvis_to_colour_camera"])

    recs = read_records(args.episode)
    end = len(recs) if args.frames <= 0 else min(len(recs), args.start + args.frames * args.stride)
    recs = recs[args.start:end:args.stride]
    if not recs:
        sys.exit("no frames selected")
    print(f"[convert] {args.episode.name}: {len(recs)} frames "
          f"(start={args.start} stride={args.stride}, frame={args.frame})", flush=True)

    results = args.out / "seq" / "results"
    results.mkdir(parents=True, exist_ok=True)
    out_intr = colour_intr if args.frame == "colour" else depth_intr

    depths = []
    for i, rec in enumerate(recs):
        raw = read_depth(args.episode, rec)
        if args.frame == "colour":
            d = fill_pinholes(align_depth_to_colour(
                raw, depth_intr, colour_intr, T_colour_depth))
            d[d > args.depth_max * 1000] = 0
            shutil.copyfile(args.episode / rec["image"], results / f"frame{i:06d}.jpg")
        else:
            d = raw.copy()
            d[d > args.depth_max * 1000] = 0
            colour = np.asarray(Image.open(args.episode / rec["image"]).convert("RGB"))
            warped = warp_colour_to_depth(colour, d, depth_intr, colour_intr,
                                          T_colour_depth)
            Image.fromarray(warped).save(results / f"frame{i:06d}.jpg", quality=95)
        depths.append(d)
        Image.fromarray(d).save(results / f"depth{i:06d}.png")
        if (i + 1) % 200 == 0:
            print(f"    wrote {i+1}/{len(recs)} frames", flush=True)

    (args.out / "cam_params.json").write_text(json.dumps({
        "camera": {"w": DEPTH_W, "h": DEPTH_H,
                   "fx": out_intr["fx"], "fy": out_intr["fy"],
                   "cx": out_intr["cx"], "cy": out_intr["cy"],
                   "scale": 1000.0}
    }, indent=2) + "\n")

    times = [r["time"] for r in recs]
    # In --frame depth the images belong to the depth camera, so the extrinsic
    # has to hop back across the (tiny) depth-to-colour baseline.
    T_pelvis_out = T_pelvis_colour if args.frame == "colour" else \
        T_pelvis_colour @ np.linalg.inv(T_colour_depth)
    odom = [odom_pose(r) @ T_pelvis_out for r in recs]

    if args.poses == "odom":
        traj = odom
    else:
        T0 = None
        if args.world == "gravity":
            T0 = gravity_aligned_first_pose(recs[0], T_pelvis_out, depths[0],
                                            out_intr, args.depth_max)
            print(f"[convert] world frame: gravity-aligned, camera starts "
                  f"{T0[2, 3]:.3f} m above the floor at z=0", flush=True)
        print("[convert] tracking with depth frame-to-model ICP", flush=True)
        traj = track_icp(depths, out_intr, args.icp_voxel, args.depth_max,
                         args.icp_dist, T_world_cam0=T0)

    with (args.out / "seq" / "traj.txt").open("w") as f:
        for T in traj:
            f.write(" ".join(f"{v:.9f}" for v in T.reshape(-1)) + "\n")
    write_tum(args.out / f"traj_{args.poses}.tum", times, traj)
    write_tum(args.out / "traj_odom.tum", times, odom)

    p = np.array([T[:3, 3] for T in traj])
    q = np.array([T[:3, 3] for T in odom])
    print(f"[convert] {len(traj)} poses from '{args.poses}'")
    print(f"          path length: {args.poses}="
          f"{np.linalg.norm(np.diff(p, axis=0), axis=1).sum():.2f} m, "
          f"odom={np.linalg.norm(np.diff(q, axis=0), axis=1).sum():.2f} m")
    print(f"          extent: {np.round(np.ptp(p, axis=0), 2)} m")
    print(f"[convert] done -> {args.out / 'seq'}")


if __name__ == "__main__":
    main()
