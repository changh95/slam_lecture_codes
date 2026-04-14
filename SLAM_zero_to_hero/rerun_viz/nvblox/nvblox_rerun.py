#!/usr/bin/env python3
"""
nvblox_rerun.py - Run nvblox reconstruction and publish mesh/voxels to Rerun viewer.

Supports two modes:
  1. live   - integrates frames one by one and pushes incremental mesh updates to Rerun
  2. replay - loads a previously saved .ply mesh file and displays it in Rerun

Dataset layouts supported:
  - 3DMatch / Replica:  depth/ color/ pose/ subdirectories
  - TUM RGB-D format:   depth.txt, rgb.txt, groundtruth.txt

Rerun entities logged:
  world/axes          - XYZ reference frame
  nvblox/pose         - camera transform per frame (live mode)
  nvblox/mesh         - reconstructed triangle mesh (updated every N frames)
  nvblox/tsdf_voxels  - TSDF voxel centres coloured by weight (if available)

Usage:
    python3 nvblox_rerun.py [dataset_path] [--mode live|replay] [--mesh-ply path.ply]

Examples:
    # Live reconstruction of 3DMatch scene
    python3 nvblox_rerun.py /data/3dmatch/scene --mode live

    # Replay a pre-built mesh
    python3 nvblox_rerun.py --mode replay --mesh-ply /output/nvblox_mesh.ply

Open viewer at:
    http://localhost:9090/?url=rerun+http://localhost:9876/proxy
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import rerun as rr
import rerun.blueprint as rrb


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _load_pose_file(path: str) -> np.ndarray:
    return np.loadtxt(path).reshape(4, 4)


def _discover_frames_3dmatch(root: str):
    depth_files = sorted(glob.glob(os.path.join(root, "depth", "*.png")))
    if not depth_files:
        depth_files = sorted(glob.glob(os.path.join(root, "depth", "*.pgm")))
    frames = []
    for d in depth_files:
        base = os.path.basename(d).replace(".depth.png", "").replace(".png", "").replace(".pgm", "")
        color = None
        for ext in [".color.png", ".color.jpg", ".png", ".jpg"]:
            c = os.path.join(root, "color", base + ext)
            if os.path.exists(c):
                color = c
                break
        pose_path = os.path.join(root, "pose", base + ".pose.txt")
        if not os.path.exists(pose_path):
            pose_path = os.path.join(root, "pose", base + ".txt")
        if color is not None and os.path.exists(pose_path):
            frames.append((d, color, pose_path))
    return frames


def _discover_frames_tum(root: str):
    def _read_assoc(txt):
        entries = []
        with open(txt) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                entries.append((float(parts[0]), parts[1]))
        return entries

    depth_txt = os.path.join(root, "depth.txt")
    rgb_txt = os.path.join(root, "rgb.txt")
    gt_txt = os.path.join(root, "groundtruth.txt")
    if not (os.path.exists(depth_txt) and os.path.exists(gt_txt)):
        return []

    depth_entries = _read_assoc(depth_txt)
    rgb_entries = _read_assoc(rgb_txt) if os.path.exists(rgb_txt) else []

    gt_raw = []
    with open(gt_txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            ts = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            gt_raw.append((ts, tx, ty, tz, qx, qy, qz, qw))

    def _quat_to_mat(tx, ty, tz, qx, qy, qz, qw):
        T = np.eye(4)
        T[0, 3], T[1, 3], T[2, 3] = tx, ty, tz
        T[0, 0] = 1 - 2*(qy*qy + qz*qz); T[0, 1] = 2*(qx*qy - qz*qw); T[0, 2] = 2*(qx*qz + qy*qw)
        T[1, 0] = 2*(qx*qy + qz*qw); T[1, 1] = 1 - 2*(qx*qx + qz*qz); T[1, 2] = 2*(qy*qz - qx*qw)
        T[2, 0] = 2*(qx*qz - qy*qw); T[2, 1] = 2*(qy*qz + qx*qw); T[2, 2] = 1 - 2*(qx*qx + qy*qy)
        return T

    gt_ts = np.array([r[0] for r in gt_raw])

    def _nearest_pose(ts):
        idx = np.argmin(np.abs(gt_ts - ts))
        r = gt_raw[idx]
        return _quat_to_mat(r[1], r[2], r[3], r[4], r[5], r[6], r[7])

    rgb_ts = np.array([e[0] for e in rgb_entries]) if rgb_entries else None
    frames = []
    for d_ts, d_rel in depth_entries:
        d_path = os.path.join(root, d_rel)
        color_path = None
        if rgb_ts is not None and len(rgb_ts) > 0:
            idx = np.argmin(np.abs(rgb_ts - d_ts))
            color_path = os.path.join(root, rgb_entries[idx][1])
        pose = _nearest_pose(d_ts)
        frames.append((d_path, color_path, pose))
    return frames


# ---------------------------------------------------------------------------
# PLY mesh loader (no open3d dependency - minimal fallback)
# ---------------------------------------------------------------------------

def _load_ply(path: str):
    """
    Load a PLY mesh, returning (vertices, triangles, vertex_colors).
    Tries open3d first, then a minimal ASCII/binary PLY reader for triangles.
    vertex_colors may be None.
    """
    try:
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(path)
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        tris = np.asarray(mesh.triangles, dtype=np.uint32)
        colors = np.asarray(mesh.vertex_colors, dtype=np.float32) if mesh.has_vertex_colors() else None
        return verts, tris, colors
    except ImportError:
        pass

    # Minimal ASCII PLY fallback
    verts, tris, colors = [], [], []
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            header.append(line)
            if line == "end_header":
                break
        # Parse header for element counts
        n_verts = n_faces = 0
        has_color = False
        for line in header:
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            elif line.startswith("element face"):
                n_faces = int(line.split()[-1])
            elif "red" in line or "green" in line:
                has_color = True
        # Read ASCII body
        for _ in range(n_verts):
            row = f.readline().decode().split()
            verts.append([float(row[0]), float(row[1]), float(row[2])])
            if has_color and len(row) >= 6:
                colors.append([int(row[3])/255., int(row[4])/255., int(row[5])/255.])
        for _ in range(n_faces):
            row = f.readline().decode().split()
            n = int(row[0])
            if n == 3:
                tris.append([int(row[1]), int(row[2]), int(row[3])])

    v = np.array(verts, dtype=np.float32)
    t = np.array(tris, dtype=np.uint32)
    c = np.array(colors, dtype=np.float32) if colors else None
    return v, t, c


# ---------------------------------------------------------------------------
# Rerun helpers
# ---------------------------------------------------------------------------

def _log_world_axes():
    """Log XYZ reference axes."""
    rr.log(
        "world/axes",
        rr.Arrows3D(
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            vectors=[[0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.3]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
        static=True,
    )


def _log_camera_pose(frame_idx: int, T_world_cam: np.ndarray):
    """Log camera pose as a transform + frustum indicator."""
    rr.set_time_sequence("frame", frame_idx)
    translation = T_world_cam[:3, 3]
    mat3 = T_world_cam[:3, :3]
    rr.log(
        "nvblox/pose",
        rr.Transform3D(
            translation=translation,
            mat3x3=mat3,
        ),
    )
    # Small box to mark sensor position
    rr.log(
        "nvblox/pose/sensor",
        rr.Boxes3D(half_sizes=[[0.05, 0.05, 0.05]], colors=[[0, 200, 255]]),
    )


def _log_mesh(vertices: np.ndarray, triangles: np.ndarray, colors=None):
    """Publish reconstructed mesh to Rerun."""
    if vertices is None or len(vertices) == 0:
        return
    kwargs = dict(
        vertex_positions=vertices,
        triangle_indices=triangles,
    )
    if colors is not None and len(colors) == len(vertices):
        # Rerun expects uint8 [0,255] vertex colors
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        kwargs["vertex_colors"] = colors
    rr.log("nvblox/mesh", rr.Mesh3D(**kwargs))


def _log_voxel_centres(centres: np.ndarray, weights=None):
    """Log TSDF voxel centres as point cloud coloured by weight."""
    if centres is None or len(centres) == 0:
        return
    if weights is not None:
        w_norm = np.clip(weights / weights.max(), 0, 1)
        colors = np.stack(
            [
                (w_norm * 255).astype(np.uint8),
                np.zeros_like(w_norm, dtype=np.uint8),
                ((1 - w_norm) * 255).astype(np.uint8),
            ],
            axis=1,
        )
    else:
        colors = None
    kwargs = dict(positions=centres)
    if colors is not None:
        kwargs["colors"] = colors
    rr.log("nvblox/tsdf_voxels", rr.Points3D(**kwargs))


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_live(dataset_path: str, voxel_size: float, mesh_interval: int, max_frames: int):
    """Integrate frames live and push incremental mesh updates."""
    try:
        import nvblox
    except ImportError:
        print(
            "[ERROR] nvblox Python bindings not installed.\n"
            "  pip3 install nvblox\n"
            "  Or build from source: https://github.com/nvidia-isaac/nvblox",
            file=sys.stderr,
        )
        sys.exit(1)

    frames = _discover_frames_3dmatch(dataset_path)
    if not frames:
        frames = _discover_frames_tum(dataset_path)
    if not frames:
        print(f"[ERROR] No frames found in {dataset_path}", file=sys.stderr)
        sys.exit(1)

    if max_frames > 0:
        frames = frames[:max_frames]

    print(f"Live mode: {len(frames)} frames, voxel_size={voxel_size}m, mesh every {mesh_interval} frames")

    _log_world_axes()

    mapper = nvblox.Mapper(voxel_size_m=voxel_size, projective_layer_type="tsdf")

    for i, (depth_path, color_path, pose) in enumerate(frames):
        if isinstance(pose, str):
            pose = _load_pose_file(pose)

        depth_img = nvblox.DepthImage.fromFile(depth_path)
        mapper.integrateDepth(depth_img, pose)

        if color_path is not None and os.path.exists(str(color_path)):
            color_img = nvblox.ColorImage.fromFile(color_path)
            mapper.integrateColor(color_img, pose)

        _log_camera_pose(i, pose)

        if i % mesh_interval == 0:
            mapper.updateMesh()
            try:
                mesh = mapper.extractMesh()
                verts = np.array(mesh.vertices, dtype=np.float32).reshape(-1, 3)
                tris = np.array(mesh.triangles, dtype=np.uint32).reshape(-1, 3)
                colors = np.array(mesh.colors, dtype=np.float32).reshape(-1, 3) if hasattr(mesh, "colors") else None
                _log_mesh(verts, tris, colors)
            except Exception as e:
                print(f"  [WARN] mesh extraction failed at frame {i}: {e}")

        if i % 50 == 0:
            print(f"  Frame {i}/{len(frames)}")

    print("Done. Final mesh extract...")
    try:
        mapper.updateMesh()
        mesh = mapper.extractMesh()
        verts = np.array(mesh.vertices, dtype=np.float32).reshape(-1, 3)
        tris = np.array(mesh.triangles, dtype=np.uint32).reshape(-1, 3)
        colors = np.array(mesh.colors, dtype=np.float32).reshape(-1, 3) if hasattr(mesh, "colors") else None
        _log_mesh(verts, tris, colors)
        print(f"Final mesh: {len(verts)} vertices, {len(tris)} triangles")
    except Exception as e:
        print(f"[WARN] Final mesh extraction failed: {e}")


def run_replay(mesh_ply: str):
    """Load and display a pre-built .ply mesh in Rerun."""
    if not os.path.exists(mesh_ply):
        print(f"[ERROR] Mesh file not found: {mesh_ply}", file=sys.stderr)
        sys.exit(1)

    print(f"Replay mode: loading {mesh_ply}")
    _log_world_axes()

    verts, tris, colors = _load_ply(mesh_ply)
    print(f"Loaded mesh: {len(verts)} vertices, {len(tris)} triangles")
    _log_mesh(verts, tris, colors)
    print("Mesh logged to Rerun. Viewer will hold open.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="nvblox 3D reconstruction visualizer using Rerun"
    )
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default="/data",
        help="Dataset root path (live mode only)",
    )
    parser.add_argument(
        "--mode",
        choices=["live", "replay"],
        default="live",
        help="live: integrate + visualize; replay: display pre-built mesh PLY",
    )
    parser.add_argument(
        "--mesh-ply",
        default="/output/nvblox_mesh.ply",
        help="Path to .ply mesh for replay mode",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.05,
        help="TSDF voxel size in metres (live mode, default: 0.05)",
    )
    parser.add_argument(
        "--mesh-interval",
        type=int,
        default=10,
        help="Update mesh every N frames (live mode, default: 10)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Limit number of frames processed (0 = all)",
    )
    parser.add_argument(
        "--rerun-addr",
        default="0.0.0.0:9876",
        help="Rerun gRPC serve address (default: 0.0.0.0:9876)",
    )
    args = parser.parse_args()

    # Initialise Rerun - serve so the web viewer can connect
    rr.init("nvblox_rerun", spawn=False)
    import os
    _ip = os.environ.get("SERVER_IP", "localhost")
    rr.serve_grpc(grpc_port=9876)
    rr.serve_web_viewer(web_port=9090, open_browser=False,
                        connect_to=f"rerun+http://{_ip}:9876/proxy")
    print(f"Rerun web viewer: http://{_ip}:9090")

    if args.mode == "replay":
        run_replay(args.mesh_ply)
    else:
        run_live(args.dataset_path, args.voxel_size, args.mesh_interval, args.max_frames)

    # Keep server alive so the viewer can stay connected
    print("Streaming complete. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
