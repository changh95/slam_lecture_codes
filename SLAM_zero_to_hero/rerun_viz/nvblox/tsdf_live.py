#!/usr/bin/env python3
"""
tsdf_live.py - Real-time camera tracking + growing nvblox TSDF mesh in Rerun.

Strategy:
  - Background thread: run fuse_replica with increasing --num_frames to
    generate TSDF mesh snapshots. These meshes are in nvblox's internal
    world frame (whatever traj.txt encodes).
  - Main thread: stream RGB/depth + camera poses from traj.txt (RAW, no
    flip) at playback FPS. Poses and mesh are in the SAME nvblox frame,
    so the camera walks through the reconstructed surface.

The camera pose is logged as a Transform3D + a small axes indicator (not
a Pinhole frustum, because frustum orientation depends on interpreting
the local camera axes which nvblox does not document).
"""

import argparse
import os
import queue as queue_mod
import struct
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image


FUSE_REPLICA = "/usr/local/bin/nvblox/fuse_replica"


def load_ply_mesh(path):
    """Load a PLY mesh via Open3D (handles ASCII + binary + normals + colors)."""
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(path)
    verts = np.asarray(m.vertices, dtype=np.float32)
    tris = np.asarray(m.triangles, dtype=np.uint32)
    vcols = np.asarray(m.vertex_colors)
    if len(vcols) == len(verts):
        cols = (vcols * 255).astype(np.uint8)
    else:
        cols = None
    return verts, (tris if len(tris) > 0 else None), cols


def load_raw_poses(path):
    """Load traj.txt poses AS-IS (same frame as fuse_replica mesh output)."""
    traj_file = os.path.join(path, "traj.txt")
    poses = []
    with open(traj_file) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 16:
                poses.append(np.array(vals, dtype=np.float32).reshape(4, 4))
    return poses


def load_replica_intrinsics(dataset_path):
    """Load cam_params.json from <dataset_path>/../cam_params.json (the same
    file nvblox's fuse_replica reads)."""
    import json
    cam_file = os.path.join(os.path.dirname(dataset_path.rstrip('/')), "cam_params.json")
    with open(cam_file) as f:
        data = json.load(f)["camera"]
    return {
        "W": int(data["w"]),
        "H": int(data["h"]),
        "fx": float(data["fx"]),
        "fy": float(data["fy"]),
        "cx": float(data["cx"]),
        "cy": float(data["cy"]),
        "depth_scale": float(data["scale"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", help="Replica sequence dir")
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--playback-fps", type=float, default=5.0)
    args = parser.parse_args()

    server_ip = os.environ.get("SERVER_IP", "localhost")

    # World rotation: -90° about X axis, applied to both mesh and poses.
    # Rx(-90°) = [[1,0,0],[0,0,1],[0,-1,0]]
    R_world = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ], dtype=np.float32)
    T_world = np.eye(4, dtype=np.float32)
    T_world[:3, :3] = R_world

    poses = load_raw_poses(args.dataset_path)
    # Apply world rotation to all poses: T_new = T_world @ T_c2w
    poses = [T_world @ p for p in poses]
    n_frames = min(args.max_frames, len(poses))
    rgb_dir = os.path.join(args.dataset_path, "results")

    # Load intrinsics from the same file nvblox/fuse_replica reads
    intr = load_replica_intrinsics(args.dataset_path)
    print(f"Loaded {len(poses)} poses, streaming {n_frames}", flush=True)
    print(f"Intrinsics (from cam_params.json): W={intr['W']} H={intr['H']} "
          f"fx={intr['fx']} fy={intr['fy']} cx={intr['cx']} cy={intr['cy']} "
          f"depth_scale={intr['depth_scale']}", flush=True)

    # Background mesh worker
    current_frame = [0]
    stop_flag = [False]
    mesh_queue = queue_mod.Queue()
    mesh_dir = tempfile.mkdtemp()

    def mesh_worker():
        last_n = 0
        while not stop_flag[0]:
            n = max(current_frame[0], 10)
            if n <= last_n:
                time.sleep(0.05)
                continue
            out = os.path.join(mesh_dir, f"mesh_{n}.ply")
            t0 = time.time()
            subprocess.run([
                FUSE_REPLICA, args.dataset_path,
                "--voxel_size", str(args.voxel_size),
                "--num_frames", str(n),
                "--mesh_output_path", out,
            ], capture_output=True)
            elapsed = time.time() - t0
            if os.path.exists(out):
                mesh_queue.put((n, out))
                print(f"[mesh] snapshot {n} frames in {elapsed:.1f}s", flush=True)
                last_n = n

    # Rerun
    rr.init("tsdf_live")
    rr.serve_grpc(grpc_port=9876)
    rr.serve_web_viewer(web_port=9090, open_browser=False,
                        connect_to=f"rerun+http://{server_ip}:9876/proxy")
    print(f"\n>>> Rerun web viewer at http://{server_ip}:9090 <<<\n", flush=True)

    rr.send_blueprint(rrb.Blueprint(
        rrb.TimePanel(state="collapsed"),
        rrb.Vertical(row_shares=[0.7, 0.3], contents=[
            rrb.Spatial3DView(name="Map", origin="/", contents=["+ $origin/**"]),
            rrb.Horizontal(contents=[
                rrb.Spatial2DView(name="RGB", origin="camera/image"),
                rrb.Spatial2DView(name="Depth", origin="camera/depth"),
            ]),
        ]),
    ))
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
    rr.log("world_axes", rr.Arrows3D(
        vectors=[[0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.3]],
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    ), static=True)

    worker = threading.Thread(target=mesh_worker, daemon=True)
    worker.start()

    traj = []
    frame_interval_s = 1.0 / args.playback_fps

    for i in range(n_frames):
        current_frame[0] = i
        rr.set_time("frame", sequence=i)

        rgb_path = os.path.join(rgb_dir, f"frame{i:06d}.jpg")
        depth_path = os.path.join(rgb_dir, f"depth{i:06d}.png")
        if os.path.exists(rgb_path):
            img = np.asarray(Image.open(rgb_path))
            rr.log("camera/image", rr.Image(img).compress(jpeg_quality=70))
        if os.path.exists(depth_path):
            depth = np.asarray(Image.open(depth_path))
            rr.log("camera/depth", rr.DepthImage(depth, meter=intr["depth_scale"]))

        # Camera pose from RAW traj.txt — same frame as fuse_replica mesh.
        # Pass mat3x3 directly to Rerun so we don't risk bugs in our own
        # rotation-to-quaternion conversion.
        T = poses[i]
        t = T[:3, 3].astype(np.float32)
        R = T[:3, :3].astype(np.float32)
        rr.log("camera", rr.Transform3D(translation=t, mat3x3=R))
        rr.log("camera", rr.Pinhole(
            focal_length=[intr["fx"], intr["fy"]],
            principal_point=[intr["cx"], intr["cy"]],
            resolution=[intr["W"], intr["H"]],
            image_plane_distance=0.2,
            camera_xyz=rr.ViewCoordinates.RDF,
        ))
        traj.append(t.tolist())
        if len(traj) >= 2:
            rr.log("trajectory", rr.LineStrips3D([traj], colors=[[255, 255, 0]]))

        # Drain mesh queue
        while not mesh_queue.empty():
            try:
                nf, ply_path = mesh_queue.get_nowait()
                verts, tris, colors = load_ply_mesh(ply_path)
                # Apply the same world rotation as we did to the poses
                verts = (R_world @ verts.T).T
                if tris is not None and len(tris) > 0:
                    rr.log("tsdf/mesh", rr.Mesh3D(
                        vertex_positions=verts,
                        triangle_indices=tris,
                        vertex_colors=colors,
                    ))
                else:
                    rr.log("tsdf/mesh", rr.Points3D(verts, colors=colors, radii=0.02))
                print(f"  frame {i}: mesh refreshed ({len(verts)} verts, "
                      f"{len(tris) if tris is not None else 0} tris)", flush=True)
            except queue_mod.Empty:
                break

        time.sleep(frame_interval_s)

    stop_flag[0] = True
    worker.join(timeout=20)
    # Final drain
    while not mesh_queue.empty():
        try:
            nf, ply_path = mesh_queue.get_nowait()
            verts, tris, colors = load_ply_mesh(ply_path)
            verts = (R_world @ verts.T).T
            if tris is not None and len(tris) > 0:
                rr.log("tsdf/mesh", rr.Mesh3D(
                    vertex_positions=verts, triangle_indices=tris, vertex_colors=colors,
                ))
        except queue_mod.Empty:
            break

    print("\nStreaming complete. Press Ctrl+C to exit.", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
