#!/usr/bin/env python3
"""
cuVSLAM KITTI tracker with rerun visualization.

Runs cuVSLAM on a KITTI sequence and either:
  - Saves a .rrd file that can be opened later with `rerun <file>.rrd`
  - Serves a live web viewer at http://<host>:9090

Usage:
    python3 track_kitti_gui.py <kitti_sequence_path> [--rrd output.rrd] [--web]

Examples:
    # Save .rrd file for later viewing
    python3 track_kitti_gui.py /data/00 --rrd /output/cuvslam_kitti00.rrd

    # Serve live web viewer on port 9090
    python3 track_kitti_gui.py /data/00 --web
"""

import argparse
import os
import sys
from numpy import loadtxt, asarray
from PIL import Image
import rerun as rr
import rerun.blueprint as rrb
import cuvslam


def color_from_id(identifier):
    return [(identifier * 17) % 256, (identifier * 31) % 256, (identifier * 47) % 256]


def main():
    parser = argparse.ArgumentParser(description="cuVSLAM KITTI tracker with GUI")
    parser.add_argument("sequence_path", help="Path to KITTI sequence (e.g. /data/00)")
    parser.add_argument("--rrd", help="Save visualization to .rrd file")
    parser.add_argument("--web", action="store_true", help="Serve web viewer on port 9090")
    parser.add_argument("--spawn", action="store_true", help="Spawn native rerun viewer (needs X11)")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit number of frames (0=all)")
    args = parser.parse_args()

    sequence_path = args.sequence_path
    print(f"Sequence: {sequence_path}")

    # Initialize rerun
    if args.web:
        rr.init("cuvslam_kitti")
        server_uri = rr.serve_grpc(grpc_port=9876)
        print(f"gRPC server URI: {server_uri}")
        # Web viewer connects to host's forwarded port, not container's internal hostname
        web_connect = "rerun+http://localhost:9876/proxy"
        rr.serve_web_viewer(web_port=9090, open_browser=False, connect_to=web_connect)
        print(f"\n=== Web viewer at http://localhost:9090 (data: {web_connect}) ===\n")
    elif args.spawn:
        rr.init("cuvslam_kitti", spawn=True)
    elif args.rrd:
        rr.init("cuvslam_kitti")
        rr.save(args.rrd)
        print(f"Recording to {args.rrd}")
    else:
        print("ERROR: Must specify --rrd, --web, or --spawn")
        sys.exit(1)

    # Setup rerun blueprint
    rr.send_blueprint(rrb.Blueprint(
        rrb.TimePanel(state="collapsed"),
        rrb.Vertical(
            row_shares=[0.6, 0.4],
            contents=[rrb.Spatial3DView(), rrb.Spatial2DView(origin="car/cam0")],
        ),
    ))

    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    rr.log("xyz", rr.Arrows3D(
        vectors=[[50, 0, 0], [0, 50, 0], [0, 0, 50]],
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        labels=["[x]", "[y]", "[z]"],
    ), static=True)

    # Load calibration
    intrinsics = loadtxt(
        os.path.join(sequence_path, "calib.txt"),
        usecols=range(1, 13),
    )[:4].reshape(4, 3, 4)

    first_image = os.path.join(sequence_path, "image_0", "000000.png")
    size = Image.open(first_image).size

    cameras = [cuvslam.Camera(), cuvslam.Camera()]
    for i in [0, 1]:
        cameras[i].size = size
        cameras[i].principal = [intrinsics[i][0][2], intrinsics[i][1][2]]
        cameras[i].focal = [intrinsics[i].diagonal()[0], intrinsics[i].diagonal()[1]]
    cameras[1].rig_from_camera.translation[0] = (
        -intrinsics[1][0][3] / intrinsics[1][0][0]
    )

    cfg = cuvslam.Tracker.OdometryConfig(
        async_sba=False,
        enable_final_landmarks_export=True,
        rectified_stereo_camera=True,
    )
    tracker = cuvslam.Tracker(cuvslam.Rig(cameras), cfg)

    # Load timestamps
    with open(os.path.join(sequence_path, "times.txt")) as f:
        timestamps = [int(10**9 * float(line)) for line in f.readlines()]

    num_frames = len(timestamps)
    if args.max_frames > 0:
        num_frames = min(num_frames, args.max_frames)
    print(f"Processing {num_frames} frames...")

    trajectory = []
    failed = 0

    for frame in range(num_frames):
        # Load stereo images
        images = [
            asarray(Image.open(
                os.path.join(sequence_path, f"image_{cam}", f"{frame:0>6}.png")
            ))
            for cam in [0, 1]
        ]

        # Track
        odom_pose_estimate, _ = tracker.track(timestamps[frame], images)

        if odom_pose_estimate.world_from_rig is None:
            failed += 1
            continue

        odom_pose = odom_pose_estimate.world_from_rig.pose

        # Get visualization data
        observations = tracker.get_last_observations(0)
        landmarks = tracker.get_last_landmarks()
        final_landmarks = tracker.get_final_landmarks()

        observations_uv = [[o.u, o.v] for o in observations]
        observations_colors = [color_from_id(o.id) for o in observations]
        landmark_xyz = [l.coords for l in landmarks]
        landmarks_colors = [color_from_id(l.id) for l in landmarks]
        trajectory.append(odom_pose.translation)

        # Log to rerun
        rr.set_time("frame", sequence=frame)
        rr.log("trajectory", rr.LineStrips3D(trajectory))
        if final_landmarks:
            rr.log("final_landmarks", rr.Points3D(list(final_landmarks.values()), radii=0.1))
        rr.log("car", rr.Transform3D(
            translation=odom_pose.translation,
            quaternion=odom_pose.rotation,
        ))
        rr.log("car/body", rr.Boxes3D(centers=[0, 1.65 / 2, 0], sizes=[[1.6, 1.65, 2.71]]))
        rr.log("car/landmarks_center", rr.Points3D(
            landmark_xyz, radii=0.25, colors=landmarks_colors
        ))
        rr.log("car/landmarks_lines", rr.Arrows3D(
            vectors=landmark_xyz, radii=0.05, colors=landmarks_colors
        ))
        rr.log("car/cam0", rr.Pinhole(
            image_plane_distance=1.68,
            image_from_camera=intrinsics[0][:3, :3],
            width=size[0],
            height=size[1],
        ))
        rr.log("car/cam0/image", rr.Image(images[0]).compress(jpeg_quality=80))
        rr.log("car/cam0/observations", rr.Points2D(
            observations_uv, radii=5, colors=observations_colors
        ))

        if frame % 100 == 0:
            print(f"  Frame {frame}/{num_frames}  failed={failed}")

    print(f"Done. Processed {num_frames} frames, {failed} failed.")
    print(f"Trajectory length: {len(trajectory)}")

    if args.web:
        print("\nWeb viewer still running. Press Ctrl+C to exit.")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
