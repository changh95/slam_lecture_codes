#!/usr/bin/env python3
"""
Headless cuVSLAM KITTI tracker with easy_profiler-compatible JSON output.

Based on nvidia-isaac/cuVSLAM examples/kitti/track_kitti.py but without
rerun visualization and with profiler instrumentation.

Usage:
    python3 track_kitti_profiled.py <kitti_sequence_path> [output.json]

Example:
    python3 track_kitti_profiled.py /data/sequences/00 /output/cuvslam_desktop.json
"""

import os
import sys
from numpy import loadtxt, asarray
from PIL import Image
import cuvslam

from py_profiler import enable, dump, block


def main():
    if len(sys.argv) < 2:
        print("Usage: track_kitti_profiled.py <kitti_sequence_path> [output.json]")
        sys.exit(1)

    sequence_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "/output/cuvslam_profiler.json"

    print(f"Sequence: {sequence_path}")
    print(f"Output:   {output_path}")

    # Enable profiler
    enable()

    # Load KITTI calibration and initialize cameras
    with block("SLAM/Init"):
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
    print(f"Processing {num_frames} frames...")

    # Tracking loop
    trajectory = []
    failed = 0
    for frame in range(num_frames):
        with block("SLAM/FrameProcess"):
            # Load stereo images
            with block("SLAM/ImageLoad"):
                images = [
                    asarray(
                        Image.open(
                            os.path.join(sequence_path, f"image_{cam}", f"{frame:0>6}.png")
                        )
                    )
                    for cam in [0, 1]
                ]

            # Call cuVSLAM tracker (this is the binary-opaque GPU SLAM core)
            with block("SLAM/Track"):
                odom_pose_estimate, _ = tracker.track(timestamps[frame], images)

            if odom_pose_estimate.world_from_rig is None:
                failed += 1
                if failed < 5:
                    print(f"  Warning: tracking failed at frame {frame}")
                continue

            # Query pose and observations
            with block("SLAM/PoseQuery"):
                odom_pose = odom_pose_estimate.world_from_rig.pose
                trajectory.append(odom_pose.translation)

            with block("SLAM/Observations"):
                observations = tracker.get_last_observations(0)

            with block("SLAM/Landmarks"):
                landmarks = tracker.get_last_landmarks()

        if frame % 500 == 0:
            print(f"  Frame {frame}/{num_frames}  trajectory len={len(trajectory)}  failed={failed}")

    print(f"Done. Processed {num_frames} frames, {failed} failed.")
    print(f"Final trajectory length: {len(trajectory)}")

    # Dump profiler data
    dump(output_path)


if __name__ == "__main__":
    main()
