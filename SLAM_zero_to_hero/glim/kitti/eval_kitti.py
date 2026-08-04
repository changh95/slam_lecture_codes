#!/usr/bin/env python3
"""Absolute trajectory error of a GLIM dump against KITTI odometry ground truth.

GLIM writes traj_lidar.txt / odom_lidar.txt in TUM format
(stamp x y z qx qy qz qw) in the *Velodyne* frame of the first scan.
KITTI poses/NN.txt is T_world_cam0 (3x4, row-major), so ground truth is
transformed with Tr (= T_cam0_velo) from sequences/NN/calib.txt and re-anchored
to the first frame before comparison.

usage: eval_kitti.py <dump_dir> <kitti_sequence_dir> <poses.txt>
"""
import sys

import numpy as np


def load_gt(seq_dir, poses_path):
    poses = np.loadtxt(poses_path).reshape(-1, 3, 4)
    Tw_cam = np.zeros((len(poses), 4, 4))
    Tw_cam[:, :3, :] = poses
    Tw_cam[:, 3, 3] = 1.0

    Tr = np.eye(4)
    for line in open(f"{seq_dir}/calib.txt"):
        if line.startswith("Tr:"):
            Tr[:3, :] = np.array(line.split()[1:], dtype=float).reshape(3, 4)

    Tw_velo = Tw_cam @ Tr                      # T_world_velo
    return (np.linalg.inv(Tw_velo[0]) @ Tw_velo)[:, :3, 3]


def ate(traj_path, gt_xyz, times):
    tr = np.loadtxt(traj_path)
    stamps, xyz = tr[:, 0] - tr[0, 0], tr[:, 1:4]
    nearest = np.abs(stamps[:, None] - times[None, :]).argmin(axis=1)
    err = np.linalg.norm(xyz - gt_xyz[nearest], axis=1)
    path = np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()
    return len(xyz), path, err.mean(), np.sqrt((err ** 2).mean()), err.max()


def main():
    dump_dir, seq_dir, poses_path = sys.argv[1:4]
    gt_xyz = load_gt(seq_dir, poses_path)
    times = np.loadtxt(f"{seq_dir}/times.txt")
    gt_path = np.linalg.norm(np.diff(gt_xyz, axis=0), axis=1).sum()

    print(f"ground truth: {len(gt_xyz)} frames, path length {gt_path:.2f} m")
    print(f"{'file':16s} {'n':>4s} {'path[m]':>8s} {'ATEmean':>8s} {'ATErmse':>8s} {'ATEmax':>8s}")
    for name in ("odom_lidar.txt", "traj_lidar.txt"):
        n, path, mean, rmse, mx = ate(f"{dump_dir}/{name}", gt_xyz, times)
        print(f"{name:16s} {n:4d} {path:8.2f} {mean:8.2f} {rmse:8.2f} {mx:8.2f}")


if __name__ == "__main__":
    main()
