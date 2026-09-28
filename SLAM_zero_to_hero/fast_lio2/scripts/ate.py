#!/usr/bin/env python3
"""ATE (translation RMSE) of a TUM trajectory against Hilti 2022 ground truth.

usage: python3 ate.py est_tum.txt gt_tum.txt [max_dt=0.02]

Both files are 't x y z qx qy qz qw'. The Hilti exp14 GT (exp14_basement_2_imu.txt)
is the Alphasense IMU pose, and FAST-LIO's /Odometry is the same IMU body frame, so
no extrinsic is needed. Poses are matched by nearest timestamp (|dt| <= max_dt),
then aligned with a rigid SE(3) Umeyama fit (no scale). numpy only.
"""
import sys
import numpy as np

est = np.loadtxt(sys.argv[1])
gt = np.loadtxt(sys.argv[2])
max_dt = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02

idx = np.clip(np.searchsorted(gt[:, 0], est[:, 0]), 1, len(gt) - 1)
left = np.abs(gt[idx - 1, 0] - est[:, 0]) < np.abs(gt[idx, 0] - est[:, 0])
idx = np.where(left, idx - 1, idx)
ok = np.abs(gt[idx, 0] - est[:, 0]) <= max_dt
P, Q = est[ok, 1:4], gt[idx[ok], 1:4]

mp, mq = P.mean(0), Q.mean(0)
U, _, Vt = np.linalg.svd((Q - mq).T @ (P - mp))
S = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
R = U @ S @ Vt
err = np.linalg.norm((P - mp) @ R.T + mq - Q, axis=1)

print("est poses %d, gt poses %d, matched %d (|dt| <= %.3f s)" % (len(est), len(gt), ok.sum(), max_dt))
print("gt path length: %.3f m" % np.linalg.norm(np.diff(gt[:, 1:4], axis=0), axis=1).sum())
print("ATE RMSE %.4f m | mean %.4f | median %.4f | max %.4f" %
      (np.sqrt((err ** 2).mean()), err.mean(), np.median(err), err.max()))
