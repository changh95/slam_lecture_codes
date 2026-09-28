"""Score a trajectory against Hilti 2022's sparse survey-point ground truth.

Hilti 2022 publishes, per sequence, only a handful of timestamps at which the
IMU position was surveyed (exp21_outside_building: 5 points). This script
interpolates the estimated position (TUM file, T_world_imu) at each of those
timestamps, fits the rigid SE(3) transform (no scale) that best aligns the
estimate to the survey frame (Umeyama / Kabsch), and prints the per-point
residuals and their RMSE -- the same alignment the challenge's scorer uses.

usage: eval_survey.py EST_TUM.txt GT.txt
  GT.txt  lines "t x y z qx qy qz qw" (the quaternion is a dummy identity);
          lines starting with '#' are ignored.
"""
import sys
import numpy as np


def load(path):
    rows = [l.split() for l in open(path) if l.strip() and not l.startswith("#")]
    a = np.array([[float(v) for v in r[:4]] for r in rows])
    return a[:, 0], a[:, 1:4]


t_est, p_est = load(sys.argv[1])
t_gt, p_gt = load(sys.argv[2])

keep = (t_gt >= t_est[0]) & (t_gt <= t_est[-1])
if not keep.all():
    print("skipping %d survey point(s) outside the trajectory's time span" % (~keep).sum())
t_gt, p_gt = t_gt[keep], p_gt[keep]
gap = np.array([np.min(np.abs(t_est - t)) for t in t_gt])
p_at = np.stack([np.interp(t_gt, t_est, p_est[:, i]) for i in range(3)], axis=1)

# Kabsch: find R, t minimising sum |R p_at + t - p_gt|^2
mu_e, mu_g = p_at.mean(0), p_gt.mean(0)
H = (p_at - mu_e).T @ (p_gt - mu_g)
U, _, Vt = np.linalg.svd(H)
D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
R = Vt.T @ D @ U.T
t = mu_g - R @ mu_e
err = np.linalg.norm((R @ p_at.T).T + t - p_gt, axis=1)

print("survey points used: %d" % len(t_gt))
for i, (tt, e, g) in enumerate(zip(t_gt, err, gap)):
    print("  #%d  t=%.3f  error %.3f m   (nearest pose %.3f s away)" % (i, tt, e, g))
print("RMSE %.3f m   max %.3f m" % (np.sqrt(np.mean(err ** 2)), err.max()))
d = [np.linalg.norm(p_gt[i] - p_gt[j]) for i in range(len(p_gt)) for j in range(i + 1, len(p_gt))]
print("survey-point spread: max pairwise distance %.1f m" % max(d))
