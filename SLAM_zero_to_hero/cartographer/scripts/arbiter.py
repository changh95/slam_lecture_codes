"""Voxel-sharpness arbiter -- generalised from
SLAM_zero_to_hero/fast_lio2/scripts/traj_arbiter.py (same maths, same constants,
same nearest-neighbour pose lookup) so the numbers are directly comparable to the
published FAST-LIO2 reference 0.587 / 0.485 / 0.434.

Aggregate NSCAN consecutive /hesai/pandar scans into the world frame using each
trajectory's poses and count occupied VOX-sized voxels. A wrong trajectory smears
surfaces across MORE voxels. 'identity' (pretend the sensor never moved) = 1.000
is the floor any real SLAM must beat.

usage: arbiter.py START NSCAN VOX  name=/path/traj_tum.txt[:dt] ...
  the optional :dt is added to the scan header stamp before looking up a pose
  (FAST-LIO2 stamps a scan at its END, so it needs :0.0997; Cartographer stamps a
   node at the scan's header stamp because Hilti publishes no per-point 'time'
   field, so it needs :0).
Every trajectory is assumed to be the pose of the IMU body frame in the world
(T_world_imu), which is what both FAST-LIO2 and a Cartographer run with
tracking_frame = imu_sensor_frame produce. The fixed LiDAR<-IMU extrinsic below is
FAST-LIVO2's Hilti-2022 calibration.
"""
import numpy as np, rosbag, sys, os

IN_DTYPE = np.dtype({"names": ["x", "y", "z", "intensity", "timestamp", "ring"],
                     "formats": ["<f4", "<f4", "<f4", "<f4", "<f8", "<u2"],
                     "offsets": [0, 4, 8, 16, 24, 32], "itemsize": 48})


def quat_to_R(qx, qy, qz, qw):
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def load_tum(p):
    out = []
    for l in open(p):
        v = l.split()
        if len(v) != 8:
            continue
        t = float(v[0]); tr = np.array([float(v[1]), float(v[2]), float(v[3])])
        R = quat_to_R(*[float(x) for x in v[4:8]])
        out.append((t, R, tr))
    out.sort(key=lambda r: r[0])
    return out


def pose_at(traj, ts, t):
    i = int(np.argmin(np.abs(ts - t)))
    return traj[i], abs(ts[i] - t)


R_IL = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]])
t_IL = np.array([-0.001, -0.00855, 0.055])

START = int(sys.argv[1]); NSCAN = int(sys.argv[2]); VOX = float(sys.argv[3])
specs = []
for a in sys.argv[4:]:
    name, rest = a.split("=", 1)
    if ":" in rest:
        path, dt = rest.rsplit(":", 1); dt = float(dt)
    else:
        path, dt = rest, 0.0
    specs.append((name, path, dt))

BAG = os.environ.get("ARB_BAG", "/data/exp14_basement_2.bag")
bag = rosbag.Bag(BAG)
scans = []
for i, (topic, msg, t) in enumerate(bag.read_messages(topics=["/hesai/pandar"])):
    if i < START:
        continue
    if i >= START + NSCAN:
        break
    a = np.frombuffer(msg.data, dtype=IN_DTYPE, count=msg.width * msg.height)
    p = np.stack([a["x"], a["y"], a["z"]], axis=1).astype(np.float64)
    r = np.linalg.norm(p, axis=1)
    p = p[(r > 0.5) & (r < 20.0)]
    scans.append((msg.header.stamp.to_sec(), p))
bag.close()
npts = sum(len(p) for _, p in scans)
print("scans %d..%d  points used: %d  voxel %.2f m" % (START, START + NSCAN - 1, npts, VOX))

ident = None


def count(allp):
    P = np.concatenate(allp, axis=0)
    keys = np.floor(P / VOX).astype(np.int64)
    keys -= keys.min(axis=0)
    k = (keys[:, 0] << 42) | (keys[:, 1] << 21) | keys[:, 2]
    return np.unique(k).size


ident = count([p for _, p in scans])
print("  %-12s occupied voxels %8d   ratio-to-identity %.3f" % ("identity", ident, 1.0))

for name, path, dt0 in specs:
    if not os.path.exists(path):
        print("  %-12s MISSING %s" % (name, path)); continue
    traj = load_tum(path)
    ts = np.array([p[0] for p in traj])
    allp = []; maxdt = 0.0
    for tstamp, p in scans:
        (tt, R, tr), d = pose_at(traj, ts, tstamp + dt0)
        maxdt = max(maxdt, d)
        Rw = R @ R_IL; tw = R @ t_IL + tr
        allp.append(p @ Rw.T + tw)
    occ = count(allp)
    print("  %-12s occupied voxels %8d   ratio-to-identity %.3f  (max stamp mismatch %.4f s, dt0 %+.4f, %d poses)"
          % (name, occ, occ / ident, maxdt, dt0, len(traj)))
