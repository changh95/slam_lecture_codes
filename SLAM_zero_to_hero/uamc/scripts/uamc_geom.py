"""UAMC rig geometry and LiDAR message parsing shared by color_mapper.py and export_keyframes.py.

Frames: I = Mid-360 IMU (the frame FAST-LIVO2's Mid-360 LIO poses are in), A = Avia LiDAR,
M = Mid-360 LiDAR, C = Oak-D RGB camera.
"""
import numpy as np

# Avia LiDAR in the Mid-360 IMU frame, estimated on lvi_ghm_set (R: Kabsch of the two IMUs' gyros,
# T: lever arm from both accelerometers) and re-checked on lvi_coex_set_2 to 0.2 deg (NOTES.md).
R_IA = np.array([[0.005266, 0.999983, -0.002657],
                 [-0.706744, 0.005602, 0.707447],
                 [0.707450, -0.001847, 0.706761]])
T_IA = np.array([0.0801, -0.0927, 0.0180])
# Mid-360 LiDAR in its IMU frame (config/ghm_mid360_lio.yaml extrinsic_T/R).
T_IM = np.array([0.011, 0.02329, -0.04412])
# Avia LiDAR -> Oak-D camera, p_c = R_CA p_a + T_CA (upstream avia_lvi.yaml Rcl/Pcl).
R_CA = np.array([[0.00610193, -0.999863, -0.0154172],
                 [-0.00615449, 0.0153796, -0.999863],
                 [0.999962, 0.00619598, -0.0060598]])
T_CA = np.array([0.0194384, 0.104689, -0.0251952])
R_CI = R_CA @ R_IA.T                                    # Mid-360 IMU -> camera
T_CI = T_CA - R_CI @ T_IA
# Oak-D RGB, FAST-LIVO2-ROS2 config/camera_rational.yaml (OpenCV rational model).
K = np.array([[1023.9491577148438, 0, 654.52294921875],
              [0, 1023.9491577148438, 346.38995361328125], [0, 0, 1]])
DIST = np.array([36.78335189819336, -273.5762939453125, 0.0005355329485610127,
                 -9.484394104219973e-05, 711.198974609375, 36.21692657470703,
                 -270.2464904785156, 701.9930419921875])          # k1 k2 p1 p2 k3 k4 k5 k6
CAM_W, CAM_H = 1280, 720

AVIA_TOPIC = "/livox/lidar_3JEDM180010C211"
MID_TOPIC = "/livox/lidar_192_168_1_150"
IMAGE_TOPIC = "/oak/rgb/image_raw"

# Avia CustomMsg as raw CDR, parsed with numpy (rclpy builds one Python object per point).
# After the 4-byte encapsulation: Header{int32 sec, uint32 nsec, string frame_id}, uint64 timebase
# (8-aligned), uint32 point_num, uint8 lidar_id, uint8 rsvd[3],
# sequence<CustomPoint>{uint32 offset_time, float32 x y z, uint8 reflectivity tag line} -> 20-byte stride.
PT = np.dtype([("off", "<u4"), ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
               ("refl", "u1"), ("tag", "u1"), ("line", "u1"), ("pad", "u1")])
# Mid-360 PointCloud2 on this bag: x y z intensity f32, tag line u8, timestamp f64 (ns), 26 bytes.
MPT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4"),
                ("tag", "u1"), ("line", "u1"), ("ts", "<f8")])


def parse_avia(buf, rng):
    """Raw CustomMsg bytes -> (point times, points in the Mid-360 IMU frame), range within rng."""
    o = 4 + 8
    o += 4 + int.from_bytes(buf[o:o + 4], "little")    # frame_id
    o = 4 + ((o - 4 + 7) // 8) * 8
    timebase = int.from_bytes(buf[o:o + 8], "little")
    o += 8 + 4 + 4
    n = int.from_bytes(buf[o:o + 4], "little")
    # the last point has no trailing pad byte; checked against rclpy deserialization
    p = np.frombuffer(buf + b"\0", dtype=PT, count=n, offset=o + 4)
    xyz = np.stack([p["x"], p["y"], p["z"]], 1).astype(np.float64)
    r = np.linalg.norm(xyz, axis=1)
    tags = p["tag"] & 0x30
    keep = ((tags == 0x00) | (tags == 0x10)) & (r >= rng[0]) & (r <= rng[1])
    return timebase * 1e-9 + p["off"][keep] * 1e-9, xyz[keep] @ R_IA.T + T_IA


def parse_mid(m, rng):
    """sensor_msgs/PointCloud2 -> (point times, points in the Mid-360 IMU frame), range within rng."""
    assert m.point_step == MPT.itemsize and [f.name for f in m.fields] == list(MPT.names[:4]) + ["tag", "line", "timestamp"]
    p = np.frombuffer(bytes(m.data), dtype=MPT)
    xyz = np.stack([p["x"], p["y"], p["z"]], 1).astype(np.float64)
    r = np.linalg.norm(xyz, axis=1)
    keep = (r >= rng[0]) & (r <= rng[1])
    return p["ts"][keep] * 1e-9, xyz[keep] + T_IM


def quat_to_R(q):
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)


def interp_pose(tt, tp, tq, t):
    """TUM trajectory (times, xyz, xyzw) at times t: (R, p, q), clamped to its ends."""
    i = np.clip(np.searchsorted(tt, t) - 1, 0, len(tt) - 2)
    f = np.clip((t - tt[i]) / np.maximum(tt[i + 1] - tt[i], 1e-9), 0, 1)[:, None]
    q0, q1 = tq[i], tq[i + 1]
    q1 = np.where((np.sum(q0 * q1, 1) < 0)[:, None], -q1, q1)
    q = q0 * (1 - f) + q1 * f                          # nlerp: poses are 0.1 s apart
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return quat_to_R(q), tp[i] * (1 - f) + tp[i + 1] * f, q
