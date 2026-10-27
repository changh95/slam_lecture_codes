#!/usr/bin/env python3
"""Online cuVSLAM + nvblox on RGB-D sequences: track, fuse, stream, frame by frame.

Two kinds of input, told apart by what the directory holds:

  - a Humanoid Everyday episode_N: the head-camera RealSense D435 of a Unitree
    G1, with its IMU and legged odometry;
  - a TUM RGB-D freiburg3 sequence (rgb.txt, depth.txt, groundtruth.txt): a
    hand-held Kinect with motion-capture ground truth. fr3/long_office_household
    is a 22 m loop round two office desks, far more ground than any Humanoid
    Everyday episode covers.

For every frame:

  1. cuVSLAM, RGB-D mode, tracks the colour camera. It needs depth registered
     to the image it tracks. TUM stores it that way; Humanoid Everyday's raw
     depth is warped into the colour camera and the grid of holes that warp
     leaves is filled (see fill_warp_holes).
  2. nvblox (through nvblox_torch) integrates the depth in the depth camera's
     own frame -- for Humanoid Everyday the *raw* depth, no warp needed there --
     and the colour image in the colour camera's, both at the pose cuVSLAM just
     returned. On the G1, the pixels that see the robot's own body are left out
     of both: its joint states pose a sphere model of it (see BodyModel).
  3. The pose, the image with the tracked features and, every few frames, the
     parts of the colour mesh, of the occupancy map and of a slice through the
     ESDF that changed (see Stream.mesh, Stream.occupancy, Stream.esdf) are
     streamed, on the same frame timeline, to
     a rerun viewer -- one already open on the host (--stream-url), or, with
     --serve, a web viewer started next to this process on ports 9090 / 9877
     (the URL to open is printed first thing).

The world frame is gravity-aligned with the floor at z = 0. On Humanoid Everyday
the robot IMU anchors the first frame, exactly as he_to_nvblox.py does for depth
ICP; on TUM the first frame starts at its motion-capture pose, whose frame is
already z-up with the floor at 0. Every run is scored against a reference
trajectory cuVSLAM never sees: the legged odometry, or the mocap ground truth.

    cuvslam_nvblox.py <episode_or_sequence_dir> [...] --out /results/cuvslam

Several sequences run one after another, each with its own tracker, its own map
and its own rerun recording -- they are separate recordings with unrelated
coordinate frames, so there is nothing to join. Each output folder gets
traj_cuvslam.tum, the reference (traj_odom.tum, or traj_groundtruth.tum on TUM)
in the same world frame, mesh.ply and stats.json.
"""

import argparse
import atexit
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlparse

# nvblox logs every hash resize and layer allocation through glog at INFO; keep warnings.
os.environ.setdefault("GLOG_minloglevel", "1")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from he_to_nvblox import (DEPTH_H, DEPTH_W, align_depth_to_colour,  # noqa: E402
                          gravity_aligned_first_pose, load_calib, matrix_from_quat_wxyz, odom_pose,
                          read_depth, read_records, transform_from_dict, write_tum)

DEFAULT_CALIB = Path(__file__).resolve().parent.parent / "config" / "humanoid_everyday_d435.json"
MESH_CHUNK_M = 1.0   # the streamed mesh is cut into cubes this big; only changed ones are re-sent
MESH_REFRESH = 16    # ...and one whose triangle count barely moved, every this many mesh calls
# Occupied voxels are coloured by height over this range, metres: floor to a little above
# table height, the band where the objects in both datasets are; higher saturates red.
OCC_Z_RANGE = (-0.1, 1.2)
OCC_MIN_WEIGHT = 1e-4      # TSDF weight a voxel needs to count as observed: nvblox' mesh_integrator_min_weight
ESDF_UNKNOWN = 100.0       # what nvblox returns for an ESDF voxel it has no distance for
G1_BODY_MODEL = Path(__file__).resolve().parent.parent / "config" / "g1_body_model.json"
G1_VISUAL_MESH = G1_BODY_MODEL.with_name("g1_visual_mesh.npz")
# Depth points this close to a body sphere are the body: the fitted camera pose leaves
# 1.6 cm of median residual, and the D435 is noisy at arm's length.
BODY_MARGIN = 0.03
BODY_RANGE = 1.6           # metres; no part of the G1 is further than this from its head camera
ROBOT_JOINTS_EVERY = 3     # the drawn robot's joints move at 10 Hz; its torso every frame


# --------------------------------------------------------------------------- #
# depth for the tracker
# --------------------------------------------------------------------------- #
def fill_warp_holes(depth_mm: np.ndarray, passes: int = 2) -> np.ndarray:
    """Fill the empty rows and columns the depth->colour warp leaves behind.

    The depth camera is ~79 deg wide and the colour camera ~56 deg, so forward
    projection magnifies the depth grid by ~1.57 and leaves every second or third
    row and column empty -- about 60 % of interior pixels are zero.
    he_to_nvblox.fill_pinholes only fills pixels with five valid neighbours,
    which a missing *row* never has. cuVSLAM reads depth around each feature,
    so on the raw warp it sees roughly half the true distance and its whole
    trajectory comes out at ~0.45x scale. Each pass sets a zero pixel to the
    mean of its non-zero 3x3 neighbours (when it has at least two); real
    occlusion shadows are wider than two passes and stay empty.
    """
    d = depth_mm.astype(np.float32)
    for _ in range(passes):
        nb = np.stack([np.roll(np.roll(d, dv, 0), du, 1)
                       for dv in (-1, 0, 1) for du in (-1, 0, 1) if dv or du])
        cnt = (nb > 0).sum(0)
        mean = nb.sum(0) / np.maximum(cnt, 1)
        hole = (d == 0) & (cnt >= 2)
        d[hole] = mean[hole]
    return np.rint(d).astype(np.uint16)


# --------------------------------------------------------------------------- #
# poses
# --------------------------------------------------------------------------- #
def matrix_from_quat_xyzw(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def pose_to_matrix(pose) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = matrix_from_quat_xyzw(pose.rotation)
    T[:3, 3] = pose.translation
    return T


def gravity_error_deg(T_w_c: np.ndarray, R_up: np.ndarray) -> float:
    """Angle between "down" as cuVSLAM has the camera and "down" as an independent
    sensor has it: the robot IMU, or the motion capture.

    The world frame starts gravity-aligned, so a perfect tracker keeps the two
    equal for the whole run; the IMU's roll and pitch are gravity-referenced
    and do not drift. Yaw is unobservable this way, which is why it is a check on
    rotation drift only -- but rotation drift is how these maps fail.
    """
    down = np.array([0.0, 0.0, -1.0])
    g_vslam = T_w_c[:3, :3].T @ down
    g_ref = R_up.T @ down
    return float(np.degrees(np.arccos(np.clip(g_vslam @ g_ref, -1.0, 1.0))))


def verdict(max_gravity_err: float) -> str:
    if max_gravity_err < 5.0:
        return "ok"
    return "marginal" if max_gravity_err < 15.0 else "drifted"


def umeyama_se3(src: np.ndarray, dst: np.ndarray):
    """R, t minimising |R src + t - dst|. No scale: RGB-D odometry is metric."""
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - ms).T @ (dst - md))
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ S @ U.T
    return R, md - R @ ms


def path_length(p: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


# --------------------------------------------------------------------------- #
# occupancy
# --------------------------------------------------------------------------- #
def occupied_voxels(mapper):
    """Centres of the occupied voxels of an nvblox_torch mapper's TSDF, N x 3 on the GPU.

    A voxel is occupied when it is on the inside of the surface (signed distance <= 0)
    and one of its six neighbours is observed free space (distance > 0); both observed
    means weight above OCC_MIN_WEIGHT, the threshold nvblox meshes with. That is the
    one-voxel shell where the distance changes sign. A band such as distance in
    (-voxel, 0] is not enough: the projective TSDF measures distance along the camera
    ray, which on a floor seen at a slant is 1 / cos(incidence) times the true one,
    twice at 60 deg, so the sign change can skip the band and the floor came out full
    of holes. The voxels further
    inside, filled down to the truncation distance, are left out: they are hidden behind
    the shell and three to four times as many. Unobserved voxels are never returned.
    """
    import torch
    from nvblox_torch.mapper import QueryType
    tsdf = mapper.tsdf_layer_view()
    idx = tsdf.get_all_block_indices().cuda().long()
    if not len(idx):
        return torch.zeros((0, 3), device="cuda")
    # Every voxel of every allocated block in one batched lookup: ~3 ms for the 53 k blocks
    # of the TUM office, where get_all_blocks(), a Python list of one tensor per block,
    # takes ~80 ms.
    grid = torch.stack(torch.meshgrid(*[torch.arange(8, device="cuda")] * 3, indexing="ij"), -1)
    centre = ((idx[:, None, None, None] * 8 + grid).float() + 0.5) * tsdf.voxel_size()
    b = mapper.query_layer(QueryType.TSDF, centre.reshape(-1, 3), mapper_id=0).reshape(
        len(idx), 8, 8, 8, 2)        # blocks x 8 x 8 x 8 (x, y, z) x [distance m, weight]
    seen = b[..., 1] > OCC_MIN_WEIGHT
    inside, free = seen & (b[..., 0] <= 0.0), seen & (b[..., 0] > 0.0)

    # free, padded by one voxel with the touching face of each neighbouring block (false
    # where that block is not allocated), so the six-neighbour test crosses block borders
    def key(i):
        i = i + (1 << 20)
        return (i[:, 0] << 42) | (i[:, 1] << 21) | i[:, 2]
    keys, perm = torch.sort(key(idx))
    pad = torch.zeros((len(b), 10, 10, 10), dtype=torch.bool, device=b.device)
    pad[:, 1:9, 1:9, 1:9] = free
    for axis in range(3):
        for step, src, dst in ((-1, 7, 0), (1, 0, 9)):
            off = torch.zeros(3, dtype=torch.long, device=b.device)
            off[axis] = step
            nk = key(idx + off)
            pos = torch.searchsorted(keys, nk).clamp(max=len(keys) - 1)
            hit = keys[pos] == nk
            face = free[perm[pos]].select(axis + 1, src) & hit[:, None, None]
            inner = [slice(None)] + [slice(1, 9)] * 3
            inner[axis + 1] = dst
            pad[tuple(inner)] = face
    near_free = (pad[:, :8, 1:9, 1:9] | pad[:, 2:, 1:9, 1:9] | pad[:, 1:9, :8, 1:9]
                 | pad[:, 1:9, 2:, 1:9] | pad[:, 1:9, 1:9, :8] | pad[:, 1:9, 1:9, 2:])
    return centre[inside & near_free]


def esdf_slice(mapper, height):
    """The voxels of nvblox's 3D ESDF in the horizontal layer that holds `height`, where
    the ESDF has a distance: centres N x 3 and signed distances N, metres, on the GPU.

    The ESDF must be up to date (mapper.update_esdf()). Every allocated TSDF block the
    layer crosses is queried in one batch, as in occupied_voxels.
    """
    import torch
    from nvblox_torch.mapper import QueryType
    tsdf = mapper.tsdf_layer_view()
    v = tsdf.voxel_size()
    k = int(np.floor(height / v))           # the voxel layer, and below its block
    idx = tsdf.get_all_block_indices().cuda().long()
    idx = idx[idx[:, 2] == k // 8]
    if not len(idx):
        return torch.zeros((0, 3), device="cuda"), torch.zeros(0, device="cuda")
    grid = torch.stack(torch.meshgrid(*[torch.arange(8, device="cuda")] * 2, indexing="ij"), -1)
    xy = ((idx[:, None, None, :2] * 8 + grid).float() + 0.5).reshape(-1, 2) * v
    centre = torch.cat([xy, torch.full_like(xy[:, :1], (k + 0.5) * v)], 1)
    # An ESDF query is x, y, z and a radius to subtract; N x 3 fails ("Inputs do not have
    # the required sizes").
    d = mapper.query_layer(QueryType.ESDF, torch.cat([centre, torch.zeros_like(xy[:, :1])], 1),
                           mapper_id=0)[:, 0]
    known = d < ESDF_UNKNOWN
    return centre[known], d[known]


# --------------------------------------------------------------------------- #
# the robot's own body
# --------------------------------------------------------------------------- #
def matrix_from_rpy(r: float, p: float, y: float) -> np.ndarray:
    """URDF's fixed-axis roll, pitch, yaw: Rz(y) Ry(p) Rx(r)."""
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def matrix_from_axis_angle(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


class BodyModel:
    """The Unitree G1's own body as spheres, posed by a Humanoid Everyday frame's joint
    states: config/g1_body_model.json, written by make_g1_body_model.py, which also
    writes the URDF's meshes, simplified, for the viewer to draw the robot with.

    The G1's head camera looks 56 deg down, and the depth camera sees more than the
    colour camera does: most depth images show the robot's own hands and thighs, and
    an arm whenever it reaches for something. Integrated, they leave a trail of blobs
    along the path and an arm-shaped wall wherever the robot touched something.
    """

    def __init__(self, model: dict, visual: dict = None):
        self.state_joints = model["state_joints"]   # states key -> joint names, in order
        self.joints = []                             # parents before children
        for j in model["joints"]:
            T = np.eye(4)
            T[:3, :3] = matrix_from_rpy(*j["rpy"])
            T[:3, 3] = j["xyz"]
            self.joints.append((j["name"], j["parent"], j["child"], T, j["axis"]))
        self.link_spheres = {link: np.asarray(s, dtype=np.float64) for link, s in model["spheres"].items()}
        self.T_torso_depth = transform_from_dict(model["torso_to_depth_camera"])
        self.visual = visual or {}   # link -> (vertices, triangles, rgb), for drawing

    @classmethod
    def load(cls, path: Path, visual_path: Path = None) -> "BodyModel":
        visual = {}
        if visual_path is not None and visual_path.exists():
            with np.load(visual_path) as z:
                for key in z.files:
                    if key.endswith("/vertices"):
                        link = key.rsplit("/", 1)[0]
                        visual[link] = (z[key], z[f"{link}/triangles"], z[f"{link}/colour"])
        return cls(json.loads(path.read_text()), visual)

    def link_poses(self, rec: dict) -> dict:
        """T_torso_link of every link. The head camera is fixed to the torso, so this is
        all the depth mask needs, whatever the waist joints do."""
        q = {}
        for key, names in self.state_joints.items():
            q.update(zip(names, rec["states"][key]))
        T = {self.joints[0][1]: np.eye(4)}            # the root, the pelvis
        for name, parent, child, T_origin, axis in self.joints:
            M = T[parent] @ T_origin
            if axis is not None:
                M[:3, :3] = M[:3, :3] @ matrix_from_axis_angle(axis, q.get(name, 0.0))
            T[child] = M
        T_torso_root = np.linalg.inv(T["torso_link"])
        return {link: T_torso_root @ M for link, M in T.items()}

    def spheres(self, poses: dict, T_frame_torso: np.ndarray):
        """Centres (N x 3) and radii (N) of the spheres at link_poses() poses, in the given frame."""
        centres, radii = [], []
        for link, s in self.link_spheres.items():
            M = T_frame_torso @ poses[link]
            centres.append(s[:, :3] @ M[:3, :3].T + M[:3, 3])
            radii.append(s[:, 3])
        return np.concatenate(centres), np.concatenate(radii)


def camera_rays(intr: dict, W: int, H: int):
    """H x W x 3 rays on the GPU, z = 1: a depth image times them is its points."""
    import torch
    v, u = torch.meshgrid(torch.arange(H, device="cuda"), torch.arange(W, device="cuda"),
                          indexing="ij")
    return torch.stack([(u - intr["cx"]) / intr["fx"], (v - intr["cy"]) / intr["fy"],
                        torch.ones_like(u, dtype=torch.float32)], -1)


def body_pixels(depth, rays, centres, radii):
    """The pixels of a depth image (metres, H x W on the GPU) whose point is within
    BODY_MARGIN of a body sphere (centres N x 3, radii N, in that camera's frame)."""
    import torch
    near = (depth > 0) & (depth < BODY_RANGE)
    idx = near.nonzero(as_tuple=True)
    p = rays[idx] * depth[idx][:, None]
    hit = torch.zeros(len(p), dtype=torch.bool, device=depth.device)
    for s in range(0, len(p), 1 << 16):     # ~150 k near points x ~200 spheres, in parts
        hit[s:s + (1 << 16)] = (torch.cdist(p[s:s + (1 << 16)], centres) < radii + BODY_MARGIN).any(1)
    out = torch.zeros_like(near)
    out[idx] = hit
    return out


# --------------------------------------------------------------------------- #
# rerun
# --------------------------------------------------------------------------- #
def viewer_reachable(url: str, timeout: float = 0.5) -> bool:
    """rerun's connect_grpc never fails on a missing viewer, it just retries, so probe first."""
    u = urlparse(url.replace("rerun+", "", 1))
    try:
        with socket.create_connection((u.hostname, u.port or 9876), timeout=timeout):
            return True
    except OSError:
        return False


def serve(args):
    """Start `rerun --serve-web` next to this process and point every episode's stream at it.

    A server process rather than rr.serve_grpc(): that one lives only as long as the
    recording it is attached to, and shows up in the viewer as an extra, empty recording.
    The server buffers what it is sent, so a browser opened late still gets it all.
    """
    import rerun_cli
    for port in (args.grpc_port, args.web_port):
        if viewer_reachable(f"http://127.0.0.1:{port}"):
            sys.exit(f"[serve] port {port} is already in use (another viewer?); "
                     "pick another with --grpc-port / --web-port")
    server = subprocess.Popen(
        [str(Path(rerun_cli.__file__).parent / "rerun"), "--serve-web",
         "--port", str(args.grpc_port), "--web-viewer-port", str(args.web_port),
         "--server-memory-limit", "25%"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    atexit.register(server.terminate)
    args.stream_url = f"rerun+http://127.0.0.1:{args.grpc_port}/proxy"
    for _ in range(100):
        if viewer_reachable(args.stream_url) and viewer_reachable(f"http://127.0.0.1:{args.web_port}"):
            break
        time.sleep(0.1)
    else:
        sys.exit("[serve] the rerun web viewer did not come up")
    args.viewer_url = (f"http://localhost:{args.web_port}/"
                       f"?url=rerun+http://localhost:{args.grpc_port}/proxy")
    print(f"\n  open  {args.viewer_url}\n", flush=True)
    return server


class Stream:
    """Live rerun output for one episode. All methods are no-ops when disabled."""

    def __init__(self, args, seq):
        self.on = False
        self.images = not args.no_stream_images
        self.occupied = not args.no_occupancy
        self.esdf_height = None if args.no_esdf else args.esdf_height
        self.ref_path = f"world/trajectory/{seq.ref[1]}"
        name, colour_intr = seq.name, seq.colour_intr
        if args.no_stream and not args.save_rrd:
            return
        import rerun as rr
        import rerun.blueprint as rrb
        import torch
        self.rr, self.torch = rr, torch
        # A recording object of its own, not rr.init's global one: if the viewer goes away
        # mid-episode, the next rr.init raises "gRPC connection severed" and the global
        # recording stays broken for every episode after it.
        rid = name.replace("/", "_")
        rec = rr.RecordingStream("cuvslam_nvblox", recording_id=rid)
        if args.save_rrd:
            rec.save(str(args.save_rrd_dir / f"{rid}.rrd"))
        elif viewer_reachable(args.stream_url):
            rec.connect_grpc(args.stream_url)
            print(f"[stream] {name} -> {args.viewer_url}", flush=True)
        else:
            print(f"[stream] no rerun viewer at {args.stream_url}; running without "
                  "streaming (start `rerun` on the host, pass --serve to view it in a "
                  "browser, or --no-stream)", flush=True)
            return
        self.rec = rec
        self.on = True
        rec.send_recording_name(name)  # the viewer lists recordings by start time otherwise
        # The occupied voxels sit exactly where the mesh is, so each gets a tab of its own;
        # the ESDF slice goes over the mesh, as in nvblox_torch's ESDF example.
        def view(name, hide):
            return rrb.Spatial3DView(name=name, origin="world",
                                     contents=["+ $origin/**"] + [f"- /world/{h}/**" for h in hide])
        views = [view("nvblox mesh + trajectory", ("occupancy", "esdf"))]
        if self.occupied:
            views.insert(0, view("occupancy voxels + trajectory", ("mesh", "esdf")))
        if self.esdf_height is not None:
            views.append(view(f"ESDF slice at z = {self.esdf_height:g} m + mesh", ("occupancy",)))
        map_view = views[0] if len(views) == 1 else rrb.Tabs(*views)
        rec.send_blueprint(rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(rrb.Spatial2DView(name="colour + cuVSLAM features",
                                               origin="world/camera/image"),
                             rrb.TimeSeriesView(name="features / gravity error (deg)", origin="plots")),
                map_view,
                column_shares=[1, 2]),
            rrb.TimePanel(state="collapsed")), make_active=True)
        rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rec.log("world/camera/image", rr.Pinhole(
            focal_length=[colour_intr["fx"], colour_intr["fy"]],
            principal_point=[colour_intr["cx"], colour_intr["cy"]],
            resolution=list(seq.size), image_plane_distance=0.2), static=True)
        self.track_colours = {}
        # per layer: chunk entity path -> (fingerprint, element count, call) last sent
        self.chunks = {"mesh": {}, "occupancy": {}, "esdf": {}}
        self.calls = {"mesh": 0, "occupancy": 0, "esdf": 0}
        self.luts = {}
        self.robot_calls = 0

    def frame(self, i, T_w_c, rgb, observations, traj, ref_traj, grav_err):
        if not self.on:
            return
        rr, rec = self.rr, self.rec
        rec.set_time("frame", sequence=i)
        rec.log("world/camera", rr.Transform3D(translation=T_w_c[:3, 3], mat3x3=T_w_c[:3, :3]))
        rec.log("world/trajectory/cuvslam", rr.LineStrips3D([traj], colors=[[255, 80, 40]], radii=0.006))
        rec.log(self.ref_path, rr.LineStrips3D([ref_traj], colors=[[60, 160, 255]], radii=0.004))
        rec.log("plots/features", rr.Scalars(len(observations)))
        rec.log("plots/gravity_error_deg", rr.Scalars(grav_err))
        if self.images:
            rec.log("world/camera/image", rr.Image(rgb).compress(jpeg_quality=80))
            if observations:
                for o in observations:
                    if o.id not in self.track_colours:
                        self.track_colours[o.id] = np.random.randint(64, 256, 3)
                rec.log("world/camera/image/features", rr.Points2D(
                    [[o.u, o.v] for o in observations], radii=3,
                    colors=[self.track_colours[o.id] for o in observations]))

    def robot(self, model, T_w_torso, poses):
        """Draw the G1 from its URDF meshes, posed as in this frame: the meshes once per
        recording, the torso every frame, and the links on it every ROBOT_JOINTS_EVERY
        frames. A rerun row costs ~400 bytes, so 50 link poses every frame added 12 MB
        to a 64 MB recording."""
        if not (self.on and model.visual):
            return
        rr, rec = self.rr, self.rec
        if self.robot_calls == 0:
            for link, (v, tri, rgb) in model.visual.items():
                n = np.zeros_like(v)
                for k in range(3):   # area-weighted vertex normals, for shading
                    np.add.at(n, tri[:, k], np.cross(v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]]))
                n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
                rec.log(f"world/robot/{link}", rr.Mesh3D(vertex_positions=v, triangle_indices=tri,
                                                         vertex_normals=n, albedo_factor=rgb), static=True)
        rec.log("world/robot", rr.Transform3D(translation=T_w_torso[:3, 3], mat3x3=T_w_torso[:3, :3]))
        if self.robot_calls % ROBOT_JOINTS_EVERY == 0:
            for link in model.visual:
                T = poses[link]
                rec.log(f"world/robot/{link}", rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
        self.robot_calls += 1

    def mesh(self, mesh, final=False):
        """Log the parts of the mesh that changed, at the current frame, so the map grows
        on the timeline along with the trajectory.

        The whole mesh every --mesh-every frames would do that too, but it is ~80 % of the
        stream, 1.6 GB for the 8 Humanoid Everyday episodes: more than a browser tab (4 GB
        wasm) keeps, so it drops the oldest data and a tab opened late shows little more
        than each episode's final mesh. So the mesh is cut into MESH_CHUNK_M cubes, and a
        cube is sent when it is new or its triangle count moved by more than 5 %. Every
        surface in view is re-meshed on every call, so a cube whose surface only shifted a
        little waits until its last copy is MESH_REFRESH calls old; final=True sends
        whatever changed.

        The cutting happens on the GPU, where nvblox keeps the mesh, and only the cubes
        to send come back: the office map of TUM fr3/long_office_household grows to ~2 M
        triangles, and sorting all of them on the CPU every call took the frame rate from
        45 down to 16 fps by the end.
        """
        if not self.on:
            return
        v, tri, colours = mesh.vertices(), mesh.triangles().long(), mesh.vertex_colors()
        # Whether a cube changed at all: the sums of its triangle centres and colours.
        order, send = self._changed_cubes("mesh", v[tri].mean(dim=1),
                                          colours[tri].sum(dim=(1, 2), dtype=self.torch.float64),
                                          final)
        if send:
            tri_send = tri[order].cpu().numpy()
            v, colours = v.cpu().numpy(), colours.cpu().numpy()
            ends = np.cumsum([n for _, n in send])
            for (path, _), t in zip(send, np.split(tri_send, ends[:-1])):
                used, local = np.unique(t, return_inverse=True)
                self.rec.log(path, self.rr.Mesh3D(
                    vertex_positions=v[used], triangle_indices=local.reshape(-1, 3),
                    vertex_colors=colours[used]))

    def occupancy(self, mapper, final=False):
        """Log the occupied voxels of the TSDF as solid cubes coloured by height, at the
        current frame, in MESH_CHUNK_M cubes exactly as the mesh is (see mesh).

        At the TSDF's own resolution: 1.5 cm by default, ~10^5 voxels on a Humanoid
        Everyday episode and ~10^6 on the TUM office.
        """
        if not (self.on and self.occupied):
            return
        centre = occupied_voxels(mapper)
        # Voxel centres sit on a grid, so moves inside a cube can cancel in their plain sum;
        # the sum of squares as well tells those apart.
        order, send = self._changed_cubes("occupancy", centre,
                                          (centre.double() ** 2).sum(dim=1), final)
        if send:
            c = centre[order]
            colours = self._colours("turbo", c[:, 2], *OCC_Z_RANGE)
            c, colours = c.cpu().numpy(), colours.cpu().numpy()
            half = 0.5 * mapper.tsdf_layer_view().voxel_size()
            ends = np.cumsum([n for _, n in send])
            for (path, _), p, col in zip(send, np.split(c, ends[:-1]), np.split(colours, ends[:-1])):
                self.rec.log(path, self.rr.Boxes3D(centers=p, half_sizes=[[half, half, half]],
                                                   colors=col, fill_mode="solid"))

    def esdf(self, mapper, final=False):
        """Log a horizontal slice of nvblox's 3D ESDF at --esdf-height, at the current frame:
        one flat tile per voxel, coloured by its distance to the nearest surface, in
        MESH_CHUNK_M cubes as the mesh is (see mesh).

        The colours run from 0 m (and inside surfaces) to the slice height, the distance
        down to the floor: the floor is a surface too, so wherever it is mapped below the
        slice nothing in the slice is further from a surface than that. What the slice
        shows is the obstacles that come closer than the floor.
        """
        if not (self.on and self.esdf_height is not None):
            return
        mapper.update_esdf()
        centre, dist = esdf_slice(mapper, self.esdf_height)
        order, send = self._changed_cubes("esdf", centre, dist, final)
        if send:
            c = centre[order]
            colours = self._colours("plasma", dist[order], 0.0, self.esdf_height)
            c, colours = c.cpu().numpy(), colours.cpu().numpy()
            half = 0.5 * mapper.tsdf_layer_view().voxel_size()
            ends = np.cumsum([n for _, n in send])
            for (path, _), p, col in zip(send, np.split(c, ends[:-1]), np.split(colours, ends[:-1])):
                self.rec.log(path, self.rr.Boxes3D(centers=p, half_sizes=[[half, half, 0.001]],
                                                   colors=col, fill_mode="solid"))

    def _colours(self, cmap, x, lo, hi):
        """RGB uint8 on the GPU of matplotlib's colour map cmap at x, scaled from [lo, hi]."""
        torch = self.torch
        if cmap not in self.luts:
            from matplotlib import colormaps
            self.luts[cmap] = torch.as_tensor(colormaps[cmap](np.linspace(0.0, 1.0, 256))[:, :3] * 255,
                                              dtype=torch.uint8, device=x.device)
        return self.luts[cmap][((x - lo) / (hi - lo) * 255).clamp(0, 255).long()]

    def _changed_cubes(self, layer, centre, value, final):
        """Cut a layer's elements into MESH_CHUNK_M cubes on the GPU and pick the cubes to
        re-send: the new ones, those whose element count moved by more than 5 %, and those
        whose content changed at all once their last copy is MESH_REFRESH calls old, or on
        the final call. Cubes the layer no longer has are cleared here.

        centre: N x 3 element positions; value: N numbers whose per-cube sum, with that of
        the positions, fingerprints the cube's content. Returns the indices of the elements
        to send, grouped by cube, and the (entity path, element count) of each cube in the
        same order.
        """
        torch = self.torch
        chunks = self.chunks[layer]
        self.calls[layer] += 1
        calls = self.calls[layer]
        seen, send = set(), []
        order = torch.zeros(0, dtype=torch.long, device=centre.device)
        if len(centre):
            k = torch.floor(centre / MESH_CHUNK_M).long() + (1 << 20)
            keys, cube, count = torch.unique((k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2],
                                             return_inverse=True, return_counts=True)
            fp = torch.zeros(len(keys), 4, dtype=torch.float64, device=centre.device)
            fp.index_add_(0, cube, torch.cat([centre.double(), value.double()[:, None]], 1))
            keys, count, fp = keys.cpu().numpy(), count.cpu().numpy(), fp.cpu().numpy()
            ijk = np.stack([keys >> 42, (keys >> 21) & 0x1FFFFF, keys & 0x1FFFFF], 1) - (1 << 20)
            picked = []
            for c, label in enumerate(ijk):
                path = "world/{}/{}_{}_{}".format(layer, *label)
                seen.add(path)
                last = chunks.get(path)
                # tolerances: the GPU sums in no fixed order, so equal cubes differ in the last bits
                if last is not None and (np.allclose(fp[c], last[0], rtol=1e-9, atol=1e-6) or (
                        abs(count[c] - last[1]) <= 0.05 * last[1]
                        and calls - last[2] < MESH_REFRESH and not final)):
                    continue
                picked.append(c)
                send.append((path, int(count[c])))
                chunks[path] = (fp[c], count[c], calls)
            if picked:
                pick = torch.nonzero(torch.isin(cube, torch.as_tensor(
                    picked, device=cube.device))).squeeze(1)
                order = pick[torch.argsort(cube[pick], stable=True)]
        for path in chunks.keys() - seen:
            self.rec.log(path, self.rr.Clear(recursive=False))
            del chunks[path]
        return order, send


# --------------------------------------------------------------------------- #
# sequences
# --------------------------------------------------------------------------- #
class Frame(NamedTuple):
    time: float               # seconds
    rgb: np.ndarray           # H x W x 3 uint8, writable (torch wants that)
    depth_track: np.ndarray   # uint16, registered to rgb, in 1 / depth_scale m: for cuVSLAM
    depth_map: np.ndarray     # float32 metres, in the depth camera's own frame: for nvblox
    R_up: np.ndarray          # camera orientation in a z-up frame, from a sensor cuVSLAM never sees
    T_ref: np.ndarray         # camera pose on the reference trajectory, in that trajectory's frame
    body: dict = None         # the robot's link poses in its torso frame (BodyModel.link_poses)


class HumanoidEpisode:
    """A Humanoid Everyday <task>/episode_N: head-camera D435, robot IMU, legged odometry."""
    ref = ("leg odometry", "leg_odometry", "traj_odom.tum")   # label, rerun entity, file
    depth_scale = 1000.0
    size = (DEPTH_W, DEPTH_H)

    def __init__(self, path: Path, args):
        self.path, self.depth_max = path, args.depth_max
        self.name = f"{path.parent.name}/{path.name}"
        self.out = Path(path.parent.name) / path.name
        calib = load_calib(args.calib, args.robot)
        self.depth_intr, self.colour_intr = calib["depth_intrinsics"], calib["colour_intrinsics"]
        self.T_colour_depth = transform_from_dict(calib["depth_to_colour"])
        self.T_pelvis_colour = transform_from_dict(calib["pelvis_to_colour_camera"])
        self.recs = read_records(path)[args.start:]
        if args.frames > 0:
            self.recs = self.recs[:args.frames]
        self.body = None
        if not args.no_body_mask:
            if args.robot == "g1":
                self.body = BodyModel.load(G1_BODY_MODEL, G1_VISUAL_MESH)
                self.T_depth_torso = np.linalg.inv(self.body.T_torso_depth)
            else:
                print(f"[{self.name}] no body model for the {args.robot}: its own arms get mapped",
                      flush=True)

    def __len__(self):
        return len(self.recs)

    def frames(self):
        for rec in self.recs:
            rgb = np.array(Image.open(self.path / rec["image"]).convert("RGB"))
            raw = read_depth(self.path, rec).copy()
            raw[raw > self.depth_max * 1000] = 0
            depth_c = fill_warp_holes(align_depth_to_colour(raw, self.depth_intr, self.colour_intr,
                                                             self.T_colour_depth))
            yield Frame(rec["time"], rgb, depth_c, raw.astype(np.float32) / 1000.0,
                        matrix_from_quat_wxyz(rec["states"]["imu"]["quaternion"])
                        @ self.T_pelvis_colour[:3, :3],
                        odom_pose(rec) @ self.T_pelvis_colour,
                        self.body.link_poses(rec) if self.body else None)

    def first_pose(self, f: Frame) -> np.ndarray:
        return gravity_aligned_first_pose(self.recs[0], self.T_pelvis_colour, f.depth_track,
                                          self.colour_intr, self.depth_max)


class TumSequence:
    """A TUM RGB-D sequence directory: hand-held Kinect, motion-capture ground truth.

    freiburg3 only: its images are stored undistorted, while fr1 and fr2 would
    need their lens distortion modelled. Depth is stored registered to the colour
    image, so one camera model serves both.
    """
    ref = ("mocap ground truth", "ground_truth", "traj_groundtruth.tum")
    depth_scale = 5000.0
    size = (640, 480)
    FR3 = {"fx": 535.4, "fy": 539.2, "cx": 320.1, "cy": 247.6}   # TUM's calibration page

    def __init__(self, path: Path, args):
        if "freiburg3" not in path.name:
            raise ValueError(f"{path.name}: only freiburg3 sequences are supported")
        self.path, self.depth_max = path, args.depth_max
        self.name = self.out = path.name
        self.colour_intr = self.depth_intr = self.FR3
        self.T_colour_depth = np.eye(4)

        def listing(name):
            rows = [ln.split() for ln in (path / name).read_text().splitlines()
                    if ln.strip() and not ln.startswith("#")]
            return np.array([float(r[0]) for r in rows]), [r[1] for r in rows]

        t_rgb, rgb = listing("rgb.txt")
        t_depth, depth = listing("depth.txt")
        gt = np.loadtxt(path / "groundtruth.txt")

        def nearest(t, ts):   # index into the sorted ts of the stamp nearest each t, and the gap
            hi = np.clip(np.searchsorted(ts, t), 0, len(ts) - 1)
            lo = np.maximum(hi - 1, 0)
            j = np.where(np.abs(ts[lo] - t) <= np.abs(ts[hi] - t), lo, hi)
            return j, np.abs(ts[j] - t)

        # Pair each colour image with a depth image and a ground-truth sample at most
        # 20 ms away, as TUM's associate.py does, and use each depth image once. Colour
        # images left without either are dropped: ~4 % on fr3/long_office_household,
        # mostly where the Kinect skipped a depth frame, plus two before the motion
        # capture starts.
        jd, dd = nearest(t_rgb, t_depth)
        jg, dg = nearest(t_rgb, gt[:, 0])
        keep = np.flatnonzero((dd <= 0.02) & (dg <= 0.02))
        keep = keep[np.argsort(dd[keep], kind="stable")]
        keep = np.sort(keep[np.unique(jd[keep], return_index=True)[1]])[args.start:]
        if args.frames > 0:
            keep = keep[:args.frames]
        self.items = [(t_rgb[k], rgb[k], depth[jd[k]], gt[jg[k], 1:]) for k in keep]

    def __len__(self):
        return len(self.items)

    def frames(self):
        for t, rgb_file, depth_file, (x, y, z, *q) in self.items:
            rgb = np.array(Image.open(self.path / rgb_file).convert("RGB"))
            raw = np.array(Image.open(self.path / depth_file), dtype=np.uint16)
            raw[raw > self.depth_max * self.depth_scale] = 0
            T = np.eye(4)
            T[:3, :3] = matrix_from_quat_xyzw(q)
            T[:3, 3] = x, y, z
            yield Frame(t, rgb, raw, raw.astype(np.float32) / self.depth_scale, T[:3, :3], T)

    def first_pose(self, f: Frame) -> np.ndarray:
        # Start where the motion capture has the camera: its frame is z-up with the floor
        # at z = 0, and fr3 has no accelerometer data to level the first frame with.
        return f.T_ref


def open_sequence(path: Path, args):
    if (path / "rgb.txt").is_file() and (path / "depth.txt").is_file():
        return TumSequence(path, args)
    return HumanoidEpisode(path, args)


# --------------------------------------------------------------------------- #
# one episode
# --------------------------------------------------------------------------- #
def run_episode(seq, out: Path, args) -> dict:
    import cuvslam
    import torch
    from nvblox_torch.mapper import Mapper
    from nvblox_torch.mapper_params import MapperParams, ProjectiveIntegratorParams
    from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType
    from nvblox_torch.sensor import Sensor

    name = seq.name
    out.mkdir(parents=True, exist_ok=True)
    depth_intr, colour_intr = seq.depth_intr, seq.colour_intr
    W, H = seq.size
    ref_label, _, ref_file = seq.ref
    print(f"[{name}] {len(seq)} frames", flush=True)
    # Before the tracker and the mapper: building them takes ~25 s on the first episode
    # (CUDA start-up), and the viewer should show the layout during that, not nothing.
    stream = Stream(args, seq)

    # cuVSLAM tracks the colour camera; depth is registered to it (camera 0).
    cam = cuvslam.Camera(size=(W, H),
                         principal=(colour_intr["cx"], colour_intr["cy"]),
                         focal=(colour_intr["fx"], colour_intr["fy"]))
    odom_cfg = cuvslam.Tracker.OdometryConfig(
        odometry_mode=cuvslam.Tracker.OdometryMode.RGBD,
        async_sba=False,  # deterministic: the same episode gives the same trajectory
        enable_observations_export=True,
        rgbd_settings=cuvslam.Tracker.OdometryRGBDSettings(depth_scale_factor=seq.depth_scale,
                                                           depth_camera_id=0))
    tracker = cuvslam.Tracker(cuvslam.Rig([cam]), odom_cfg,
                              cuvslam.Tracker.SlamConfig() if args.slam else None)

    params = MapperParams()
    pi = ProjectiveIntegratorParams()
    pi.projective_integrator_max_integration_distance_m = args.max_integration
    params.set_projective_integrator_params(pi)
    mapper = Mapper(voxel_sizes_m=args.voxel, integrator_types=ProjectiveIntegratorType.TSDF,
                    mapper_parameters=params)
    depth_sensor = Sensor.from_camera(depth_intr["fx"], depth_intr["fy"], depth_intr["cx"],
                                      depth_intr["cy"], W, H)
    colour_sensor = Sensor.from_camera(colour_intr["fx"], colour_intr["fy"], colour_intr["cx"],
                                       colour_intr["cy"], W, H)

    depth_rays, colour_rays = camera_rays(depth_intr, W, H), camera_rays(colour_intr, W, H)
    R_cd = torch.from_numpy(seq.T_colour_depth[:3, :3]).float().cuda()
    t_cd = torch.from_numpy(seq.T_colour_depth[:3, 3]).float().cuda()

    T0 = None                       # world <- cuVSLAM start frame (gravity alignment)
    T_ref0_inv = None
    times, poses, ref, n_obs, grav = [], [], [], [], []
    body_share = torch.zeros((), device="cuda")    # summed share of depth pixels masked
    lost = 0
    t_start = time.time()
    for i, f in enumerate(seq.frames()):
        if T0 is None:
            T0 = seq.first_pose(f)

        est, slam_pose = tracker.track(int(round(f.time * 1e9)),
                                       images=[f.rgb], depths=[f.depth_track])
        pose = slam_pose if (args.slam and slam_pose is not None) else (
            est.world_from_rig.pose if est.world_from_rig is not None else None)
        if pose is None:
            lost += 1
            continue
        T_w_c = T0 @ pose_to_matrix(pose)

        # The reference trajectory, expressed in the same world: anchored to start where
        # cuVSLAM starts, so the two trajectories can be overlaid without fitting.
        if T_ref0_inv is None:
            T_ref0_inv = np.linalg.inv(f.T_ref)
        ref.append((T0 @ T_ref0_inv @ f.T_ref)[:3, 3])

        depth = torch.from_numpy(f.depth_map).cuda()
        colour_mask = None
        if f.body is not None:
            # Leave the robot out of the map: no depth where it is, and no colour from it
            # (the colour camera's own depth is the registered one cuVSLAM tracks with).
            centres, radii = (torch.from_numpy(a).float().cuda()
                              for a in seq.body.spheres(f.body, seq.T_depth_torso))
            body = body_pixels(depth, depth_rays, centres, radii)
            depth[body] = 0.0
            body_share += body.float().mean()
            colour_mask = (~body_pixels(torch.from_numpy(f.depth_track).cuda().float() / seq.depth_scale,
                                        colour_rays, centres @ R_cd.T + t_cd, radii)).to(torch.uint8)
        mapper.add_depth_frame(depth, torch.from_numpy(T_w_c @ seq.T_colour_depth).float(), depth_sensor)
        mapper.add_color_frame(torch.from_numpy(f.rgb).cuda(), torch.from_numpy(T_w_c).float(),
                               colour_sensor, colour_mask)

        times.append(f.time)
        poses.append(T_w_c)
        grav.append(gravity_error_deg(T_w_c, f.R_up))
        obs = tracker.get_last_observations(0)
        n_obs.append(len(obs))
        stream.frame(i, T_w_c, f.rgb, obs, np.array([T[:3, 3] for T in poses]), np.array(ref),
                     grav[-1])
        if f.body is not None:
            stream.robot(seq.body, T_w_c @ seq.T_colour_depth @ seq.T_depth_torso, f.body)
        if stream.on and i % args.mesh_every == 0:
            mapper.update_color_mesh()
            stream.mesh(mapper.get_color_mesh())
            stream.occupancy(mapper)
            stream.esdf(mapper)
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(seq)}  features {len(obs)}  "
                  f"{(i+1)/(time.time()-t_start):.1f} fps", flush=True)
    wall = time.time() - t_start

    mapper.update_color_mesh()
    mesh = mapper.get_color_mesh()
    stream.mesh(mesh, final=True)
    stream.occupancy(mapper, final=True)
    stream.esdf(mapper, final=True)
    mesh.save(str(out / "mesh.ply"))
    write_tum(out / "traj_cuvslam.tum", times, poses)
    ref_T = []
    for p in ref:
        T = np.eye(4)
        T[:3, 3] = p
        ref_T.append(T)
    write_tum(out / ref_file, times, ref_T)

    p = np.array([T[:3, 3] for T in poses]) if poses else np.zeros((0, 3))
    q = np.array(ref) if ref else np.zeros((0, 3))
    stats = {
        "episode": name, "frames": len(seq), "tracked": len(poses),
        "tracked_pct": round(100.0 * len(poses) / max(len(seq), 1), 1),
        "lost_frames": lost, "fps": round(len(seq) / wall, 1),
        "features_median": int(np.median(n_obs)) if n_obs else 0,
        "path_m": round(path_length(p), 3), "net_m": round(float(np.linalg.norm(p[-1] - p[0])), 3) if len(p) else 0.0,
        "reference": ref_label, "ref_path_m": round(path_length(q), 3),
        "ref_net_m": round(float(np.linalg.norm(q[-1] - q[0])), 3) if len(q) else 0.0,
        "mesh_vertices": int(mesh.vertices().shape[0]),
        "occupied_voxels": int(len(occupied_voxels(mapper))),
        "slam": bool(args.slam),
    }
    if getattr(seq, "body", None) is not None:
        stats["body_masked_pct"] = round(100.0 * body_share.item() / max(len(poses), 1), 2)
    if grav:
        stats["gravity_err_median_deg"] = round(float(np.median(grav)), 2)
        stats["gravity_err_max_deg"] = round(float(np.max(grav)), 2)
        stats["verdict"] = verdict(stats["gravity_err_max_deg"])
    if len(p) > 2:
        R, t = umeyama_se3(p, q)
        err = np.linalg.norm((R @ p.T).T + t - q, axis=1)
        stats["ate_vs_ref_rmse_m"] = round(float(np.sqrt((err ** 2).mean())), 3)
        stats["ate_vs_ref_max_m"] = round(float(err.max()), 3)
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"[{name}] {stats.get('verdict', '?')}: gravity error max "
          f"{stats.get('gravity_err_max_deg', float('nan'))} deg, tracked {stats['tracked_pct']} %, path {stats['path_m']} m "
          f"({ref_label} {stats['ref_path_m']} m), ATE vs {ref_label} "
          f"{stats.get('ate_vs_ref_rmse_m', float('nan'))*100:.1f} cm, {stats['fps']} fps", flush=True)
    del mapper, tracker
    torch.cuda.empty_cache()
    return stats


# --------------------------------------------------------------------------- #
def write_summary(rows: list, path: Path) -> None:
    icon = {"ok": "✅ ok", "marginal": "⚠️ marginal", "drifted": "❌ drifted"}
    lines = ["| episode | frames | verdict | gravity err median / max (°) | path (m) | "
             "reference | reference path (m) | net (m) | ATE vs reference (cm) | fps |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for s in rows:
        if "error" in s:
            lines.append(f"| {s['episode']} | — | ❌ {s['error']} | | | | | | | |")
            continue
        ate = s.get("ate_vs_ref_rmse_m")
        lines.append(f"| {s['episode']} | {s['frames']} | {icon.get(s.get('verdict'), '?')} | "
                     f"{s.get('gravity_err_median_deg', '—')} / {s.get('gravity_err_max_deg', '—')} | "
                     f"{s['path_m']} | {s['reference']} | {s['ref_path_m']} | {s['net_m']} | "
                     f"{'—' if ate is None else f'{ate*100:.1f}'} | {s['fps']} |")
    n_ok = sum(s.get("verdict") == "ok" for s in rows)
    lines.append(f"\n{n_ok}/{len(rows)} episodes ok (gravity error < 5° throughout).")
    path.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episodes", type=Path, nargs="+",
                    help="Humanoid Everyday episode_N or TUM RGB-D sequence directories")
    ap.add_argument("--out", type=Path, default=Path("/results/cuvslam"))
    ap.add_argument("--robot", default="g1", choices=("g1", "h1"))
    ap.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    ap.add_argument("--slam", action="store_true",
                    help="enable cuVSLAM's SLAM layer (loop closure) and map with its poses")
    ap.add_argument("--voxel", type=float, default=0.015, help="TSDF voxel size, metres")
    ap.add_argument("--max-integration", type=float, default=5.0, help="metres")
    ap.add_argument("--depth-max", type=float, default=6.0, help="metres")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=0, help="0 = whole episode")
    ap.add_argument("--mesh-every", type=int, default=15,
                    help="stream the mesh every N frames (meshing costs ~10 ms)")
    ap.add_argument("--stream-url", default="rerun+http://127.0.0.1:9876/proxy")
    ap.add_argument("--serve", action="store_true",
                    help="host a rerun web viewer here instead of streaming to --stream-url; "
                         "prints the URL to open and keeps serving after the last episode "
                         "until Ctrl-C")
    ap.add_argument("--web-port", type=int, default=9090)
    ap.add_argument("--grpc-port", type=int, default=9877)
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--no-stream-images", action="store_true")
    ap.add_argument("--no-occupancy", action="store_true",
                    help="do not stream the occupied voxels (streamed with the mesh by default)")
    ap.add_argument("--no-body-mask", action="store_true",
                    help="map the G1's own body too (masked out of depth and colour by default)")
    ap.add_argument("--no-esdf", action="store_true",
                    help="do not compute and stream the ESDF slice (streamed with the mesh by default)")
    ap.add_argument("--esdf-height", type=float, default=0.5,
                    help="height of the streamed ESDF slice above the floor, metres")
    ap.add_argument("--save-rrd", action="store_true",
                    help="write <out>/rrd/<episode>.rrd instead of streaming")
    args = ap.parse_args()
    args.save_rrd_dir = args.out / "rrd"
    if args.save_rrd:
        args.save_rrd_dir.mkdir(parents=True, exist_ok=True)

    # run_all.sh runs this as a background job, which starts with SIGINT ignored; re-arm it
    # so the Ctrl-C the script forwards still stops us.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    args.viewer_url = args.stream_url
    server = serve(args) if args.serve and not (args.no_stream or args.save_rrd) else None

    rows = []
    try:
        for ep in args.episodes:
            ep = ep.resolve()
            name = f"{ep.parent.name}/{ep.name}"
            try:
                seq = open_sequence(ep, args)
                name = seq.name
                rows.append(run_episode(seq, args.out / seq.out, args))
            except Exception as e:  # keep going: one bad episode must not stop the sweep
                print(f"[{name}] FAILED: {e!r}", flush=True)
                rows.append({"episode": name, "error": type(e).__name__})
    except KeyboardInterrupt:
        print("\n[run] interrupted", flush=True)
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # a second Ctrl-C must not cut the summary
        server = None
    write_summary(rows, args.out / "summary.md")

    if server is not None and server.poll() is None:
        print(f"\n  viewer still serving at  {args.viewer_url}  -- Ctrl-C to stop\n", flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
