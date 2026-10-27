#!/usr/bin/env python3
"""Write config/g1_body_model.json: the Unitree G1's body as spheres, which
cuvslam_nvblox.py poses with each frame's joint states to mask the robot out of its
own head camera's depth (see cuvslam_nvblox.BodyModel), and config/g1_visual_mesh.npz,
the URDF's meshes simplified for drawing the robot in the viewer.

The JSON holds three things:

  - the kinematic tree of the G1 with Dex3 hands, from unitree_ros'
    robots/g1_description/g1_29dof_with_hand.urdf;
  - spheres around every link's mesh: k-means over the mesh's vertices, one sphere
    per SPACING of the link's longest extent, each as big as its furthest vertex;
  - where the depth camera sits on the torso. Humanoid Everyday does not give it,
    and pelvis_to_colour_camera in humanoid_everyday_d435.json was measured against
    the pelvis with a nominal forward offset. Starting from that, the camera-facing
    facets of the posed arm and hand meshes are fitted to the depth points by ICP,
    over every 12th frame of the episodes given.

    git clone --depth 1 https://github.com/unitreerobotics/unitree_ros /tmp/unitree_ros
    make_g1_body_model.py /tmp/unitree_ros/robots/g1_description/g1_29dof_with_hand.urdf \\
        /data/<task>/episode_N [...] --out config/g1_body_model.json
"""

import argparse
import json
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cuvslam_nvblox import DEFAULT_CALIB, BodyModel, umeyama_se3  # noqa: E402
from he_to_nvblox import DEPTH_H, DEPTH_W, load_calib, read_depth, read_records, transform_from_dict  # noqa: E402

SPACING = 0.035   # metres of a link's longest extent per sphere
# The drawn meshes merge their vertices on a grid this fine: 620 k triangles become 84 k,
# ~2 MB in each recording instead of ~15 MB.
VISUAL_CELL = 0.005
ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
HAND = ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"]
LEG = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
# The order of the joint values in a Humanoid Everyday G1 record: left before right, and
# the three waist joints after the twelve leg joints.
STATE_JOINTS = {
    "arm_state": [f"{s}_{j}_joint" for s in ("left", "right") for j in ARM],
    "hand_state": [f"{s}_hand_{j}_joint" for s in ("left", "right") for j in HAND],
    "leg_state": [f"{s}_{j}_joint" for s in ("left", "right") for j in LEG]
                 + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
}


def read_stl(path: Path) -> np.ndarray:
    """Triangles, N x 3 x 3, of a binary STL."""
    b = path.read_bytes()
    n = struct.unpack("<I", b[80:84])[0]
    tri = np.frombuffer(b[84:84 + 50 * n], dtype=np.dtype([("normal", "<3f4"), ("v", "<3f4", (3,)),
                                                          ("attr", "<u2")]))
    return tri["v"].astype(np.float64)


def read_urdf(path: Path):
    """Joints, parents before children, and each link's mesh file and colour."""
    root = ET.parse(path).getroot()
    joints = []
    for j in root.findall("joint"):
        o, a = j.find("origin"), j.find("axis")
        joints.append({
            "name": j.get("name"), "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
            "xyz": [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()],
            "rpy": [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()],
            "axis": [float(v) for v in a.get("xyz").split()] if j.get("type") == "revolute" else None,
        })
    children = {j["child"] for j in joints}
    order, todo = [], [j for j in joints if j["parent"] not in children]
    while todo:
        j = todo.pop(0)
        order.append(j)
        todo += [k for k in joints if k["parent"] == j["child"]]
    colours = {m.get("name"): [float(c) for c in m.find("color").get("rgba").split()[:3]]
               for m in root.findall("material") if m.find("color") is not None}
    meshes = {}
    for link in root.findall("link"):
        g, o = link.find("visual/geometry/mesh"), link.find("visual/origin")
        if g is None:
            continue
        assert o is None or (o.get("xyz", "0 0 0").split() == ["0"] * 3 and o.get("rpy", "0 0 0").split() == ["0"] * 3), \
            f"{link.get('name')}: visual origin not at the link frame"
        m = link.find("visual/material")
        meshes[link.get("name")] = (path.parent / g.get("filename"),
                                    colours.get(m.get("name") if m is not None else None, [0.7, 0.7, 0.7]))
    return order, meshes


def simplify(tri: np.ndarray, cell: float = VISUAL_CELL):
    """Vertex clustering: vertices in one grid cell merge at their mean, and the triangles
    that collapse are dropped. Returns vertices and triangle indices."""
    v = tri.reshape(-1, 3)
    cells, inv = np.unique(np.floor(v / cell).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.ravel()
    pos = np.zeros((len(cells), 3))
    np.add.at(pos, inv, v)
    pos /= np.bincount(inv)[:, None]
    f = inv.reshape(-1, 3)
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    f = f[np.sort(np.unique(np.sort(f, 1), axis=0, return_index=True)[1])]
    used, f = np.unique(f, return_inverse=True)
    return pos[used], f.reshape(-1, 3)


def fit_spheres(v: np.ndarray, spacing: float = SPACING) -> np.ndarray:
    """Spheres (x, y, z, r) holding every vertex: k-means seeded along the principal axis."""
    v = np.unique(v.round(5), axis=0)
    axis = np.linalg.svd(v - v.mean(0), full_matrices=False)[2][0]
    s = (v - v.mean(0)) @ axis
    k = max(1, int(np.ceil(np.ptp(s) / spacing)))
    c = v[[np.argmin(np.abs(s - q)) for q in np.linspace(s.min(), s.max(), k + 2)[1:-1]]]
    for _ in range(30):
        label = np.argmin(((v[:, None] - c[None]) ** 2).sum(-1), 1)
        c = np.array([v[label == i].mean(0) if (label == i).any() else c[i] for i in range(k)])
    label = np.argmin(((v[:, None] - c[None]) ** 2).sum(-1), 1)
    return np.array([[*c[i], np.linalg.norm(v[label == i] - c[i], axis=1).max()]
                     for i in range(k) if (label == i).any()])


def fit_camera(model: BodyModel, meshes: dict, episodes: list, calib: dict):
    """T_torso_depth by ICP of the posed arm and hand meshes onto the depth points."""
    rng = np.random.default_rng(0)
    facets = {}
    for link, (f, _) in meshes.items():
        if not any(k in link for k in ("shoulder", "elbow", "wrist", "hand")):
            continue
        tri = read_stl(f)
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area = np.linalg.norm(n, axis=1)
        ok = area > 1e-9
        pick = rng.choice(ok.sum(), size=min(ok.sum(), max(20, int(area[ok].sum() / 2 / 1.5e-4))),
                          p=area[ok] / area[ok].sum(), replace=False)
        facets[link] = (tri[ok].mean(1)[pick], (n[ok] / area[ok, None])[pick])

    intr = calib["depth_intrinsics"]
    v, u = np.mgrid[0:DEPTH_H:2, 0:DEPTH_W:2]
    frames = []
    for ep in episodes:
        for rec in read_records(ep)[::12]:
            d = read_depth(ep, rec)[::2, ::2] / 1000.0
            ok = (d > 0.15) & (d < 1.3)
            if not ok.any():
                continue
            q = np.stack([(u[ok] - intr["cx"]) / intr["fx"] * d[ok], (v[ok] - intr["cy"]) / intr["fy"] * d[ok],
                          d[ok]], 1)
            poses = model.link_poses(rec)
            pts = np.concatenate([c @ poses[k][:3, :3].T + poses[k][:3, 3] for k, (c, _) in facets.items()])
            nrm = np.concatenate([n @ poses[k][:3, :3].T for k, (_, n) in facets.items()])
            frames.append((pts, nrm, q, cKDTree(q)))

    def pairs(T_dt, gate):
        src, dst = [], []
        for pts, nrm, q, tree in frames:
            p = pts @ T_dt[:3, :3].T + T_dt[:3, 3]
            facing = ((nrm @ T_dt[:3, :3].T) * p).sum(1) < 0
            z = np.maximum(p[:, 2], 1e-6)
            pu, pv = p[:, 0] / z * intr["fx"] + intr["cx"], p[:, 1] / z * intr["fy"] + intr["cy"]
            p = p[facing & (p[:, 2] > 0.1) & (pu >= 0) & (pu < DEPTH_W) & (pv >= 0) & (pv < DEPTH_H)]
            dist, j = tree.query(p, distance_upper_bound=gate)
            hit = np.isfinite(dist)
            src.append(p[hit])
            dst.append(q[j[hit]])
        return np.concatenate(src), np.concatenate(dst)

    # start: the camera as calibrated against the pelvis, through a straight waist
    T_pelvis_torso = np.linalg.inv(model.link_poses({"states": {k: [0.0] * len(n) for k, n in
                                                                 model.state_joints.items()}})["pelvis"])
    T_dt = np.linalg.inv(np.linalg.inv(T_pelvis_torso) @ transform_from_dict(calib["pelvis_to_colour_camera"])
                         @ transform_from_dict(calib["depth_to_colour"]))
    for gate in [0.10] * 5 + [0.06] * 5 + [0.04] * 10 + [0.03] * 10:
        src, dst = pairs(T_dt, gate)
        r = np.linalg.norm(src - dst, axis=1)
        keep = r <= np.quantile(r, 0.8)
        R, t = umeyama_se3(src[keep], dst[keep])
        step = np.eye(4)
        step[:3, :3], step[:3, 3] = R, t
        T_dt = step @ T_dt
    src, dst = pairs(T_dt, 0.03)
    return np.linalg.inv(T_dt), len(frames), float(np.median(np.linalg.norm(src - dst, axis=1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("urdf", type=Path, help="unitree_ros' g1_29dof_with_hand.urdf, next to its meshes/")
    ap.add_argument("episodes", type=Path, nargs="+", help="Humanoid Everyday G1 episode_N directories")
    ap.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    config = Path(__file__).resolve().parent.parent / "config"
    ap.add_argument("--out", type=Path, default=config / "g1_body_model.json")
    ap.add_argument("--visual-out", type=Path, default=config / "g1_visual_mesh.npz")
    args = ap.parse_args()

    joints, meshes = read_urdf(args.urdf)
    spheres = {link: fit_spheres(read_stl(f).reshape(-1, 3)) for link, (f, _) in meshes.items()}
    visual = {}
    for link, (f, colour) in meshes.items():
        v, tri = simplify(read_stl(f))
        visual[f"{link}/vertices"] = v.astype(np.float32)
        visual[f"{link}/triangles"] = tri.astype(np.int32)
        visual[f"{link}/colour"] = np.round(np.asarray(colour) * 255).astype(np.uint8)
    np.savez_compressed(args.visual_out, **visual)
    model = {
        "_comment": [
            "The Unitree G1 (29 DoF, Dex3 hands) as spheres, for masking the robot out of",
            "its head camera's depth: see cuvslam_nvblox.BodyModel. Written by",
            "scripts/make_g1_body_model.py from unitree_ros' (BSD-3-Clause)",
            "robots/g1_description/g1_29dof_with_hand.urdf and its meshes.",
            "spheres: per link, x y z r in the link frame. torso_to_depth_camera: fitted to",
            "the depth points by ICP of the posed arm and hand meshes; see fit. The meshes",
            "themselves, simplified, are in g1_visual_mesh.npz, for the viewer.",
        ],
        "state_joints": STATE_JOINTS,
        "joints": joints,
        "spheres": {link: np.round(s, 4).tolist() for link, s in spheres.items()},
        "torso_to_depth_camera": {"rotation": np.eye(3).tolist(), "translation": [0.0, 0.0, 0.0]},
    }
    T, n_frames, residual = fit_camera(BodyModel(model), meshes, args.episodes,
                                       load_calib(args.calib, "g1"))
    model["torso_to_depth_camera"] = {"rotation": np.round(T[:3, :3], 6).tolist(),
                                      "translation": np.round(T[:3, 3], 5).tolist()}
    model["fit"] = {"frames": n_frames, "median_residual_m": round(residual, 4),
                    "episodes": [f"{e.parent.name}/{e.name}" for e in args.episodes]}
    args.out.write_text(json.dumps(model, indent=1) + "\n")
    print(f"{sum(len(s) for s in spheres.values())} spheres on {len(spheres)} links; camera fitted on "
          f"{n_frames} frames, median residual {residual * 100:.1f} cm -> {args.out}")
    print(f"{sum(len(v) for k, v in visual.items() if k.endswith('/triangles'))} triangles to draw "
          f"-> {args.visual_out}")


if __name__ == "__main__":
    main()
