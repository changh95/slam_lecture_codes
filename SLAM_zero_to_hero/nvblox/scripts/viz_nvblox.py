#!/usr/bin/env python3
"""View an nvblox run: the mesh, the ESDF slice and the camera trajectory.

Two sinks:

  --serve            rerun web viewer inside the container (default). Publish
                     ports 9090 and 9877 and open the URL it prints.
  --png out.png      one offscreen render of the mesh through Open3D's EGL
                     backend. No display, no viewer, no X11.

`--connect rerun+http://host:9876/proxy` pushes into a viewer you already have
open on the host instead of serving one.
"""

import argparse
import os
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def read_tum(path: Path):
    """The converter writes the TUM files beside cam_params.json, in dataset/."""
    if not path.exists():
        path = path.parent / "dataset" / path.name
    if not path.exists():
        return None, None
    d = np.loadtxt(path, comments="#")
    if d.ndim == 1:
        d = d[None]
    return d[:, 0], d[:, 1:4]


def read_traj_matrices(path: Path):
    if not path.exists():
        return None
    rows = np.loadtxt(path)
    return rows.reshape(-1, 4, 4)


# --------------------------------------------------------------------------- #
# rerun
# --------------------------------------------------------------------------- #
def log_to_rerun(run: Path, args):
    import open3d as o3d
    import rerun as rr
    import rerun.blueprint as rrb

    mesh_path = run / args.mesh
    if not mesh_path.exists():
        mesh_path = run / "mesh.ply"
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    cols = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="world", name="nvblox map"),
            rrb.Vertical(
                rrb.Spatial2DView(origin="camera/image", name="colour"),
                rrb.Spatial2DView(origin="camera/depth", name="depth"),
            ),
            column_shares=[3, 1],
        ),
        collapse_panels=True,
    )

    rr.init("nvblox_humanoid_everyday", spawn=False, default_blueprint=blueprint)
    if args.connect:
        rr.connect_grpc(args.connect)
        print(f"streaming to {args.connect}")
    else:
        rr.serve_grpc(grpc_port=args.grpc_port)
        rr.serve_web_viewer(web_port=args.web_port, open_browser=False,
                            connect_to=f"rerun+http://localhost:{args.grpc_port}/proxy")
        print(f"\n  open  http://localhost:{args.web_port}"
              f"/?url=rerun+http://localhost:{args.grpc_port}/proxy\n")

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/mesh",
           rr.Mesh3D(vertex_positions=verts, triangle_indices=tris,
                     vertex_colors=cols),
           static=True)
    print(f"logged {mesh_path.name}: {len(verts)} vertices, {len(tris)} triangles")

    # ESDF as points coloured by distance to the nearest surface.
    esdf = run / "esdf.ply"
    if esdf.exists() and args.esdf:
        pc = o3d.io.read_point_cloud(str(esdf))
        p = np.asarray(pc.points)
        # nvblox writes the distance in the intensity channel, which Open3D
        # surfaces as a grey colour; take one channel back out as the scalar.
        c = np.asarray(pc.colors)
        keep = np.ones(len(p), bool) if len(c) == 0 else c[:, 0] < 0.5
        rr.log("world/esdf", rr.Points3D(p[keep], radii=0.01,
                                        colors=None if len(c) == 0 else c[keep]),
               static=True)
        print(f"logged esdf.ply: {keep.sum()} voxels")

    times, xyz = read_tum(run / "traj_icp.tum")
    if xyz is None:
        times, xyz = read_tum(run / "traj_odom.tum")
    poses = read_traj_matrices(run / "dataset" / "seq" / "traj.txt")

    if xyz is not None:
        rr.log("world/trajectory", rr.LineStrips3D([xyz], colors=[[80, 200, 255]],
                                                   radii=0.008), static=True)
    _, odom = read_tum(run / "traj_odom.tum")
    if odom is not None and args.odom:
        # Odometry starts at the robot's own world origin; shift it onto the
        # tracked trajectory so the two shapes can be compared at all.
        shifted = odom - odom[0] + (xyz[0] if xyz is not None else 0)
        rr.log("world/odometry", rr.LineStrips3D([shifted], colors=[[255, 140, 60]],
                                                 radii=0.008), static=True)

    frames = sorted((run / "dataset" / "seq" / "results").glob("frame*.jpg"))
    depths = sorted((run / "dataset" / "seq" / "results").glob("depth*.png"))
    if poses is None or not frames:
        print("no per-frame data to replay; map is static in the viewer")
        return
    step = max(1, args.frame_stride)
    for i in range(0, len(poses), step):
        rr.set_time("frame", sequence=i)
        if times is not None and i < len(times):
            rr.set_time("sensor_time", duration=float(times[i] - times[0]))
        T = poses[i]
        rr.log("world/camera", rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
        rr.log("world/camera/frustum",
               rr.Pinhole(image_from_camera=CAMERA_K, resolution=[640, 480],
                          camera_xyz=rr.ViewCoordinates.RDF))
        if i < len(frames):
            rr.log("camera/image", rr.EncodedImage(path=frames[i]))
        if i < len(depths):
            from PIL import Image
            d = np.asarray(Image.open(depths[i]))
            rr.log("camera/depth", rr.DepthImage(d, meter=1000.0))
    print(f"replayed {len(range(0, len(poses), step))} frames")
    if not args.connect:
        print("viewer still serving -- Ctrl-C to stop")
        try:
            import time
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


CAMERA_K = np.array([[606.2996826171875, 0.0, 330.7660217285156],
                     [0.0, 606.292236328125, 252.64605712890625],
                     [0.0, 0.0, 1.0]])


def frustum_corners(T_world_cam, scale):
    """Apex plus the four image corners at `scale` metres along the optical axis."""
    K_inv = np.linalg.inv(CAMERA_K)
    px = np.array([[0, 0, 1], [640, 0, 1], [0, 480, 1], [640, 480, 1]], float)
    rays = (K_inv @ px.T).T
    rays = rays / rays[:, 2:3] * scale
    pts = np.vstack([np.zeros(3), rays])
    return (T_world_cam[:3, :3] @ pts.T).T + T_world_cam[:3, 3]


# --------------------------------------------------------------------------- #
# offscreen still
# --------------------------------------------------------------------------- #
def render_png(run: Path, args):
    import open3d as o3d
    import open3d.visualization.rendering as rendering

    mesh_path = run / args.mesh
    if not mesh_path.exists():
        mesh_path = run / "mesh.ply"
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()

    if args.colour_by == "height":
        # For the --frame depth maps: the depth camera sees a good deal more than
        # the colour camera, and that extra area is otherwise just a black fringe.
        z = np.asarray(mesh.vertices)[:, 2]
        lo, hi = np.percentile(z, [1, 96])
        t = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)[:, None]
        cold, warm = np.array([0.10, 0.22, 0.48]), np.array([0.97, 0.80, 0.35])
        mesh.vertex_colors = o3d.utility.Vector3dVector(cold * (1 - t) + warm * t)
    elif args.colour_by == "shade":
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.tile([0.80, 0.80, 0.82], (len(mesh.vertices), 1)))

    # Filament treats vertex colours as linear and gamma-encodes on the way out,
    # so sRGB bytes come back visibly washed out. Decode here and the round trip
    # is the identity.
    if mesh.has_vertex_colors() and not args.no_gamma:
        c = np.asarray(mesh.vertex_colors)
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4))

    w, h = args.width, args.height
    renderer = rendering.OffscreenRenderer(w, h)
    scene = renderer.scene
    scene.set_background(list(args.background) + [1.0])
    mat = rendering.MaterialRecord()
    # nvblox already bakes the observed RGB into the vertices. "defaultLit"
    # multiplies a light on top of that and washes the brick out to grey, so
    # keep it unlit unless the geometry needs shading to read.
    mat.shader = "defaultLit" if args.lit else "defaultUnlit"
    mat.base_color = [1.0, 1.0, 1.0, 1.0]
    scene.add_geometry("mesh", mesh, mat)

    # Frame on a robust bound, not the full one: stereo depth throws a few
    # streaks metres past the real surfaces and they would shrink the map to a
    # dot in the middle of the picture.
    V = np.asarray(mesh.vertices)
    frame_lo = np.percentile(V, args.clip, axis=0)
    frame_hi = np.percentile(V, 100 - args.clip, axis=0)

    poses = read_traj_matrices(run / "dataset" / "seq" / "traj.txt")
    if poses is not None and args.trajectory:
        lmat = rendering.MaterialRecord()
        lmat.shader = "unlitLine"
        lmat.line_width = 4.0
        xyz = poses[:, :3, 3]
        ls = o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(xyz),
            o3d.utility.Vector2iVector(np.stack([np.arange(len(xyz) - 1),
                                                np.arange(1, len(xyz))], 1)))
        ls.colors = o3d.utility.Vector3dVector(
            np.tile([0.05, 0.55, 0.95], (len(xyz) - 1, 1)))
        scene.add_geometry("traj", ls, lmat)

        # A few frusta: the trajectory runs 1.2 m above the map and would
        # otherwise read as a line floating in space with no relation to it.
        fmat = rendering.MaterialRecord()
        fmat.shader = "unlitLine"
        fmat.line_width = 2.0
        step = max(1, len(poses) // max(args.frusta, 1))
        pts, lines = [], []
        for T in poses[::step]:
            corners = frustum_corners(T, args.frustum_scale)
            base = len(pts)
            pts.extend(corners)
            lines.extend([[base, base + i] for i in range(1, 5)])
            lines.extend([[base + 1, base + 2], [base + 2, base + 4],
                          [base + 4, base + 3], [base + 3, base + 1]])
        if lines:
            P = np.array(pts)
            fr = o3d.geometry.LineSet(o3d.utility.Vector3dVector(P),
                                      o3d.utility.Vector2iVector(np.array(lines)))
            fr.colors = o3d.utility.Vector3dVector(
                np.tile([0.95, 0.45, 0.15], (len(lines), 1)))
            scene.add_geometry("frusta", fr, fmat)
            frame_lo = np.minimum(frame_lo, P.min(axis=0))
            frame_hi = np.maximum(frame_hi, P.max(axis=0))
        frame_lo = np.minimum(frame_lo, xyz.min(axis=0))
        frame_hi = np.maximum(frame_hi, xyz.max(axis=0))

    bb = mesh.get_axis_aligned_bounding_box()
    centre = (frame_lo + frame_hi) / 2
    extent = float(np.linalg.norm(frame_hi - frame_lo))
    # Frame the whole map: back off far enough that the bounding sphere fits the
    # vertical FOV, with a little margin, then look down on it from --eye.
    reach = 0.5 * extent / np.tan(np.radians(args.fov) / 2) * args.zoom
    direction = np.array(args.eye, dtype=float)
    direction /= np.linalg.norm(direction)
    renderer.setup_camera(args.fov, centre, centre + direction * reach,
                          np.array(args.up, dtype=float))
    scene.scene.set_sun_light([-0.3, -0.4, -0.9], [1.0, 1.0, 1.0], 45000)
    scene.scene.enable_sun_light(args.lit)

    img = renderer.render_to_image()
    out = Path(args.png)
    out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_image(str(out), img)
    print(f"wrote {out} ({w}x{h}) from {mesh_path.name}: "
          f"{len(mesh.vertices)} vertices, extent {np.round(bb.get_extent(), 2)} m")


def plot_trajectories(run: Path, args):
    """Top-down: the ICP trajectory nvblox was given, against the robot odometry.

    Both are put in their own frames by construction, so the odometry is
    rotated and shifted onto the tracked one -- Umeyama without scale. What
    survives that fit is the shape and the length, which is the point: the
    legged odometry of a walking G1 comes out roughly half as long.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, icp = read_tum(run / "traj_icp.tum")
    _, odo = read_tum(run / "traj_odom.tum")
    if icp is None or odo is None:
        raise SystemExit("need traj_icp.tum and traj_odom.tum in the run directory")
    n = min(len(icp), len(odo))
    icp, odo = icp[:n], odo[:n]

    a, b = odo - odo.mean(0), icp - icp.mean(0)
    U, _, Vt = np.linalg.svd(b.T @ a)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    aligned = (R @ a.T).T + icp.mean(0)

    def length(p):
        return np.linalg.norm(np.diff(p, axis=0), axis=1).sum()

    def net(p):
        return np.linalg.norm(p[-1] - p[0])

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    ax.plot(icp[:, 0], icp[:, 1], color="#1a8fd1", lw=2.0,
            label=f"depth ICP (into nvblox)   path {length(icp):.2f} m, "
                  f"net {net(icp):.2f} m")
    ax.plot(aligned[:, 0], aligned[:, 1], color="#e8761f", lw=2.0, ls="--",
            label=f"robot legged odometry     path {length(odo):.2f} m, "
                  f"net {net(odo):.2f} m")
    ax.scatter(icp[0, 0], icp[0, 1], s=60, color="#1a8fd1", zorder=5, marker="o")
    ax.scatter(icp[-1, 0], icp[-1, 1], s=70, color="#1a8fd1", zorder=5, marker="s")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{run.name}: camera trajectory, top-down "
                 f"(odometry rigidly aligned)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out = Path(args.plot)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}: icp path {length(icp):.2f} m vs odom {length(odo):.2f} m "
          f"({length(icp)/max(length(odo), 1e-9):.2f}x)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="an output directory made by run_nvblox.sh")
    ap.add_argument("--mesh", default="mesh.ply",
                    help="which mesh to show. mesh.ply shares its frame with "
                         "traj.txt, so the trajectory overlays correctly; "
                         "mesh_ground_aligned.ply does not")
    ap.add_argument("--png", help="render one still here instead of serving a viewer")
    ap.add_argument("--connect", help="push into an existing rerun viewer, e.g. "
                                      "rerun+http://127.0.0.1:9876/proxy")
    ap.add_argument("--web-port", type=int, default=9090)
    ap.add_argument("--grpc-port", type=int, default=9877)
    ap.add_argument("--frame-stride", type=int, default=2)
    ap.add_argument("--no-esdf", dest="esdf", action="store_false")
    ap.add_argument("--no-odom", dest="odom", action="store_false")
    ap.add_argument("--no-trajectory", dest="trajectory", action="store_false")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--fov", type=float, default=50.0)
    ap.add_argument("--eye", type=float, nargs=3, default=(-0.55, -0.55, 0.62),
                    help="viewing direction from the map centre (z is up)")
    ap.add_argument("--up", type=float, nargs=3, default=(0.0, 0.0, 1.0))
    ap.add_argument("--zoom", type=float, default=0.62,
                    help="<1 moves the camera closer; 1.0 fits the bounding sphere")
    ap.add_argument("--clip", type=float, default=1.0,
                    help="percentile of vertices to ignore when framing, per axis")
    ap.add_argument("--lit", action="store_true",
                    help="shade the mesh with a sun light instead of showing the "
                         "vertex colours as nvblox integrated them")
    ap.add_argument("--background", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    ap.add_argument("--no-gamma", action="store_true",
                    help="skip the sRGB decode of the vertex colours")
    ap.add_argument("--colour-by", default="rgb", choices=("rgb", "height", "shade"),
                    help="rgb shows the colours nvblox integrated; height and "
                         "shade are for --frame depth maps, whose periphery has "
                         "no colour at all (shade implies --lit)")
    ap.add_argument("--frusta", type=int, default=8,
                    help="how many camera frusta to draw (0 for none)")
    ap.add_argument("--frustum-scale", type=float, default=0.5,
                    help="frustum depth in metres")
    ap.add_argument("--plot", help="write a top-down trajectory plot here "
                                   "(tracked against the robot's odometry) "
                                   "instead of rendering the map")
    args = ap.parse_args()
    if args.colour_by == "shade":
        args.lit = True

    if args.plot:
        plot_trajectories(args.run, args)
    elif args.png:
        os.environ.setdefault("EGL_PLATFORM", "surfaceless")
        render_png(args.run, args)
    else:
        log_to_rerun(args.run, args)


if __name__ == "__main__":
    main()
