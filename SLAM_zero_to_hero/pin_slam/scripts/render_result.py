#!/usr/bin/env python3
"""Render a finished PIN-SLAM run to PNG with the Open3D legacy Visualizer.

Draws the reconstructed mesh (or the neural point map when no mesh exists)
plus the SLAM trajectory (red) and the KITTI ground truth (green), and saves a
top-down view.  Without $DISPLAY it starts its own Xvfb (xvfb-run hangs in
rootless podman).

    python3 demo_scripts/render_result.py experiments/<run_dir> experiments/out.png
"""
import glob
import os
import subprocess
import sys
import time

import numpy as np
import open3d as o3d


def load_traj(path, color):
    if not os.path.exists(path):
        return None
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)
    if len(pts) < 2:
        return None
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(pts) - 1)])
    ls.paint_uniform_color(color)
    return ls


def main():
    run, out = sys.argv[1], sys.argv[2]
    xvfb = None
    if not os.environ.get("DISPLAY"):
        xvfb = subprocess.Popen(["Xvfb", ":98", "-screen", "0", "1920x1080x24"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.environ["DISPLAY"] = ":98"
        time.sleep(2)
    geoms = []
    meshes = sorted(glob.glob(os.path.join(run, "mesh", "*.ply")))
    if meshes:
        mesh = o3d.io.read_triangle_mesh(meshes[-1])
        mesh.compute_vertex_normals()
        if not mesh.has_vertex_colors():
            z = np.asarray(mesh.vertices)[:, 2]
            t = np.clip((z - np.percentile(z, 2)) / (np.ptp(z) + 1e-6) * 3, 0, 1)
            mesh.vertex_colors = o3d.utility.Vector3dVector(np.stack([0.3 + 0.6 * t, 0.5 + 0.3 * t, 0.9 - 0.5 * t], 1))
        print(f"mesh {meshes[-1]}: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        geoms.append(mesh)
    else:
        pcd = o3d.io.read_point_cloud(os.path.join(run, "map", "neural_points.ply"))
        print(f"neural points: {len(pcd.points)}")
        geoms.append(pcd)
    for name, color in (("gt_poses.ply", [0.0, 0.8, 0.0]),
                        ("slam_poses.ply", [1.0, 0.0, 0.0])):
        ls = load_traj(os.path.join(run, name), color)
        if ls is not None:
            geoms.append(ls)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1600, height=1200, visible=True)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0])
    opt.line_width = 5.0
    opt.mesh_show_back_face = True
    ctr = vis.get_view_control()
    ctr.set_front([0.0, -0.3, 1.0])   # mostly top-down, slight tilt
    ctr.set_up([0.0, 1.0, 0.0])
    ctr.set_zoom(0.7)
    for _ in range(5):
        vis.poll_events()
        vis.update_renderer()
    vis.capture_screen_image(out, do_render=True)
    vis.destroy_window()
    print("saved", out)
    if xvfb is not None:
        xvfb.terminate()


if __name__ == "__main__":
    main()
