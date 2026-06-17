#!/usr/bin/env python3
"""Visualize a 2D pose-graph optimization result.

Reads the standard ``pose_graph.txt`` dump and renders a matplotlib PNG that
overlays the ground-truth, noisy-initial and optimized trajectories, plus the
loop-closure edge that pulls the graph back together.

Data format::

    nodes <N>
    <id> <gx> <gy> <gth> <ix> <iy> <ith> <ox> <oy> <oth>
    ... (N rows)
    edges <M>
    <i> <j> <type>       # type: 0 = odometry, 1 = loop closure
    ... (M rows)
"""
import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse(path):
    nodes = {}
    edges = []
    with open(path) as f:
        mode = None
        remaining = 0
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tok = line.split()
            if tok[0] == "nodes":
                mode, remaining = "nodes", int(tok[1])
            elif tok[0] == "edges":
                mode, remaining = "edges", int(tok[1])
            elif mode == "nodes" and remaining > 0:
                i = int(tok[0])
                vals = [float(v) for v in tok[1:10]]
                nodes[i] = vals  # gx gy gth ix iy ith ox oy oth
                remaining -= 1
            elif mode == "edges" and remaining > 0:
                edges.append((int(tok[0]), int(tok[1]), int(tok[2])))
                remaining -= 1
    return nodes, edges


def traj(nodes, off):
    ids = sorted(nodes)
    return (np.array([nodes[i][off] for i in ids]),
            np.array([nodes[i][off + 1] for i in ids]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default="pose_graph.txt")
    ap.add_argument("-o", "--out", default="pose_graph.png")
    args = ap.parse_args()

    try:
        nodes, edges = parse(args.data)
    except FileNotFoundError:
        sys.exit(f"[plot_pose_graph] cannot open {args.data}; run the "
                 f"pose_graph example first to generate it.")

    gx, gy = traj(nodes, 0)
    ix, iy = traj(nodes, 3)
    ox, oy = traj(nodes, 6)

    plt.figure(figsize=(7, 7))
    plt.plot(gx, gy, "g-o", lw=2, ms=8, label="ground truth")
    plt.plot(ix, iy, color="gray", ls="--", marker="s", ms=6,
             label="initial (noisy)")
    plt.plot(ox, oy, "r-^", lw=2, ms=7, label="optimized")

    # draw loop-closure edges on the optimized trajectory
    for i, j, t in edges:
        if t == 1 and i in nodes and j in nodes:
            plt.plot([nodes[i][6], nodes[j][6]],
                     [nodes[i][7], nodes[j][7]],
                     "b:", lw=1.5,
                     label="loop closure" if "loop closure" not in
                     plt.gca().get_legend_handles_labels()[1] else None)

    for i in sorted(nodes):
        plt.annotate(f"x{i}", (ox[i], oy[i]),
                     textcoords="offset points", xytext=(6, 6), fontsize=9)

    plt.title("2D pose-graph optimization")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"[plot_pose_graph] wrote {args.out}")


if __name__ == "__main__":
    main()
