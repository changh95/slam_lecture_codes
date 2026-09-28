#!/usr/bin/env python3
"""Publishes a saved Kimera-Semantics PLY on /kimera_semantics_node/mesh once
(voxblox_msgs/Mesh, latched), so RViz can show a finished run without playing
the bag again (the office bag takes ~26 min).

  view_mesh.py mesh.ply [block_size=1.6]    (1.6 = 32 voxels x 5 cm)"""
import sys

import numpy as np
import rospy
from voxblox_msgs.msg import Mesh, MeshBlock

ply = sys.argv[1]
bs = float(sys.argv[2]) if len(sys.argv) > 2 else 1.6

with open(ply) as f:
    nv = nf = 0
    while True:
        line = f.readline().strip()
        if line.startswith("element vertex"):
            nv = int(line.split()[-1])
        elif line.startswith("element face"):
            nf = int(line.split()[-1])
        elif line == "end_header":
            break
    # x y z nx ny nz r g b a  /  3 i j k   (ASCII, as voxblox writes it)
    v = np.array(" ".join(f.readline() for _ in range(nv)).split(), np.float32).reshape(nv, -1)
    tri = np.array(" ".join(f.readline() for _ in range(nf)).split(), np.int64).reshape(nf, 4)[:, 1:]

xyz, rgb = v[:, :3], v[:, 6:9].astype(np.uint8)
# every triangle goes to the block holding its lowest corner; the other corners
# then lie within 2 * block_size of that block's origin, as MeshBlock requires
corner = xyz[tri].min(1)
blk = np.floor(corner / bs).astype(np.int64)
order = np.lexsort(blk.T[::-1])
blk, tri = blk[order], tri[order]
cuts = np.flatnonzero(np.any(np.diff(blk, axis=0), axis=1)) + 1

msg = Mesh(block_edge_length=bs)
msg.header.frame_id = "world"
for b, t in zip(np.split(blk, cuts), np.split(tri, cuts)):
    idx = t.reshape(-1)
    q = np.clip((xyz[idx] / bs - b[0]) * 0.5 * 65535, 0, 65535).astype(np.uint16)
    c = rgb[idx]
    msg.mesh_blocks.append(MeshBlock(index=b[0].tolist(), x=q[:, 0].tolist(), y=q[:, 1].tolist(),
                                     z=q[:, 2].tolist(), r=c[:, 0].tobytes(), g=c[:, 1].tobytes(),
                                     b=c[:, 2].tobytes()))
print(f"{nv} vertices, {nf} faces -> {len(msg.mesh_blocks)} blocks")

rospy.init_node("view_mesh")
pub = rospy.Publisher("/kimera_semantics_node/mesh", Mesh, queue_size=1, latch=True)
msg.header.stamp = rospy.Time.now()
pub.publish(msg)
rospy.spin()
