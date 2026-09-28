#!/usr/bin/env python3
"""Vertex/face count, bounding box and per-class share of a Kimera-Semantics
PLY (vertex colours = class colours from the label csv)."""
import csv
import sys
from collections import Counter

import numpy as np

ply, csv_path = sys.argv[1], sys.argv[2]
names = {0: "unknown", 1: "appliance", 2: "books", 3: "floor", 4: "ceiling", 5: "chair",
         6: "vase", 7: "couch", 8: "tree", 9: "furniture", 10: "objects", 11: "lamp",
         12: "painting", 13: "plant", 14: "bed", 15: "stairs", 16: "table", 17: "screen",
         18: "bin", 19: "wall", 20: "human"}
# Kimera keeps the LAST csv row of each id as that class's display colour
# (kimera_semantics/src/color.cpp), so invert that map.
id2col = {}
for r in csv.DictReader(open(csv_path)):
    id2col[int(r["id"])] = (int(r["red"]), int(r["green"]), int(r["blue"]))
col2id = {c: i for i, c in id2col.items()}
col2id[(255, 255, 255)] = 0   # ...except label 0, which color.cpp always paints white

with open(ply, "rb") as f:
    header, props, nv, nf = [], [], 0, 0
    while True:
        line = f.readline().decode().strip()
        header.append(line)
        if line.startswith("element vertex"):
            nv = int(line.split()[-1])
        elif line.startswith("element face"):
            nf = int(line.split()[-1])
        elif line.startswith("property") and nf == 0:
            props.append(line.split())
        if line == "end_header":
            break
    fmt = header[1].split()[1]
    types = {"float": "f4", "uchar": "u1", "double": "f8", "int": "i4"}
    dt = np.dtype([(p[2], types[p[1]]) for p in props])
    if fmt == "ascii":
        v = np.loadtxt(f, max_rows=nv)
        col = {p[2]: i for i, p in enumerate(props)}
        xyz = v[:, [col["x"], col["y"], col["z"]]]
        rgb = v[:, [col["red"], col["green"], col["blue"]]].astype(int)
    else:
        v = np.frombuffer(f.read(nv * dt.itemsize), dtype=dt)
        xyz = np.stack([v["x"], v["y"], v["z"]], 1)
        rgb = np.stack([v["red"], v["green"], v["blue"]], 1).astype(int)

print(f"vertices {nv}  faces {nf}")
print("bbox min", np.round(xyz.min(0), 2), "max", np.round(xyz.max(0), 2))
print("extent m", " x ".join(f"{e:.1f}" for e in xyz.max(0) - xyz.min(0)))
cnt = Counter(col2id.get(tuple(c), -1) for c in map(tuple, rgb))
for k, c in cnt.most_common():
    print(f"  {names.get(k, 'no colour'):10s} {100 * c / nv:5.1f} %")
