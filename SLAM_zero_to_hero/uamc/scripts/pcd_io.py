"""Minimal binary PCD reader/writer for FAST-LIVO2 maps.

FAST-LIVO2 saves x y z rgb (LiDAR-visual-inertial, img_en: 1) or
x y z intensity normal_x normal_y normal_z curvature (LiDAR-inertial, img_en: 0).
"""
import numpy as np

DT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")])
DT_I = np.dtype([(n, "<f4") for n in
                 ("x", "y", "z", "intensity", "normal_x", "normal_y", "normal_z", "curvature")])


def read_pcd(path):
    """Return a structured array with fields x y z and either rgb or intensity."""
    with open(path, "rb") as f:
        header = {}
        while True:
            line = f.readline().decode("ascii").strip()
            key, _, val = line.partition(" ")
            header[key] = val
            if key == "DATA":
                break
        assert header["DATA"] == "binary", "only binary PCD is supported"
        fields = header["FIELDS"].split()
        dt = {tuple(DT.names): DT, tuple(DT_I.names): DT_I}.get(tuple(fields))
        assert dt is not None, f"unsupported PCD fields: {fields}"
        n = int(header["POINTS"])
        return np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)


def write_pcd(path, pts):
    """Write x y z rgb points (structured array with dtype DT)."""
    n = len(pts)
    header = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z rgb\n"
              "SIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\n"
              f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(np.ascontiguousarray(pts, dtype=DT).tobytes())


def height_rgb(z, lo, hi):
    """Blue -> cyan -> green -> yellow -> red over [lo, hi]; returns uint32 r, g, b arrays."""
    s = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
    stops = np.array([[0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]], float)
    x = s * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    f = (x - i)[:, None]
    rgb = stops[i] * (1 - f) + stops[i + 1] * f
    return tuple(rgb[:, j].astype(np.uint32) for j in range(3))
