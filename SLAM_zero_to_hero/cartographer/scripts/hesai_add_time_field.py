"""Rewrite /hesai/pandar so Cartographer can de-skew each sweep.

Hilti publishes an ABSOLUTE float64 'timestamp' per point. cartographer_ros looks
for a float32 field literally named 'time' (see PointXYZIT in msg_conversion.cc);
finding none it gives every point time = 0, so a whole 100 ms sweep is treated as
instantaneous and LocalTrajectoryBuilder3D never de-skews it.

This writes a new bag whose /hesai/pandar messages carry
    x, y, z, intensity, time   (all float32, point_step 20)
with time = point.timestamp - header.stamp  (measured range 0.000 .. 0.100 s,
already monotonic in the source, so no re-sorting is needed -- but we sort anyway
because cartographer_ros CHECKs that the LAST point has the largest stamp).

Side effect, and a good one: cartographer_ros then sets the node timestamp to
header.stamp + time_of_last_point, i.e. the END of the sweep -- the same
convention FAST-LIO2 uses.

usage: hesai_add_time_field.py IN.bag OUT.bag [KEEP_TOPICS_CSV]
  KEEP_TOPICS_CSV, if given, is the only set of topics copied through -- pass
  "/hesai/pandar,/alphasense/imu" to drop the five unused camera streams and turn
  the 5.8 GB source bag into a ~0.9 GB one in the same pass.
"""
import sys
import numpy as np
import rosbag
from sensor_msgs.msg import PointCloud2, PointField

IN_DTYPE = np.dtype({"names": ["x", "y", "z", "intensity", "timestamp", "ring"],
                     "formats": ["<f4", "<f4", "<f4", "<f4", "<f8", "<u2"],
                     "offsets": [0, 4, 8, 16, 24, 32], "itemsize": 48})
OUT_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("intensity", "<f4"), ("time", "<f4")])

FIELDS = []
for i, n in enumerate(["x", "y", "z", "intensity", "time"]):
    f = PointField()
    f.name = n
    f.offset = 4 * i
    f.datatype = PointField.FLOAT32
    f.count = 1
    FIELDS.append(f)

src, dst = sys.argv[1], sys.argv[2]
keep = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
inb = rosbag.Bag(src)
outb = rosbag.Bag(dst, "w")
n = 0
tmin, tmax = 1e9, -1e9
for topic, msg, t in inb.read_messages(topics=sorted(keep) if keep else None):
    if topic != "/hesai/pandar":
        outb.write(topic, msg, t)
        continue
    a = np.frombuffer(msg.data, dtype=IN_DTYPE, count=msg.width * msg.height)
    rel = (a["timestamp"] - msg.header.stamp.to_sec()).astype(np.float64)
    order = np.argsort(rel, kind="stable")
    out = np.empty(a.size, dtype=OUT_DTYPE)
    out["x"] = a["x"][order]
    out["y"] = a["y"][order]
    out["z"] = a["z"][order]
    out["intensity"] = a["intensity"][order]
    out["time"] = rel[order].astype(np.float32)
    tmin = min(tmin, rel.min()); tmax = max(tmax, rel.max())
    m = PointCloud2()
    m.header = msg.header
    m.height = 1
    m.width = a.size
    m.fields = FIELDS
    m.is_bigendian = False
    m.point_step = 20
    m.row_step = 20 * a.size
    m.is_dense = True
    m.data = out.tobytes()
    outb.write(topic, m, t)
    n += 1
inb.close()
outb.close()
print("rewrote %d /hesai/pandar messages; per-point time range %.6f .. %.6f s"
      % (n, tmin, tmax))
