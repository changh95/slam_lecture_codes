"""Convert the bag produced by cartographer_dev_pbstream_trajectories_to_rosbag
into a TUM trajectory file (t tx ty tz qx qy qz qw).

That tool writes one geometry_msgs/TransformStamped per OPTIMIZED trajectory node,
on a topic named 'trajectory_<id>', with header.frame_id = 'map' (parent) and
child_frame_id = 'trajectory_<id>'. The transform is map_T_tracking_frame, i.e.
the pose of the tracking frame in the map frame -- exactly the TUM convention.
"""
import sys
import rosbag

inp, out = sys.argv[1], sys.argv[2]
traj_id = sys.argv[3] if len(sys.argv) > 3 else "0"
topic = "trajectory_" + traj_id

rows = []
bag = rosbag.Bag(inp)
for tp, msg, t in bag.read_messages():
    if tp != topic:
        continue
    tr = msg.transform.translation
    q = msg.transform.rotation
    rows.append((msg.header.stamp.to_sec(), tr.x, tr.y, tr.z, q.x, q.y, q.z, q.w))
bag.close()
rows.sort(key=lambda r: r[0])
with open(out, "w") as f:
    for r in rows:
        f.write("%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n" % r)
print("wrote %d poses from topic %s -> %s" % (len(rows), topic, out))
