#!/usr/bin/env python3
"""
Custom rosbag player that bypasses `rosbag play`.

Why: on noetic/aarch64, the upstream `rosbag play` binary stalls on
cow_and_lady (Duration stuck at 0.000000) apparently due to a time
arithmetic bug with bags whose stamps are 10+ years in the past. This
reader uses the `rosbag` Python API to iterate messages and publishes
them via rospy, rewriting header.stamp to a recent wall-clock time so
the whole system runs on "now" instead of 2016.

Usage:
    python3 bag_player.py /data/input.bag [rate]

Where `rate` is the playback speed multiplier (default 1.0).
"""
import sys
import time
import rospy
import rosbag
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped


TOPIC_TYPES = {
    "/camera/depth_registered/points": PointCloud2,
    "/kinect/vrpn_client/estimated_transform": TransformStamped,
}


def main():
    if len(sys.argv) < 2:
        print("usage: bag_player.py <bag> [rate]", file=sys.stderr)
        sys.exit(1)
    bag_path = sys.argv[1]
    rate = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    rospy.init_node("bag_player", anonymous=True)

    pubs = {
        t: rospy.Publisher(t, mt, queue_size=20, latch=False)
        for t, mt in TOPIC_TYPES.items()
    }
    rospy.sleep(0.5)  # let XMLRPC registration settle

    rospy.loginfo(f"opening {bag_path}")
    bag = rosbag.Bag(bag_path, "r")
    topics = list(TOPIC_TYPES.keys())

    wall_start = None
    bag_start = None
    count = {t: 0 for t in topics}
    t_last_log = time.monotonic()

    for topic, msg, t in bag.read_messages(topics=topics):
        if rospy.is_shutdown():
            break
        bag_t = t.to_sec()
        if bag_start is None:
            bag_start = bag_t
            wall_start = time.monotonic()
            rospy.loginfo(f"first msg at bag_t={bag_t:.3f}, publishing at real rate x{rate}")
        bag_elapsed = (bag_t - bag_start) / rate
        real_elapsed = time.monotonic() - wall_start
        sleep_s = bag_elapsed - real_elapsed
        if sleep_s > 0:
            time.sleep(sleep_s)

        # Rewrite header.stamp to current ros time so downstream nodes
        # don't get confused by 2016-era timestamps.
        now = rospy.Time.now()
        if hasattr(msg, "header"):
            msg.header.stamp = now
        pubs[topic].publish(msg)
        count[topic] += 1

        now_mono = time.monotonic()
        if now_mono - t_last_log >= 2.0:
            rospy.loginfo(
                f"bag_t={bag_t - bag_start:.1f}s counts={count}"
            )
            t_last_log = now_mono

    rospy.loginfo(f"bag finished. final counts: {count}")
    bag.close()


if __name__ == "__main__":
    main()
