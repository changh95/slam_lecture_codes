#!/usr/bin/env python3
"""Evaluate an SVO Pro run on a UZH-FPV bag against the bag's own ground truth.

Reads the ground-truth poses out of the source bag and the estimated poses out
of the bag recorded during the run, writes both as TUM-format trajectories, and
reports absolute trajectory error via evo.

SVO's world frame is gravity-aligned but otherwise arbitrary, so the comparison
is SE(3)-aligned. Monocular VO is additionally scale-free, hence --correct_scale.
"""
import argparse
import os
import subprocess
import sys

import rosbag

# Ground truth appears under different topic names across the UZH-FPV releases,
# so match on the topic rather than hard-coding one name.
GT_HINTS = ("groundtruth", "ground_truth", "gt", "vicon", "leica", "optitrack")


def find_topics(bag_path):
    with rosbag.Bag(bag_path, "r") as bag:
        return {t: (i.msg_type, i.message_count)
                for t, i in bag.get_type_and_topic_info().topics.items()}


def pick_gt_topic(topics):
    pose_types = ("geometry_msgs/PoseStamped",
                  "geometry_msgs/PoseWithCovarianceStamped",
                  "geometry_msgs/TransformStamped",
                  "nav_msgs/Odometry")
    cands = [(t, v) for t, v in topics.items()
             if v[0] in pose_types and any(h in t.lower() for h in GT_HINTS)]
    if not cands:
        return None
    # Prefer the densest ground-truth stream, breaking ties by topic name so the
    # choice does not depend on dict ordering. UZH-FPV publishes the same poses
    # on both /groundtruth/odometry and /groundtruth/pose at the same rate, so
    # without the tie-break the reported metric could silently switch topics
    # between runs.
    return min(cands, key=lambda kv: (-kv[1][1], kv[0]))[0]


def pose_from_msg(msg, msg_type):
    """Return (tx, ty, tz, qx, qy, qz, qw) for the supported pose messages."""
    if msg_type == "geometry_msgs/PoseStamped":
        p, q = msg.pose.position, msg.pose.orientation
    elif msg_type == "geometry_msgs/PoseWithCovarianceStamped":
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
    elif msg_type == "nav_msgs/Odometry":
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
    elif msg_type == "geometry_msgs/TransformStamped":
        p, q = msg.transform.translation, msg.transform.rotation
    else:
        raise ValueError("unsupported pose type: %s" % msg_type)
    return p.x, p.y, p.z, q.x, q.y, q.z, q.w


def write_tum(bag_path, topic, msg_type, out_path):
    n = 0
    with rosbag.Bag(bag_path, "r") as bag, open(out_path, "w") as fh:
        for _, msg, t in bag.read_messages(topics=[topic]):
            # Prefer the message's own stamp; fall back to bag receipt time.
            stamp = msg.header.stamp if getattr(msg, "header", None) else t
            if stamp.to_sec() == 0.0:
                stamp = t
            vals = pose_from_msg(msg, msg_type)
            fh.write("%.9f %s\n" % (stamp.to_sec(),
                                    " ".join("%.9f" % v for v in vals)))
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-bag", required=True, help="source UZH-FPV bag (has ground truth)")
    ap.add_argument("--est-bag", required=True, help="bag recorded from /svo/pose_imu")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--est-topic", default="/svo/pose_imu")
    ap.add_argument("--gt-topic", default=None, help="override ground-truth topic")
    ap.add_argument("--correct_scale", action="store_true",
                    help="also solve for scale (monocular)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    gt_topics = find_topics(args.gt_bag)
    gt_topic = args.gt_topic or pick_gt_topic(gt_topics)
    if gt_topic is None:
        print("No ground-truth pose topic found in %s." % args.gt_bag)
        print("Topics present:")
        for t, (ty, c) in sorted(gt_topics.items()):
            print("   %-40s %-45s %d msgs" % (t, ty, c))
        return 2
    gt_type = gt_topics[gt_topic][0]

    est_topics = find_topics(args.est_bag)
    if args.est_topic not in est_topics:
        print("Estimated-pose topic %s missing from %s -- SVO published nothing."
              % (args.est_topic, args.est_bag))
        return 2
    est_type = est_topics[args.est_topic][0]

    gt_txt = os.path.join(args.out, "gt_tum.txt")
    est_txt = os.path.join(args.out, "svo_tum.txt")
    n_gt = write_tum(args.gt_bag, gt_topic, gt_type, gt_txt)
    n_est = write_tum(args.est_bag, args.est_topic, est_type, est_txt)

    print("ground truth: %s (%s) -> %d poses" % (gt_topic, gt_type, n_gt))
    print("estimate    : %s (%s) -> %d poses" % (args.est_topic, est_type, n_est))
    if n_est < 10:
        print("Too few estimated poses to evaluate; the pipeline did not track.")
        return 2

    cmd = ["evo_ape", "tum", gt_txt, est_txt, "--align",
           "--t_max_diff", "0.05",
           "--save_results", os.path.join(args.out, "ape.zip"),
           "--plot_mode", "xyz",
           "--save_plot", os.path.join(args.out, "ape_plot")]
    if args.correct_scale:
        cmd.append("--correct_scale")
    print("+ " + " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
