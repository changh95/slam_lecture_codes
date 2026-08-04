#!/usr/bin/env bash
# Run FAST-LIVO2 headless on a FAST-LIVO2-Dataset (Livox Avia) bag.
#
# Mounts expected inside the container:
#   /data                                                bag dir, read-only
#   /out                                                 writable log dir
#   /catkin_ws/src/FAST-LIVO2/Log/result                 writable, trajectory lands here
#   /catkin_ws/src/FAST-LIVO2/config/avia.yaml           the config for THIS sequence
# Env:
#   BAG=/data/Retail_Street.bag   RATE=1.0   DURATION=   (empty = whole bag)
#
# Two things about mapping_avia.launch that matter:
#   * rviz defaults to TRUE, so `rviz:=false` is required for a headless run.
#   * it starts an image_transport republish node (/left_camera/image/compressed
#     -> /left_camera/image), because these bags ship compressed images.

set -uo pipefail

BAG="${BAG:-/data/Retail_Street.bag}"
RATE="${RATE:-1.0}"
DURATION="${DURATION:-}"

source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash

export ROS_MASTER_URI=http://localhost:11311
export ROS_HOSTNAME=localhost
export ROS_IP=127.0.0.1

echo "[run] roscore (container-private netns, no --net=host)"
roscore >/out/roscore.log 2>&1 &
ROSCORE_PID=$!
until rostopic list >/dev/null 2>&1; do sleep 0.5; done
echo "[run] roscore up"

echo "[run] roslaunch fast_livo mapping_avia.launch rviz:=false"
roslaunch fast_livo mapping_avia.launch rviz:=false --wait >/out/fastlivo.log 2>&1 &
LAUNCH_PID=$!

for _ in $(seq 1 60); do
    rostopic list 2>/dev/null | grep -q '^/aft_mapped_to_init$' && break
    sleep 1
done
sleep 5

PLAY_ARGS=(--clock --quiet -r "$RATE")
[ -n "$DURATION" ] && PLAY_ARGS+=(-u "$DURATION")
# The bag path must precede any --topics filter: rosbag play's --topics is greedy.
PLAY_ARGS+=("$BAG")
echo "[run] rosbag play ${PLAY_ARGS[*]}"
SECONDS=0
rosbag play "${PLAY_ARGS[@]}" >/out/rosbag_play.log 2>&1
echo "[run] bag playback wall-clock: ${SECONDS} s"

echo "[run] draining 10 s"
sleep 10

# FAST-LIVO2 flushes its trajectory (and PCD, when enabled) only as main()
# unwinds after SIGINT -- a hard kill loses it.
echo "[run] SIGINT -> nodes"
kill -INT $LAUNCH_PID 2>/dev/null
sleep 15
kill -KILL $LAUNCH_PID 2>/dev/null
kill -INT $ROSCORE_PID 2>/dev/null
sleep 2

echo "[run] done. Log/result:"
ls -la /catkin_ws/src/FAST-LIVO2/Log/result/ 2>/dev/null
