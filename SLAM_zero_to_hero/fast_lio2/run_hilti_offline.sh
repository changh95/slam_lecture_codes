#!/usr/bin/env bash
# Run FAST-LIO2 headless on the Hilti 2022 exp14_basement_2 bag.
# Mounts expected inside the container:
#   /data                bag dir, read-only
#   /out                 writable results dir
#   /catkin_ws/src/FAST_LIO/config/hilti_pandarxt32.yaml   (bind-mounted)
#   /catkin_ws/src/FAST_LIO/launch/mapping_hilti.launch    (bind-mounted)
#   /scripts/odom_to_tum.py                                (bind-mounted)
# Env:
#   BAG=/data/exp14_basement_2.bag   RATE=1.0   DURATION=   (empty = whole bag)
#   CONFIG=hilti_pandarxt32          SAVE_PCD=0
# Outputs in /out: fastlio_traj_tum.txt, odometry_raw.csv, fastlio_stdout.log,
#                  pos_log.txt, (optional) scans.pcd

set -uo pipefail

BAG="${BAG:-/data/exp14_basement_2.bag}"
RATE="${RATE:-1.0}"
DURATION="${DURATION:-}"
CONFIG="${CONFIG:-hilti_pandarxt32}"
SAVE_PCD="${SAVE_PCD:-0}"
RELAY="${RELAY:-0}"

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

if [ "$RELAY" = "1" ]; then
  echo "[run] hesai->velodyne relay"
  python3 /scripts/hesai_to_velodyne.py >/out/relay.log 2>&1 &
  RELAY_PID=$!
fi

PCD_ARG=false
[ "$SAVE_PCD" = "1" ] && PCD_ARG=true
echo "[run] roslaunch fast_lio mapping_hilti.launch config:=$CONFIG rviz:=false pcd_save:=$PCD_ARG"
roslaunch fast_lio mapping_hilti.launch config:="$CONFIG" rviz:=false pcd_save:="$PCD_ARG" \
    >/out/fastlio_stdout.log 2>&1 &
LAUNCH_PID=$!

# wait for the mapper to advertise
for _ in $(seq 1 60); do
  rostopic list 2>/dev/null | grep -q '^/Odometry$' && break
  sleep 1
done
echo "[run] /Odometry advertised: $(rostopic list 2>/dev/null | grep -c '^/Odometry$')"

python3 /scripts/odom_to_tum.py /out/fastlio_traj_tum.txt /Odometry >/out/tum_logger.log 2>&1 &
TUM_PID=$!
rostopic echo -p /Odometry >/out/odometry_raw.csv 2>/dev/null &
ECHO_PID=$!
sleep 2

# NOTE: no --clock / use_sim_time. FAST-LIO drives itself off message header
# stamps, so wall-clock ROS time is fine and avoids a /clock dependency.
# The bag path MUST come before --topics: rosbag play's --topics is greedy and
# will otherwise eat the bag filename as a third topic name.
PLAY_ARGS=(-r "$RATE")
[ -n "$DURATION" ] && PLAY_ARGS+=(-u "$DURATION")
PLAY_ARGS+=("$BAG" --topics /hesai/pandar /alphasense/imu)
echo "[run] rosbag play ${PLAY_ARGS[*]}"
SECONDS=0
rosbag play "${PLAY_ARGS[@]}" >/out/rosbag_play.log 2>&1
echo "[run] bag playback wall-clock: ${SECONDS} s"

echo "[run] draining 10 s"
sleep 10

echo "[run] SIGINT -> nodes (flushes PCD / pos_log)"
kill -INT $LAUNCH_PID 2>/dev/null
sleep 15
kill -INT $TUM_PID $ECHO_PID 2>/dev/null
[ "${RELAY_PID:-}" ] && kill -INT $RELAY_PID 2>/dev/null
sleep 3

cp -f /catkin_ws/src/FAST_LIO/Log/pos_log.txt /out/pos_log.txt 2>/dev/null || true
if [ "$SAVE_PCD" = "1" ]; then
  cp -f /catkin_ws/src/FAST_LIO/PCD/scans.pcd /out/scans.pcd 2>/dev/null || true
fi

kill -INT $ROSCORE_PID 2>/dev/null
sleep 2
echo "[run] done. outputs:"
ls -la /out/
echo "[run] poses logged: $(wc -l < /out/fastlio_traj_tum.txt 2>/dev/null || echo 0)"
