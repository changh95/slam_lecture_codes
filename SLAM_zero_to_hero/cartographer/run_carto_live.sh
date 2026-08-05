#!/usr/bin/env bash
# Live (online) equivalent of run_carto.sh: cartographer_node + rosbag play, with the
# LiDAR<-IMU extrinsic supplied by a tf2_ros static_transform_publisher instead of a
# URDF. Proves the same config works in the online pipeline, which is what
# hilti_3d.launch would drive if roslaunch could find it.
set -euo pipefail
CFG="${CFG:-hilti_3d_lio.lua}"
BAG="${BAG:-/data/lidar_imu_full_t.bag}"
TAG="${TAG:-live}"
RATE="${RATE:-3.0}"

source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash
O=/out/$TAG; mkdir -p "$O"
CARTO_BIN=/catkin_ws/devel/.private/cartographer_ros/lib/cartographer_ros

roscore >"$O/roscore.log" 2>&1 & ROSCORE=$!
for i in $(seq 1 30); do rostopic list >/dev/null 2>&1 && break; sleep 1; done
rosparam set /use_sim_time true

# x y z qx qy qz qw parent child   -- T_imu_lidar, see the URDF comment for why.
rosrun tf2_ros static_transform_publisher \
  -0.001 -0.00855 0.055 0.7071068 -0.7071068 0 0 imu_sensor_frame PandarXT-32 \
  >"$O/stf.log" 2>&1 & STF=$!

"$CARTO_BIN/cartographer_node" \
  -configuration_directory /cfg -configuration_basename "$CFG" \
  points2:=/hesai/pandar imu:=/alphasense/imu >"$O/node.log" 2>&1 & NODE=$!
sleep 3

rosbag play --clock --quiet -r "$RATE" "$BAG" >"$O/play.log" 2>&1
sleep 5

rosservice call /finish_trajectory 0 >"$O/finish.log" 2>&1 || true
sleep 3
rosservice call /write_state "{filename: '$O/map.pbstream', include_unfinished_submaps: true}" \
  >"$O/write_state.log" 2>&1
sleep 2
kill -INT $NODE $STF 2>/dev/null || true; sleep 3
kill -INT $ROSCORE 2>/dev/null || true; sleep 1

echo "[live] pbstream: $(ls -l "$O/map.pbstream" | awk '{print $5}') bytes"
"$CARTO_BIN/cartographer_dev_pbstream_trajectories_to_rosbag" \
  -input "$O/map.pbstream" -output "$O/traj.bag" >"$O/traj_export.log" 2>&1
python3 /scripts/tfbag_to_tum.py "$O/traj.bag" "$O/carto_tum.txt" >>"$O/traj_export.log" 2>&1
wc -l "$O/carto_tum.txt"
grep -icE "could not|lookup would require|no transform|Dropped" "$O/node.log" || true
