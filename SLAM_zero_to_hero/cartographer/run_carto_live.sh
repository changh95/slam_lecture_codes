#!/usr/bin/env bash
# Live (online) equivalent of run_carto.sh: cartographer_node + rosbag play, with the
# LiDAR<-IMU extrinsic supplied by a tf2_ros static_transform_publisher instead of a
# URDF, and rviz (cartographer_rviz Submaps + trajectory) watching it.
#
# Runs inside the container, with the same bind mounts as run_carto.sh.
#
# Env:
#   CFG       lua basename              (default hilti_outdoor_3d.lua)
#   BAG       bag path                  (default /data/exp21_outside_building_carto.bag)
#   RVIZ_CFG  rviz config under /cfg    (default hilti_outdoor_3d.rviz;
#                                        hilti_3d.rviz frames the exp14 basement)
#   TAG       output subdir under /out  (default live)
#   RATE      rosbag play rate          (default 1.0)
#   RVIZ      1 => start rviz on $DISPLAY (default 1)
#   XVFB      1 => start a private Xvfb :99 and render rviz there (headless, default 0)
#   SHOT      if set, PNG path; the rviz window is captured there when the bag ends
set -euo pipefail
CFG="${CFG:-hilti_outdoor_3d.lua}"
BAG="${BAG:-/data/exp21_outside_building_carto.bag}"
RVIZ_CFG="${RVIZ_CFG:-hilti_outdoor_3d.rviz}"
TAG="${TAG:-live}"
RATE="${RATE:-1.0}"
RVIZ="${RVIZ:-1}"
XVFB="${XVFB:-0}"
SHOT="${SHOT:-}"

export DISABLE_ROS1_EOL_WARNINGS=1   # rviz otherwise opens a modal EOL dialog over the map
source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash
O=/out/$TAG; mkdir -p "$O"
CARTO_BIN=/catkin_ws/devel/.private/cartographer_ros/lib/cartographer_ros

PIDS=()
cleanup() { set +e; kill -INT "${PIDS[@]}" 2>/dev/null; sleep 2; kill "${PIDS[@]}" 2>/dev/null; }
trap cleanup EXIT

if [ "$XVFB" = "1" ]; then
  export DISPLAY=:99
  Xvfb :99 -screen 0 1600x900x24 -nolisten tcp >"$O/xvfb.log" 2>&1 & PIDS+=($!)
  for _ in $(seq 1 40); do xdpyinfo >/dev/null 2>&1 && break; sleep 0.25; done
fi

roscore >"$O/roscore.log" 2>&1 & PIDS+=($!)
for i in $(seq 1 30); do rostopic list >/dev/null 2>&1 && break; sleep 1; done
rosparam set /use_sim_time true

# x y z qx qy qz qw parent child   -- T_imu_lidar, see the URDF comment for why.
rosrun tf2_ros static_transform_publisher \
  -0.001 -0.00855 0.055 0.7071068 -0.7071068 0 0 imu_sensor_frame PandarXT-32 \
  >"$O/stf.log" 2>&1 & PIDS+=($!)

"$CARTO_BIN/cartographer_node" \
  -configuration_directory /cfg -configuration_basename "$CFG" \
  points2:=/hesai/pandar imu:=/alphasense/imu >"$O/node.log" 2>&1 & NODE=$!
PIDS+=($NODE)

if [ "$RVIZ" = "1" ]; then
  rviz -d "/cfg/$RVIZ_CFG" >"$O/rviz.log" 2>&1 & PIDS+=($!)
  sleep 5
fi
sleep 3

T0=$SECONDS
rosbag play --clock --quiet -r "$RATE" "$BAG" >"$O/play.log" 2>&1
echo "[live] bag played in $((SECONDS-T0)) s (rate $RATE)"
sleep 5

if [ -n "$SHOT" ]; then
  # Grab only the rviz window (on a shared desktop, root would include everything else).
  WIN=$(xdotool search --name -- "- RViz" 2>/dev/null | head -1 || true)
  import -window "${WIN:-root}" "$SHOT" && echo "[live] screenshot -> $SHOT"
fi

rosservice call /finish_trajectory 0 >"$O/finish.log" 2>&1 || true
sleep 3
rosservice call /write_state "{filename: '$O/map.pbstream', include_unfinished_submaps: true}" \
  >"$O/write_state.log" 2>&1
sleep 2
cleanup; trap - EXIT

echo "[live] pbstream: $(ls -l "$O/map.pbstream" | awk '{print $5}') bytes"
"$CARTO_BIN/cartographer_dev_pbstream_trajectories_to_rosbag" \
  -input "$O/map.pbstream" -output "$O/traj.bag" >"$O/traj_export.log" 2>&1
python3 /scripts/tfbag_to_tum.py "$O/traj.bag" "$O/carto_tum.txt" >>"$O/traj_export.log" 2>&1
wc -l "$O/carto_tum.txt"
grep -icE "could not|lookup would require|no transform|Dropped" "$O/node.log" || true
