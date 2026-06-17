#!/usr/bin/env bash
# Run cartographer_offline_node on the Hilti exp14_basement_2.bag end-to-end.
# Inputs:
#   /data/<bag>     bag mounted read-only
#   /out            writable results dir
# Outputs:
#   /out/hilti_basement.pbstream
#   /out/hilti_basement.pgm + .yaml (2D occupancy projection)

set -euo pipefail

BAG="${1:-/data/exp14_basement_2.bag}"
OUT_PREFIX="/out/hilti_basement"

source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash

CFG_DIR="$(rospack find cartographer_ros)/configuration_files"

# The Dockerfile COPY landed hilti_3d.lua one directory up from where rospack
# resolves cartographer_ros (parent vs nested package dir). Symlink it into the
# resolved configuration_files dir so cartographer_offline_node can find it.
SRC_LUA="/catkin_ws/src/cartographer/cartographer_ros/configuration_files/hilti_3d.lua"
if [ -f "$SRC_LUA" ] && [ ! -e "${CFG_DIR}/hilti_3d.lua" ]; then
  ln -sf "$SRC_LUA" "${CFG_DIR}/hilti_3d.lua"
fi

# cartographer_offline_node calls ros::init() and waits for a master, so we must
# spin up roscore even though the node otherwise reads the bag offline.
echo "[run] starting roscore"
roscore &
ROSCORE_PID=$!
sleep 3

echo "[run] cartographer_offline_node on $BAG"
rosrun cartographer_ros cartographer_offline_node \
  -configuration_directory "$CFG_DIR" \
  -configuration_basenames hilti_3d.lua \
  -bag_filenames "$BAG" \
  -urdf_filenames "" \
  -save_state_filename "${OUT_PREFIX}.pbstream" \
  -load_frozen_state=false \
  points2:=/hesai/pandar

echo "[run] pbstream saved -> ${OUT_PREFIX}.pbstream"
ls -lh "${OUT_PREFIX}.pbstream"

echo "[run] generating ROS map (2D occupancy projection)"
rosrun cartographer_ros cartographer_pbstream_to_ros_map \
  -pbstream_filename="${OUT_PREFIX}.pbstream" \
  -map_filestem="${OUT_PREFIX}"

kill -INT $ROSCORE_PID 2>/dev/null || true
wait 2>/dev/null || true
echo "[run] done"
ls -lh /out/
