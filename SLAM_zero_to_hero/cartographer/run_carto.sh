#!/usr/bin/env bash
# Run cartographer_offline_node on a Hilti 2022 bag, then export everything a human
# or a script can check: pbstream, TUM trajectory, 2D occupancy pgm, 3D point cloud.
#
# Bind mounts expected inside the container:
#   /data   bag directory (read-only)
#   /cfg    directory holding the .lua (its `include`s resolve to Cartographer's own
#           installed configuration_files dir, so /cfg only needs our file)
#   /urdf   directory holding hilti_alphasense_pandar.urdf
#   /out    writable output directory
#   /scripts helper python
#
# Env:
#   CFG        lua basename            (default hilti_3d_lio.lua)
#   BAG        bag path                (default /data/hilti_deskew.bag)
#   TAG        output subdir under /out (default run)
#   ASSETS     1 => also run cartographer_assets_writer (default 1)
#   ASSETS_CFG assets_writer lua       (default assets_writer_hilti.lua = 3D cloud)
set -euo pipefail

CFG="${CFG:-hilti_3d_lio.lua}"
BAG="${BAG:-/data/hilti_deskew.bag}"
TAG="${TAG:-run}"
ASSETS="${ASSETS:-1}"
ASSETS_CFG="${ASSETS_CFG:-assets_writer_hilti.lua}"

source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash

O="/out/$TAG"
mkdir -p "$O"
CARTO_BIN=/catkin_ws/devel/.private/cartographer_ros/lib/cartographer_ros

echo "[run] roscore"
roscore >"$O/roscore.log" 2>&1 &
ROSCORE_PID=$!
for i in $(seq 1 30); do rostopic list >/dev/null 2>&1 && break; sleep 1; done

echo "[run] cartographer_offline_node  cfg=$CFG  bag=$BAG"
T0=$SECONDS
"$CARTO_BIN/cartographer_offline_node" \
  -configuration_directory /cfg \
  -configuration_basenames "$CFG" \
  -urdf_filenames /urdf/hilti_alphasense_pandar.urdf \
  -bag_filenames "$BAG" \
  -save_state_filename "$O/map.pbstream" \
  points2:=/hesai/pandar \
  imu:=/alphasense/imu \
  >"$O/offline.log" 2>&1
echo "[run] offline node wall clock: $((SECONDS-T0)) s"
echo "[run] pbstream: $(ls -l "$O/map.pbstream" | awk '{print $5}') bytes"

echo "[run] 2D occupancy projection (pbstream_to_ros_map, 0.05 m/px)"
"$CARTO_BIN/cartographer_pbstream_to_ros_map" \
  -pbstream_filename="$O/map.pbstream" \
  -map_filestem="$O/map" \
  -resolution=0.05 >"$O/pbstream_to_ros_map.log" 2>&1 || \
  echo "[run] WARNING pbstream_to_ros_map failed (expected for a pure-3D pbstream)"

echo "[run] trajectory -> rosbag -> TUM"
"$CARTO_BIN/cartographer_dev_pbstream_trajectories_to_rosbag" \
  -input "$O/map.pbstream" -output "$O/traj.bag" \
  >"$O/traj_export.log" 2>&1
python3 /scripts/tfbag_to_tum.py "$O/traj.bag" "$O/carto_tum.txt" >>"$O/traj_export.log" 2>&1
wc -l "$O/carto_tum.txt"

if [ "$ASSETS" = "1" ]; then
  echo "[run] assets_writer (3D cloud + level slices)"
  "$CARTO_BIN/cartographer_assets_writer" \
    -configuration_directory /cfg \
    -configuration_basename "$ASSETS_CFG" \
    -urdf_filename /urdf/hilti_alphasense_pandar.urdf \
    -bag_filenames "$BAG" \
    -pose_graph_filename "$O/map.pbstream" \
    -output_file_prefix "$O/assets_" \
    >"$O/assets.log" 2>&1 || echo "[run] WARNING assets_writer failed, see assets.log"
fi

kill -INT $ROSCORE_PID 2>/dev/null || true
sleep 2
echo "[run] done"
ls -l "$O"
