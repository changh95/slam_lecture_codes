#!/bin/bash
# Build the entire slam_ws workspace (Livox drivers + depthai-ros).
# Prerequisites: run ./setup_depthai.sh once to install system dependencies.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[1/3] Sourcing ROS 2 Humble..."
source /opt/ros/humble/setup.bash

echo "[2/3] Preparing livox_ros_driver2 for ROS 2..."
DRIVER2_DIR="src/livox_ros_driver2"
if [ -d "$DRIVER2_DIR" ]; then
    cp -f "$DRIVER2_DIR/package_ROS2.xml" "$DRIVER2_DIR/package.xml"
    cp -rf "$DRIVER2_DIR/launch_ROS2/" "$DRIVER2_DIR/launch/"
else
    echo "  WARNING: $DRIVER2_DIR not found, skipping."
fi

echo "[3/3] Building workspace..."
colcon build \
    --symlink-install \
    --packages-skip depthai_filters \
    --cmake-args \
        -DROS_EDITION=ROS2 \
        -DHUMBLE_ROS=humble \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_SHARED_LIBS=ON

echo ""
echo "Build complete. Source the workspace with:"
echo "  source $SCRIPT_DIR/install/setup.bash"