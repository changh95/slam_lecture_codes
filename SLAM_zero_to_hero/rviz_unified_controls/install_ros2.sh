#!/usr/bin/env bash
# Build the ROS 2 package and install it straight into /opt/ros/$ROS_DISTRO, so
# rviz2 finds the plugins with no extra workspace to source.
#   install_ros2.sh [path/to/rviz_unified_controls]
set -e
SRC="$(cd "${1:-$(dirname "$0")}" && pwd)/ros2"
DISTRO=${ROS_DISTRO:-$(ls /opt/ros | head -1)}
source /opt/ros/$DISTRO/setup.bash
BUILD=$(mktemp -d)
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/ros/$DISTRO
cmake --build "$BUILD" -j"$(nproc)"
cmake --install "$BUILD"
rm -rf "$BUILD"
