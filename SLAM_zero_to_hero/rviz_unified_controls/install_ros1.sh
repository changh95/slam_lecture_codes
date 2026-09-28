#!/usr/bin/env bash
# Build the ROS 1 package and install it straight into /opt/ros/$ROS_DISTRO, so
# rviz finds the plugins with no extra workspace to source.
#   install_ros1.sh [path/to/rviz_unified_controls]
set -e
SRC="$(cd "${1:-$(dirname "$0")}" && pwd)/ros1"
source /opt/ros/${ROS_DISTRO:-noetic}/setup.bash
BUILD=$(mktemp -d)
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/ros/$ROS_DISTRO -DCATKIN_BUILD_BINARY_PACKAGE=ON
cmake --build "$BUILD" -j"$(nproc)"
cmake --install "$BUILD"
rm -rf "$BUILD"
