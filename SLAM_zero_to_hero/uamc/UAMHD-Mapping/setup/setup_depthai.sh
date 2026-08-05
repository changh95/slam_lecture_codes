#!/bin/bash
# One-time setup for depthai-ros (Oak-D Pro) on ROS 2 Humble.
# Run this once before using build_ws.sh.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[1/2] Installing depthai and missing ROS 2 dependencies..."
sudo apt-get update
sudo apt-get install -y \
    ros-humble-depthai \
    ros-humble-image-transport-plugins \
    ros-humble-image-pipeline \
    ros-humble-camera-calibration \
    ros-humble-diagnostic-updater \
    ros-humble-vision-msgs \
    ros-humble-ffmpeg-image-transport \
    ros-humble-ffmpeg-image-transport-msgs \
    libusb-1.0-0-dev

echo "[2/2] Ensuring depthai-ros is on the humble branch..."
DEPTHAI_ROS_DIR="src/depthai-ros"
if [ -d "$DEPTHAI_ROS_DIR" ]; then
    git -C "$DEPTHAI_ROS_DIR" checkout humble
else
    echo "ERROR: $DEPTHAI_ROS_DIR not found."
    echo "Run: git clone https://github.com/luxonis/depthai-ros.git src/depthai-ros"
    exit 1
fi

echo ""
echo "Setup complete. Now run ./build_ws.sh to build the workspace."
