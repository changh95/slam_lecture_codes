#!/bin/bash
# bag_record.sh — Record all sensor topics to a ROS 2 bag.
# Run this AFTER ./sensor_bringup.sh has launched the drivers.
#
# Livox topics are named after each device's broadcast code / IP, which is
# unique per unit. Discover yours with:  ros2 topic list | grep livox
# Then set LIDAR_A / LIDAR_B below (or override via environment).
set -e

# ── Per-unit Livox topic identifiers (CHANGE THESE for your hardware) ──
# Examples of what an identifier looks like: an IP-derived suffix like
# "192_168_1_XXX", or a broadcast-code suffix like "XXXXXXXXXXXXXXX".
LIDAR_A="${LIDAR_A:-<LIVOX_A_ID>}"   # e.g. primary LiDAR's topic suffix
LIDAR_B="${LIDAR_B:-<LIVOX_B_ID>}"   # e.g. secondary LiDAR's topic suffix

OUT="${OUT:-handheld_$(date +%Y%m%d_%H%M%S)}"

ros2 bag record -o "$OUT" \
    "/livox/imu_${LIDAR_A}" \
    "/livox/imu_${LIDAR_B}" \
    "/livox/lidar_${LIDAR_A}" \
    "/livox/lidar_${LIDAR_B}" \
    /oak/rgb/image_raw \
    /oak/rgb/camera_info
