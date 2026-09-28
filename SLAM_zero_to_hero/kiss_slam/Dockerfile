FROM ubuntu:jammy

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y build-essential cmake git wget \
    python3 python3-pip python3-dev \
    libeigen3-dev libgl1-mesa-dev libgl1-mesa-dri xvfb && \
    apt-get clean

# Install KISS-SLAM (v0.0.2) and the pure-Python rosbags reader so the rosbag
# dataloader works without a ROS install. kiss-icp and open3d are pinned to the
# versions v0.0.2 was verified with.
RUN pip3 install kiss-slam==0.0.2 kiss-icp==1.3.0 open3d==0.19.0 rosbags==0.10.4

# Start the --visualize viewer playing instead of paused. v0.0.2 hard-codes
# play_crun = False and has no option for it, so the window sat at frame 0 until
# SPACE was pressed. KISS_SLAM_AUTOPLAY=0 restores the upstream paused start.
RUN f=/usr/local/lib/python3.10/dist-packages/kiss_slam/tools/visualizer.py && \
    sed -i 's/^        self.play_crun = False$/        self.play_crun = os.environ.get("KISS_SLAM_AUTOPLAY", "1") != "0"/' "$f" && \
    grep -q 'KISS_SLAM_AUTOPLAY' "$f" && grep -q '^import os' "$f"

# Unattended viewer capture (auto-plays the Open3D window, writes PNGs).
COPY scripts/capture_viewer.py /opt/kiss_slam/capture_viewer.py

WORKDIR /data
