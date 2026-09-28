# FAST-LIO2 (hku-mars/FAST_LIO master = FAST-LIO2) on ROS Noetic, CPU only.
FROM docker.io/ros:noetic

ARG DEBIAN_FRONTEND=noninteractive
# Pinned upstream commits (the versions every number in README/NOTES was measured with)
ARG LIVOX_SDK_COMMIT=9306596a2bf15c1343bc023b497465ed0a32909d
ARG LIVOX_ROS_DRIVER_COMMIT=3d240d5
ARG FAST_LIO_COMMIT=7cc4175

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y \
    build-essential cmake git wget \
    libeigen3-dev libpcl-dev libusb-1.0-0-dev \
    ros-noetic-pcl-ros ros-noetic-eigen-conversions && \
    apt-get clean

# Livox SDK v1 (needed by livox_ros_driver, which FAST_LIO depends on for its msg types)
RUN git clone https://github.com/Livox-SDK/Livox-SDK.git && \
    cd Livox-SDK && git checkout ${LIVOX_SDK_COMMIT} && \
    mkdir -p build && cd build && \
    cmake .. && make -j$(nproc) && make install && \
    cd / && rm -rf Livox-SDK

RUN mkdir -p /catkin_ws/src && cd /catkin_ws/src && \
    git clone https://github.com/Livox-SDK/livox_ros_driver.git && \
    git -C livox_ros_driver checkout ${LIVOX_ROS_DRIVER_COMMIT} && \
    git clone https://github.com/hku-mars/FAST_LIO.git && \
    cd FAST_LIO && git checkout ${FAST_LIO_COMMIT} && \
    git submodule update --init --recursive

WORKDIR /catkin_ws
RUN . /opt/ros/noetic/setup.sh && \
    catkin_make -j$(nproc)

RUN sed -i "6i source \"/catkin_ws/devel/setup.bash\"" /ros_entrypoint.sh

# rviz for the live GUI, plus Xvfb + xwd + ImageMagick so run_hilti_offline.sh can
# render rviz on a virtual display and save a PNG (SCREENSHOT=1) with no host X at all.
RUN apt-get update -y && apt-get install -y \
    ros-noetic-rviz xvfb x11-apps imagemagick && \
    rm -rf /var/lib/apt/lists/*

# Course-wide RViz mouse scheme: left drag rotates, right/middle drag pans, wheel zooms.
COPY --from=localhost/slam_zero_to_hero:rviz_unified_controls /rviz_unified_controls /opt/rviz_unified_controls
RUN /opt/rviz_unified_controls/install_ros1.sh /opt/rviz_unified_controls && \
    sed -i -E 's#Class: rviz/Orbit$#Class: slam_zero_to_hero/UnifiedOrbit#' \
        /catkin_ws/src/FAST_LIO/rviz_cfg/loam_livox.rviz

WORKDIR /data

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
