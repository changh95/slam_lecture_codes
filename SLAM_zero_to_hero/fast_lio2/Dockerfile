# Based on community fast-lio-slam-docker
FROM docker.io/ros:noetic

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y \
    build-essential cmake git wget \
    libeigen3-dev libpcl-dev libusb-1.0-0-dev \
    ros-noetic-pcl-ros ros-noetic-eigen-conversions && \
    apt-get clean

# Install Livox SDK
RUN git clone https://github.com/Livox-SDK/Livox-SDK.git && \
    cd Livox-SDK && mkdir -p build && cd build && \
    cmake .. && make -j$(nproc) && make install && \
    cd / && rm -rf Livox-SDK

# Build in catkin workspace
RUN mkdir -p /catkin_ws/src && cd /catkin_ws/src && \
    git clone https://github.com/Livox-SDK/livox_ros_driver.git && \
    git clone https://github.com/hku-mars/FAST_LIO.git && \
    cd FAST_LIO && git submodule update --init --recursive

WORKDIR /catkin_ws
RUN . /opt/ros/noetic/setup.sh && \
    catkin_make -j$(nproc)

RUN sed -i "6i source \"/catkin_ws/devel/setup.bash\"" /ros_entrypoint.sh

# rviz for the live GUI. mapping_hilti.launch has an `rviz:=true` path pointing at
# FAST_LIO's own rviz_cfg/loam_livox.rviz, but rviz itself was never installed, so
# that path died with "cannot launch node of type [rviz/rviz]". Placed after
# catkin_make so enabling the GUI costs one cheap layer rather than re-running the
# Livox-SDK and workspace builds.
RUN apt-get update -y && apt-get install -y ros-noetic-rviz && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /data

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
