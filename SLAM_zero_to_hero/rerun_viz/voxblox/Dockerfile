FROM docker.io/ros:noetic
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y \
    build-essential cmake git wget python3 python3-pip \
    python3-wstool python3-catkin-tools \
    protobuf-compiler libprotobuf-dev \
    autoconf libceres-dev libeigen3-dev \
    libgoogle-glog-dev libgflags-dev \
    ros-noetic-cmake-modules \
    ros-noetic-eigen-conversions ros-noetic-tf-conversions \
    ros-noetic-pcl-ros ros-noetic-pcl-conversions \
    ros-noetic-cv-bridge ros-noetic-image-transport \
    ros-noetic-tf ros-noetic-tf2 ros-noetic-tf2-ros \
    ros-noetic-interactive-markers ros-noetic-visualization-msgs \
    ros-noetic-geometry-msgs ros-noetic-nav-msgs ros-noetic-std-srvs \
    ros-noetic-message-runtime ros-noetic-message-generation \
    ros-noetic-genmsg ros-noetic-rosbag && \
    apt-get clean

# Rerun Python SDK + numpy
# Ubuntu 20.04 (noetic) ships Python 3.8; pin rerun-sdk to 0.21.0 (last Py3.8-compatible).
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir numpy "rerun-sdk==0.21.0"

# Clone voxblox and its catkin dependencies (unmodified upstream)
RUN mkdir -p /catkin_ws/src && cd /catkin_ws/src && \
    git clone https://github.com/catkin/catkin_simple.git && \
    git clone https://github.com/ethz-asl/eigen_catkin.git && \
    git clone https://github.com/ethz-asl/eigen_checks.git && \
    git clone https://github.com/ethz-asl/gflags_catkin.git && \
    git clone https://github.com/ethz-asl/glog_catkin.git && \
    git clone https://github.com/ethz-asl/protobuf_catkin.git && \
    git clone https://github.com/ethz-asl/minkindr.git && \
    git clone https://github.com/ethz-asl/minkindr_ros.git && \
    git clone https://github.com/ethz-asl/voxblox.git --recursive

WORKDIR /catkin_ws
# Remove voxblox_rviz_plugin (needs ros-noetic-rviz which we don't install)
# and strip the dep from voxblox_ros
RUN rm -rf /catkin_ws/src/voxblox/voxblox_rviz_plugin && \
    sed -i '/voxblox_rviz_plugin/d' /catkin_ws/src/voxblox/voxblox_ros/package.xml && \
    sed -i '/voxblox_rviz_plugin/d' /catkin_ws/src/voxblox/voxblox_ros/CMakeLists.txt
RUN catkin init && \
    catkin config --extend /opt/ros/noetic --cmake-args -DCMAKE_BUILD_TYPE=Release && \
    catkin build voxblox_ros

RUN sed -i "6i source \"/catkin_ws/devel/setup.bash\"" /ros_entrypoint.sh

# Rerun bridge + entrypoint
COPY ros_rerun_bridge.py /app/ros_rerun_bridge.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PYTHONUNBUFFERED=1
WORKDIR /data
EXPOSE 9090 9877
ENTRYPOINT ["/app/entrypoint.sh"]
