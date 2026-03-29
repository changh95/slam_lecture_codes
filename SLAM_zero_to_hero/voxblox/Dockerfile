FROM docker.io/ros:noetic

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y \
    build-essential cmake git \
    python3-wstool python3-catkin-tools \
    protobuf-compiler libprotobuf-dev \
    autoconf libceres-dev libeigen3-dev \
    libgoogle-glog-dev libgflags-dev \
    ros-noetic-cmake-modules && \
    apt-get clean

# Clone voxblox and minimal deps (skip grpc/catkin_grpc)
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
RUN catkin init && \
    catkin config --extend /opt/ros/noetic --cmake-args -DCMAKE_BUILD_TYPE=Release && \
    catkin build voxblox

RUN sed -i "6i source \"/catkin_ws/devel/setup.bash\"" /ros_entrypoint.sh

WORKDIR /data

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
