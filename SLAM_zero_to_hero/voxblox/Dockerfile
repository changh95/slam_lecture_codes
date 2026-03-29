FROM docker.io/ros:noetic

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y \
    build-essential cmake git \
    python3-wstool python3-catkin-tools \
    protobuf-compiler libprotobuf-dev \
    autoconf libceres-dev libeigen3-dev \
    ros-noetic-cmake-modules && \
    apt-get clean

# Use wstool to fetch voxblox and ALL its dependencies
RUN mkdir -p /catkin_ws/src && cd /catkin_ws && \
    catkin init && \
    catkin config --extend /opt/ros/noetic --cmake-args -DCMAKE_BUILD_TYPE=Release && \
    cd src && \
    git clone https://github.com/ethz-asl/voxblox.git --recursive && \
    wstool init . && \
    wstool merge -t . voxblox/voxblox_https.rosinstall && \
    wstool update -t .

WORKDIR /catkin_ws
RUN catkin build voxblox

RUN sed -i "6i source \"/catkin_ws/devel/setup.bash\"" /ros_entrypoint.sh

WORKDIR /data

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
