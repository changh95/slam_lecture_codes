FROM docker.io/ros:noetic

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y \
    build-essential cmake git wget \
    python3-catkin-tools \
    libeigen3-dev libpcl-dev libusb-1.0-0-dev \
    ros-noetic-pcl-ros ros-noetic-eigen-conversions && \
    apt-get clean

# Install livox_ros_driver (required by FAST-LIO2)
RUN mkdir -p /catkin_ws/src && cd /catkin_ws/src && \
    git clone https://github.com/Livox-SDK/livox_ros_driver.git

# Clone FAST-LIO2 (pinned commit)
RUN cd /catkin_ws/src && \
    git clone https://github.com/hku-mars/FAST_LIO.git && \
    cd FAST_LIO && git checkout 7cc4175 && \
    git submodule update --init --recursive

WORKDIR /catkin_ws
RUN . /opt/ros/noetic/setup.sh && \
    catkin_make -j$(nproc)

RUN sed -i "6i source \"/catkin_ws/devel/setup.bash\"" /ros_entrypoint.sh

WORKDIR /data

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
