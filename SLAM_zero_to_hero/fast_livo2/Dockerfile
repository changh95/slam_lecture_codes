# Based on rosblox/ros-fast-livo community Dockerfile
FROM docker.io/ros:noetic

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y \
    build-essential cmake git wget \
    libeigen3-dev libpcl-dev libopencv-dev libusb-1.0-0-dev \
    ros-noetic-pcl-ros ros-noetic-cv-bridge ros-noetic-image-transport \
    ros-noetic-eigen-conversions && \
    apt-get clean

# Install Eigen 3.4 (Sophus needs >= 3.4, noetic ships 3.3.7)
RUN wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.1/eigen-3.4.1.zip && \
    unzip -q eigen-3.4.1.zip && cd eigen-3.4.1 && \
    mkdir build && cd build && cmake .. && make install && \
    cd / && rm -rf eigen-3.4.1 eigen-3.4.1.zip

# Install Sophus (non-templated/double-only version as required by FAST-LIVO2)
RUN git clone https://github.com/strasdat/Sophus.git && \
    cd Sophus && git checkout a621ff && \
    mkdir build && cd build && \
    cmake .. && make -j$(nproc) && make install && ldconfig && \
    cd / && rm -rf Sophus

# Install Livox SDK
RUN git clone https://github.com/Livox-SDK/Livox-SDK.git && \
    cd Livox-SDK && mkdir -p build && cd build && \
    cmake .. && make -j$(nproc) && make install && \
    cd / && rm -rf Livox-SDK

# Build in catkin workspace
RUN mkdir -p /catkin_ws/src && cd /catkin_ws/src && \
    git clone https://github.com/Livox-SDK/livox_ros_driver.git && \
    git clone https://github.com/xuankuzcr/rpg_vikit.git && \
    git clone https://github.com/hku-mars/FAST-LIVO2.git && \
    cd FAST-LIVO2 && git submodule update --init --recursive

WORKDIR /catkin_ws
RUN . /opt/ros/noetic/setup.sh && \
    catkin_make -j$(nproc)

RUN sed -i "6i source \"/catkin_ws/devel/setup.bash\"" /ros_entrypoint.sh

WORKDIR /data

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
