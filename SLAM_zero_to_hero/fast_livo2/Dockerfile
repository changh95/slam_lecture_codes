FROM docker.io/ros:noetic

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y \
    build-essential cmake git wget \
    python3-catkin-tools \
    libeigen3-dev libpcl-dev libopencv-dev \
    ros-noetic-pcl-ros ros-noetic-cv-bridge ros-noetic-image-transport \
    ros-noetic-eigen-conversions libusb-1.0-0-dev && \
    apt-get clean

# Install Sophus from source (patch cmake_minimum_required for focal cmake 3.16)
RUN git clone https://github.com/strasdat/Sophus.git && \
    cd Sophus && \
    sed -i 's/cmake_minimum_required(VERSION [0-9.]*/cmake_minimum_required(VERSION 3.16/' CMakeLists.txt && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF && \
    make install && ldconfig && \
    cd / && rm -rf Sophus

# Install livox_ros_driver and vikit_common (required by FAST-LIVO2)
RUN mkdir -p /catkin_ws/src && cd /catkin_ws/src && \
    git clone https://github.com/Livox-SDK/livox_ros_driver.git && \
    git clone https://github.com/uzh-rpg/rpg_vikit.git

# Clone FAST-LIVO2 (pinned commit)
RUN cd /catkin_ws/src && \
    git clone https://github.com/hku-mars/FAST-LIVO2.git && \
    cd FAST-LIVO2 && git checkout 0d2c034 && \
    git submodule update --init --recursive

WORKDIR /catkin_ws
RUN . /opt/ros/noetic/setup.sh && \
    catkin_make -j$(nproc)

RUN sed -i "6i source \"/catkin_ws/devel/setup.bash\"" /ros_entrypoint.sh

WORKDIR /data

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
