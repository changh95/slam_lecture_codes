FROM docker.io/ros:noetic

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y \
    build-essential cmake git wget \
    libeigen3-dev libpcl-dev libopencv-dev libusb-1.0-0-dev \
    ros-noetic-pcl-ros ros-noetic-cv-bridge ros-noetic-image-transport \
    ros-noetic-eigen-conversions && \
    apt-get clean

# Install patched Sophus 1.22.10 (with non-templated aliases + .h symlinks)
# Uses system Eigen 3.3.7 from noetic
COPY Sophus /Sophus
RUN cd /Sophus && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF \
             -DCMAKE_CXX_FLAGS="-Wno-error" && \
    make -j$(nproc) install && ldconfig

# Install Livox SDK
RUN git clone https://github.com/Livox-SDK/Livox-SDK.git && \
    cd Livox-SDK && mkdir -p build && cd build && \
    cmake .. && make -j$(nproc) && make install && \
    cd / && rm -rf Livox-SDK

# Copy patched sources into catkin workspace
RUN mkdir -p /catkin_ws/src
COPY rpg_vikit /catkin_ws/src/rpg_vikit
COPY FAST-LIVO2 /catkin_ws/src/FAST-LIVO2
RUN cd /catkin_ws/src && \
    git clone https://github.com/Livox-SDK/livox_ros_driver.git

WORKDIR /catkin_ws
RUN . /opt/ros/noetic/setup.sh && \
    catkin_make -j$(nproc)

RUN sed -i "6i source \"/catkin_ws/devel/setup.bash\"" /ros_entrypoint.sh

WORKDIR /data

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
