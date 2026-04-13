FROM docker.io/ros:noetic

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y \
    build-essential cmake git wget python3 python3-pip \
    libeigen3-dev libpcl-dev libusb-1.0-0-dev \
    ros-noetic-pcl-ros ros-noetic-eigen-conversions && \
    apt-get clean

# Rerun Python SDK + numpy
# Ubuntu 20.04 (noetic) ships Python 3.8 + pip 20; must upgrade pip first,
# and pin rerun-sdk to 0.21.0 (last version still compatible with Py3.8).
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir numpy "rerun-sdk==0.21.0"

# Livox SDK (required by FAST-LIO2 build)
RUN git clone https://github.com/Livox-SDK/Livox-SDK.git && \
    cd Livox-SDK && mkdir -p build && cd build && \
    cmake .. && make -j$(nproc) && make install && \
    cd / && rm -rf Livox-SDK

# Build FAST-LIO2 (unmodified upstream)
RUN mkdir -p /catkin_ws/src && cd /catkin_ws/src && \
    git clone https://github.com/Livox-SDK/livox_ros_driver.git && \
    git clone https://github.com/hku-mars/FAST_LIO.git && \
    cd FAST_LIO && git submodule update --init --recursive

# Add HILTI config and launch file for Hesai PandarXT-32 + Alphasense IMU
COPY hilti.yaml /catkin_ws/src/FAST_LIO/config/hilti.yaml
COPY mapping_hilti.launch /catkin_ws/src/FAST_LIO/launch/mapping_hilti.launch

WORKDIR /catkin_ws
RUN . /opt/ros/noetic/setup.sh && \
    catkin_make -j$(nproc) -DCMAKE_BUILD_TYPE=Release

RUN sed -i "6i source \"/catkin_ws/devel/setup.bash\"" /ros_entrypoint.sh

# Rerun bridge + entrypoint
COPY ros_rerun_bridge.py /app/ros_rerun_bridge.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PYTHONUNBUFFERED=1
WORKDIR /data
EXPOSE 9090 9877
ENTRYPOINT ["/app/entrypoint.sh"]
