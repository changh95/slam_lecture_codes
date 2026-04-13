# Kimera-VIO-ROS + Rerun visualization
# Base: ROS Noetic on Ubuntu 20.04 (Python 3.8)
FROM docker.io/ros:noetic
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y \
    build-essential cmake git wget python3 python3-pip \
    libboost-all-dev libgoogle-glog-dev libgflags-dev \
    libatlas-base-dev libsuitesparse-dev libtbb-dev \
    libopencv-dev ros-noetic-cv-bridge \
    ros-noetic-tf2 ros-noetic-tf2-ros \
    ros-noetic-tf2-geometry-msgs \
    ros-noetic-pcl-ros ros-noetic-eigen-conversions \
    ros-noetic-image-transport \
    ros-noetic-visualization-msgs && \
    apt-get clean

# Rerun Python SDK + numpy
# Ubuntu 20.04 (noetic) ships Python 3.8; pin rerun-sdk to 0.21.0
# (last version compatible with Python 3.8).
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir numpy "rerun-sdk==0.21.0"

# Build GTSAM 4.2
RUN git clone https://github.com/borglab/gtsam.git -b 4.2 && \
    cd gtsam && mkdir build && cd build && \
    cmake .. -DGTSAM_BUILD_TESTS=OFF \
             -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
             -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && make install && ldconfig && \
    cd / && rm -rf gtsam

# Install Eigen 3.4.1 from source
RUN wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.1/eigen-3.4.1.zip && \
    unzip -q eigen-3.4.1.zip && cd eigen-3.4.1 && \
    mkdir build && cd build && cmake .. && make install && \
    cd / && rm -rf eigen-3.4.1 eigen-3.4.1.zip

# Build OpenGV
RUN git clone https://github.com/laurentkneip/opengv.git && \
    cd opengv && \
    rm -f modules/FindEigen.cmake && \
    sed -i 's/find_package(Eigen REQUIRED)/find_package(Eigen3 REQUIRED)/' CMakeLists.txt && \
    sed -i 's/\${EIGEN_INCLUDE_DIRS}/\${EIGEN3_INCLUDE_DIR}/g' CMakeLists.txt && \
    sed -i 's/\${EIGEN_INCLUDE_DIR}/\${EIGEN3_INCLUDE_DIR}/g' CMakeLists.txt && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && make install && ldconfig && \
    cd / && rm -rf opengv

# Build DBoW2
RUN git clone https://github.com/dorian3d/DBoW2.git && \
    cd DBoW2 && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && make install && ldconfig && \
    cd / && rm -rf DBoW2

# Build Kimera-RPGO
RUN git clone https://github.com/MIT-SPARK/Kimera-RPGO.git && \
    cd Kimera-RPGO && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && make install && ldconfig && \
    cd / && rm -rf Kimera-RPGO

# Build Kimera-VIO (standalone library, needed by Kimera-VIO-ROS)
RUN git clone https://github.com/MIT-SPARK/Kimera-VIO.git --recursive && \
    cd Kimera-VIO && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DKIMERA_BUILD_TESTS=OFF && \
    make -j$(nproc) && make install && ldconfig && \
    cd / && rm -rf Kimera-VIO

# Build Kimera-VIO-ROS in a catkin workspace (needs catkin_simple, pose_graph_tools)
RUN mkdir -p /catkin_ws/src && cd /catkin_ws/src && \
    git clone https://github.com/catkin/catkin_simple.git && \
    git clone https://github.com/MIT-SPARK/pose_graph_tools.git -b archive/ros_noetic && \
    git clone https://github.com/MIT-SPARK/Kimera-VIO-ROS.git

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-catkin-tools && rm -rf /var/lib/apt/lists/*

WORKDIR /catkin_ws
RUN . /opt/ros/noetic/setup.sh && \
    catkin init && \
    catkin config --extend /opt/ros/noetic --cmake-args -DCMAKE_BUILD_TYPE=Release && \
    catkin build kimera_vio_ros

RUN sed -i "6i source \"/catkin_ws/devel/setup.bash\"" /ros_entrypoint.sh

# Rerun bridge + entrypoint
COPY ros_rerun_bridge.py /app/ros_rerun_bridge.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PYTHONUNBUFFERED=1
VOLUME /data
WORKDIR /data
EXPOSE 9090 9877
ENTRYPOINT ["/app/entrypoint.sh"]
