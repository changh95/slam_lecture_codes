FROM docker.io/ros:noetic

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget ca-certificates ninja-build gettext-base \
    python3 python3-pip \
    libeigen3-dev libpcl-dev libopencv-dev \
    libboost-all-dev libgoogle-glog-dev libgflags-dev \
    libatlas-base-dev libsuitesparse-dev libtbb-dev \
    libspdlog-dev libfmt-dev \
    libglfw3-dev libglm-dev libpng-dev \
    libgtest-dev nlohmann-json3-dev \
    ros-noetic-cv-bridge ros-noetic-tf2-ros ros-noetic-rosbag \
    ros-noetic-pcl-ros ros-noetic-image-transport && \
    rm -rf /var/lib/apt/lists/*

# Newer CMake (gtsam_points requires >= 3.22; noetic ships 3.16)
RUN wget -qO- https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-$(uname -m).tar.gz | \
    tar xz -C /usr/local --strip-components=1

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir numpy "rerun-sdk==0.21.0"

# GTSAM 4.2.0
RUN git clone https://github.com/borglab/gtsam.git -b 4.2.0 && \
    cd gtsam && mkdir build && cd build && \
    cmake .. -DGTSAM_BUILD_TESTS=OFF \
             -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
             -DGTSAM_USE_SYSTEM_EIGEN=ON \
             -DCMAKE_BUILD_TYPE=Release \
             -DCMAKE_CXX_FLAGS="-Wno-error=maybe-uninitialized -Wno-error=array-bounds" && \
    make -j$(nproc) && make install && ldconfig && \
    cd / && rm -rf gtsam

# iridescence
RUN git clone https://github.com/koide3/iridescence.git --recursive && \
    cd iridescence && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && make install && ldconfig && \
    cd / && rm -rf iridescence

# gtsam_points v1.0.0
RUN git clone https://github.com/koide3/gtsam_points.git -b v1.0.0 --recursive && \
    cd gtsam_points && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DBUILD_WITH_CUDA=OFF \
             -DCMAKE_CXX_FLAGS="-Wno-error=maybe-uninitialized -Wno-error=array-bounds" && \
    make -j$(nproc) && make install && ldconfig && \
    cd / && rm -rf gtsam_points

# GLIM v1.0.0 (unmodified upstream)
RUN git clone https://github.com/koide3/glim.git -b v1.0.0 --recursive && \
    cd glim && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_CUDA=OFF && \
    make -j$(nproc) && make install && ldconfig

# glim_ros1 (provides glim_rosbag executable)
RUN git clone https://github.com/koide3/glim_ros1.git && \
    cd glim_ros1 && mkdir build && cd build && \
    . /opt/ros/noetic/setup.sh && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && make install && ldconfig

COPY ros_rerun_bridge.py /app/ros_rerun_bridge.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PYTHONUNBUFFERED=1
WORKDIR /data
EXPOSE 9090 9877
ENTRYPOINT ["/app/entrypoint.sh"]
