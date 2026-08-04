# Based on koide3/glim/docker/ubuntu/Dockerfile.gcc
FROM docker.io/nvidia/cuda:13.1.0-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget ca-certificates ninja-build \
    libeigen3-dev libpcl-dev libopencv-dev \
    libboost-all-dev libgoogle-glog-dev libgflags-dev \
    libatlas-base-dev libsuitesparse-dev libtbb-dev \
    libspdlog-dev libfmt-dev \
    libglfw3-dev libglm-dev libpng-dev \
    libgtest-dev nlohmann-json3-dev && \
    rm -rf /var/lib/apt/lists/*

# Build GTSAM 4.2.0 (matches gtsam_points/glim v1.0.0)
RUN git clone https://github.com/borglab/gtsam.git -b 4.2.0 && \
    cd gtsam && mkdir build && cd build && \
    cmake .. -DGTSAM_BUILD_TESTS=OFF \
             -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
             -DGTSAM_USE_SYSTEM_EIGEN=ON \
             -DCMAKE_BUILD_TYPE=Release \
             -DCMAKE_CXX_FLAGS="-Wno-error=maybe-uninitialized -Wno-error=array-bounds" && \
    make -j$(nproc) && make install && ldconfig && \
    cd / && rm -rf gtsam

# Build iridescence (visualization)
RUN git clone https://github.com/koide3/iridescence.git --recursive && \
    cd iridescence && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && make install && ldconfig && \
    cd / && rm -rf iridescence

# Build gtsam_points (pin to v1.0.0 for GTSAM 4.2.0 compat)
RUN git clone https://github.com/koide3/gtsam_points.git -b v1.0.0 --recursive && \
    cd gtsam_points && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DBUILD_WITH_CUDA=OFF \
             -DCMAKE_CXX_FLAGS="-Wno-error=maybe-uninitialized -Wno-error=array-bounds" && \
    make -j$(nproc) && make install && ldconfig && \
    cd / && rm -rf gtsam_points

# Build GLIM v1.0.0
RUN git clone https://github.com/koide3/glim.git -b v1.0.0 --recursive && \
    cd glim && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && make install && ldconfig

# A plain (non-ROS) glim build installs NO executables — it has no add_executable
# and no main() anywhere. Upstream's data-ingesting binaries (glim_rosnode,
# glim_rosbag, offline_viewer) live in the separate koide3/glim_ros1 package,
# which needs ROS 1 Noetic and therefore Ubuntu 20.04 — impossible on this
# 24.04 CUDA base. So without the driver below, this image cannot read a dataset
# at all. glim_kitti feeds KITTI velodyne .bin scans straight through the real
# GLIM pipeline (TimeKeeper -> CloudPreprocessor -> Async{Odometry,SubMapping,
# GlobalMapping}), mirroring GlimROS::insert_frame().
COPY kitti /opt/glim_kitti/src
RUN cmake -S /opt/glim_kitti/src -B /opt/glim_kitti/build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /opt/glim_kitti/build -j$(nproc) --target install && \
    rm -rf /opt/glim_kitti/build

# glim's non-ROS build does not install its config JSONs (the install(DIRECTORY
# config ...) rule is inside a ROS-only branch), so the only copy is in the
# source clone at /glim/config. Ship the KITTI-tuned variant on a stable path.
COPY config_kitti /usr/local/share/glim_kitti/config
COPY config_kitti_posegraph /usr/local/share/glim_kitti/config_posegraph

WORKDIR /data
