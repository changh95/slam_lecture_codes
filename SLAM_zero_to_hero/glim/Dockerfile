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

# Build GTSAM (develop branch — glim needs std::shared_ptr support, not in 4.2.0)
RUN git clone https://github.com/borglab/gtsam.git -b develop && \
    cd gtsam && mkdir build && cd build && \
    cmake .. -DGTSAM_BUILD_TESTS=OFF \
             -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
             -DGTSAM_USE_SYSTEM_EIGEN=ON \
             -DGTSAM_WITH_TBB=OFF \
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

# Build gtsam_points
RUN git clone https://github.com/koide3/gtsam_points.git --recursive && \
    cd gtsam_points && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DBUILD_WITH_CUDA=ON && \
    make -j$(nproc) && make install && ldconfig && \
    cd / && rm -rf gtsam_points

# Build GLIM
RUN git clone https://github.com/koide3/glim.git --recursive && \
    cd glim && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && make install && ldconfig

WORKDIR /data
