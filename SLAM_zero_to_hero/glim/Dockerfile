FROM docker.io/nvidia/cuda:12.2.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y \
    build-essential cmake git wget ninja-build \
    libeigen3-dev libpcl-dev libopencv-dev \
    libboost-all-dev libgoogle-glog-dev libgflags-dev \
    libatlas-base-dev libsuitesparse-dev libtbb-dev \
    libglew-dev libglfw3-dev libpng-dev \
    libspdlog-dev libfmt-dev && \
    apt-get clean

# Install GTSAM
RUN git clone https://github.com/borglab/gtsam.git -b 4.3a1 && \
    cd gtsam && mkdir build && cd build && \
    cmake .. -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF && \
    make -j$(nproc) && make install && ldconfig && \
    cd / && rm -rf gtsam

# Install Ceres
RUN git clone https://github.com/ceres-solver/ceres-solver.git -b 2.2.0 && \
    cd ceres-solver && mkdir build && cd build && \
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
    make -j$(nproc) && make install && ldconfig && \
    cd / && rm -rf ceres-solver

# Install iridescence (visualization)
RUN git clone https://github.com/koide3/iridescence.git --recursive && \
    cd iridescence && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && make install && ldconfig && \
    cd / && rm -rf iridescence

# Install gtsam_points (required by GLIM)
RUN git clone https://github.com/koide3/gtsam_points.git --recursive && \
    cd gtsam_points && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && make install && ldconfig && \
    cd / && rm -rf gtsam_points

# Clone and build GLIM (v1.2.0)
RUN git clone https://github.com/koide3/glim.git -b v1.2.0 --recursive && \
    cd glim && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && make install && ldconfig

WORKDIR /data
