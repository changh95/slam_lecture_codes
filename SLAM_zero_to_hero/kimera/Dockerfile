# Based on MIT-SPARK/Kimera-VIO/Dockerfile_20_04
FROM docker.io/ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils build-essential cmake git wget unzip \
    libboost-all-dev libgoogle-glog-dev libgflags-dev \
    libatlas-base-dev libsuitesparse-dev libtbb-dev \
    libopencv-dev libgtest-dev ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Build GTSAM 4.2
RUN git clone https://github.com/borglab/gtsam.git -b 4.2 && \
    cd gtsam && mkdir build && cd build && \
    cmake .. -DGTSAM_BUILD_TESTS=OFF \
             -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
             -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && make install && ldconfig && \
    cd / && rm -rf gtsam

# Build OpenGV (bypass custom FindEigen — hardcode Eigen3 path)
RUN git clone https://github.com/laurentkneip/opengv.git && \
    cd opengv && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DEIGEN_INCLUDE_DIR=/usr/include/eigen3 \
             -DEIGEN_INCLUDE_DIRS=/usr/include/eigen3 \
             -DEIGEN_VERSION_OK=TRUE \
             -DEIGEN_VERSION=3.3.7 \
             -DCMAKE_MODULE_PATH="" && \
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

# Build Kimera-VIO
RUN git clone https://github.com/MIT-SPARK/Kimera-VIO.git --recursive && \
    cd Kimera-VIO && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DKIMERA_BUILD_TESTS=OFF && \
    make -j$(nproc) && make install && ldconfig

WORKDIR /data
