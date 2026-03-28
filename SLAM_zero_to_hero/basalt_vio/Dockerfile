FROM ubuntu:focal

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y \
    build-essential git wget ninja-build python3-pip \
    libtbb-dev libfmt-dev libglew-dev libglfw3-dev \
    libpng-dev libjpeg-dev liblz4-dev libbz2-dev libboost-all-dev \
    libopencv-dev libeigen3-dev libgoogle-glog-dev \
    libgflags-dev libatlas-base-dev libsuitesparse-dev && \
    apt-get clean

# Install CMake >= 3.24 (required by Basalt; focal has 3.16)
RUN pip3 install cmake==3.28.0

# Clone Basalt (pinned commit)
RUN git clone https://github.com/VladyslavUsenko/basalt.git && \
    cd basalt && \
    git checkout 0f3b2b5 && \
    git submodule update --init --recursive

# Build Basalt using its bundled thirdparty system
WORKDIR /basalt
# Patch: Force FindBoost module mode
# Ubuntu's BoostConfig.cmake references per-component configs that don't exist
# Patch: Fix find_package issues for Ubuntu Focal
# - Boost: remove CONFIG mode and header-only 'format' from COMPONENTS
# - lz4: change CONFIG to MODULE mode (liblz4-dev has no cmake config)
RUN sed -i 's/find_package(Boost CONFIG REQUIRED/find_package(Boost REQUIRED/' CMakeLists.txt && \
    sed -i 's/ format//' CMakeLists.txt && \
    sed -i 's/find_package(lz4 CONFIG REQUIRED)/find_package(PkgConfig REQUIRED)\npkg_check_modules(LZ4 REQUIRED IMPORTED_TARGET liblz4)/' CMakeLists.txt && \
    sed -i 's/lz4::lz4/PkgConfig::LZ4/g' CMakeLists.txt && \
    rm -rf /usr/lib/x86_64-linux-gnu/cmake/[Bb]oost*

RUN mkdir build && cd build && \
    /usr/local/bin/cmake .. -DCMAKE_BUILD_TYPE=Release -GNinja && \
    ninja -j$(nproc)

WORKDIR /data
