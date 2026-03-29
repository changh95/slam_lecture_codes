FROM ubuntu:focal

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y \
    build-essential git wget curl zip unzip tar ninja-build \
    pkg-config python3-pip autoconf automake libtool \
    libgl1-mesa-dev libglu1-mesa-dev libglew-dev libglfw3-dev \
    libtbb-dev libusb-1.0-0-dev libusb-dev && \
    apt-get clean

# Install CMake >= 3.24 (required by Basalt)
RUN pip3 install cmake==3.28.0

# Clone Basalt with vcpkg submodule
RUN git clone https://github.com/VladyslavUsenko/basalt.git && \
    cd basalt && \
    git checkout 0f3b2b5 && \
    git submodule update --init --recursive

# Bootstrap vcpkg
WORKDIR /basalt
RUN thirdparty/vcpkg/bootstrap-vcpkg.sh

# Build using vcpkg toolchain (handles all dependencies automatically)
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++
ENV VCPKG_FORCE_SYSTEM_BINARIES=1

RUN mkdir build && cd build && \
    /usr/local/bin/cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -GNinja \
    -DCMAKE_TOOLCHAIN_FILE=/basalt/thirdparty/vcpkg/scripts/buildsystems/vcpkg.cmake && \
    ninja -j$(nproc)

WORKDIR /data
