FROM docker.io/nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y \
    build-essential cmake git wget \
    libboost-all-dev libeigen3-dev libglew-dev \
    libpcl-dev libgtest-dev && \
    apt-get clean

# Clone SuMa++ (pinned commit)
RUN git clone https://github.com/PRBonn/semantic_suma.git --recursive && \
    cd semantic_suma && git checkout 531954d

WORKDIR /semantic_suma

# Build
RUN mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) || true

WORKDIR /data
