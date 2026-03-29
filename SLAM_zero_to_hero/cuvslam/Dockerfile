FROM docker.io/nvidia/cuda:12.2.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y \
    build-essential cmake git git-lfs wget \
    python3 python3-pip python3-dev \
    libopencv-dev libeigen3-dev && \
    apt-get clean

# Clone and build cuVSLAM
RUN git lfs install && \
    git clone https://github.com/nvidia-isaac/cuVSLAM.git && \
    cd cuVSLAM && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && make install && ldconfig

WORKDIR /data
