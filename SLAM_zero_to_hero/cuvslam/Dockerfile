FROM docker.io/nvidia/cuda:12.6.3-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}compute,utility,graphics

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential cmake git git-lfs wget ca-certificates \
    python3 python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Install PyCuVSLAM from pre-built wheel + dependencies
# Ubuntu 22.04 (Python 3.10) + CUDA 12 + auto-detected architecture (x86_64 or aarch64)
RUN pip3 install --no-cache-dir numpy pillow rerun-sdk && \
    ARCH=$(uname -m) && \
    pip3 install --no-cache-dir \
    "https://github.com/nvidia-isaac/cuVSLAM/releases/download/v15.0.0/cuvslam-15.0.0%2Bcu12-cp310-cp310-manylinux_2_35_${ARCH}.whl"

# Clone the cuVSLAM repo for example scripts (examples/kitti, etc.)
RUN git clone --depth 1 https://github.com/nvidia-isaac/cuVSLAM.git /cuVSLAM

WORKDIR /data
