FROM docker.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y \
    build-essential cmake git wget \
    python3 python3-pip python3-dev \
    libgl1-mesa-dev libglib2.0-0 libglew-dev libglfw3-dev && \
    apt-get clean

# Clone MonoGS (CVPR 2024 Best Demo, pinned commit)
RUN git clone https://github.com/muskie82/MonoGS.git --recursive && \
    cd MonoGS && git checkout 6c9254c

WORKDIR /MonoGS

# Install PyTorch and dependencies
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install -r requirements.txt || true

# Build custom CUDA extensions
RUN pip3 install submodules/diff-gaussian-rasterization && \
    pip3 install submodules/simple-knn || true

WORKDIR /data
