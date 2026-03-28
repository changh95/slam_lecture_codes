FROM docker.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y \
    build-essential cmake git wget \
    python3 python3-pip python3-dev \
    libgl1-mesa-dev libglib2.0-0 && \
    apt-get clean

# Clone MASt3R-SLAM (pinned commit)
RUN git clone https://github.com/rmurai0610/MASt3R-SLAM.git --recursive && \
    cd MASt3R-SLAM && git checkout e6f4e3d

WORKDIR /MASt3R-SLAM

# Install PyTorch and dependencies
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install -r requirements.txt || true

WORKDIR /data
