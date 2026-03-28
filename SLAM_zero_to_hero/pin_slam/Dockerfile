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

# Clone PIN-SLAM (v1.1.1)
RUN git clone https://github.com/PRBonn/PIN_SLAM.git -b v1.1.1
WORKDIR /PIN_SLAM

# Install dependencies
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install -r requirements.txt

WORKDIR /data
