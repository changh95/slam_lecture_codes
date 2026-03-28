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

# Clone ConceptFusion (pinned commit)
RUN git clone https://github.com/concept-fusion/concept-fusion.git && \
    cd concept-fusion && git checkout 4457c1f

WORKDIR /concept-fusion

# Install PyTorch and dependencies
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install -r requirements.txt || true

# Install Segment Anything
RUN pip3 install git+https://github.com/facebookresearch/segment-anything.git || true

WORKDIR /data
