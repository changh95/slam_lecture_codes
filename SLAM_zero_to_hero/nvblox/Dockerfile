FROM docker.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y \
    build-essential cmake git wget \
    libeigen3-dev libgoogle-glog-dev libgflags-dev \
    libsqlite3-dev libgtest-dev \
    python3 python3-pip python3-dev && \
    apt-get clean

# Build nvblox standalone (without ROS)
# Install PyTorch (needed by nvblox_torch)
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

RUN git clone https://github.com/nvidia-isaac/nvblox.git --recursive && \
    cd nvblox && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90" \
             -DTorch_DIR=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")/Torch && \
    make -j$(nproc) && make install && ldconfig

WORKDIR /data
