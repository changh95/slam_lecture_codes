# SuperPoint + SuperGlue with TensorRT
# Uses NVIDIA TensorRT base image for GPU inference optimization

# Use TensorRT base image (CUDA 11.7, TensorRT 8.4.1)
FROM nvcr.io/nvidia/tensorrt:22.07-py3

LABEL maintainer="changh95"
LABEL description="SuperPoint and SuperGlue feature matching with TensorRT acceleration"

ARG DEBIAN_FRONTEND=noninteractive

# ==============================================================================
# Install System Dependencies
# ==============================================================================

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    pkg-config \
    # OpenCV dependencies
    libgtk2.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    # Eigen and YAML
    libeigen3-dev \
    libyaml-cpp-dev \
    # Python for ONNX conversion
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# Install OpenCV
# ==============================================================================

RUN wget -q https://github.com/opencv/opencv/archive/4.4.0.zip -O opencv.zip && \
    unzip -q opencv.zip && \
    cd opencv-4.4.0 && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_opencv_apps=OFF && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd ../.. && \
    rm -rf opencv-4.4.0 opencv.zip

# ==============================================================================
# Install Python Dependencies (for ONNX conversion)
# ==============================================================================

RUN pip3 install --no-cache-dir \
    torch \
    torchvision \
    onnx \
    onnxruntime \
    numpy \
    opencv-python-headless

# ==============================================================================
# Copy and Build Project
# ==============================================================================

WORKDIR /workspace

# Copy project files
COPY . /workspace/superpointglue

# Build the project
RUN cd /workspace/superpointglue && \
    mkdir -p build && cd build && \
    cmake .. && \
    make -j$(nproc)

# ==============================================================================
# Set Default Working Directory
# ==============================================================================

WORKDIR /workspace/superpointglue
