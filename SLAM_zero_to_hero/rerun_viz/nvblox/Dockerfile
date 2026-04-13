ARG BASE_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu22.04
FROM ${BASE_IMAGE}
ARG DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}compute,utility,graphics

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential cmake git git-lfs wget curl ca-certificates \
    libeigen3-dev libgoogle-glog-dev libgflags-dev \
    libsqlite3-dev libgtest-dev libepoxy-dev \
    python3 python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies: rerun-sdk (Ubuntu 22.04 = Python 3.10+)
RUN pip3 install --no-cache-dir numpy rerun-sdk open3d

# Build nvblox from source
# BUILD_PYTORCH_WRAPPER=OFF skips nvblox_torch which requires libtorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libbenchmark-dev libstdgpu-dev libpcl-dev libsqlite3-dev \
    libgtest-dev libepoxy-dev libeigen3-dev libgoogle-glog-dev libgflags-dev && \
    rm -rf /var/lib/apt/lists/* || true
RUN git clone https://github.com/nvidia-isaac/nvblox.git /nvblox_src --recursive --depth 1 && \
    cd /nvblox_src && mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90" \
        -DBUILD_TESTING=OFF \
        -DBUILD_PYTORCH_WRAPPER=OFF && \
    make -j$(nproc) && make install && ldconfig

# Install nvblox Python bindings if available
RUN pip3 install --no-cache-dir nvblox || echo "nvblox pip wheel not available"

# Copy visualization runners
COPY nvblox_rerun.py /app/nvblox_rerun.py
COPY tsdf_live.py /app/tsdf_live.py

EXPOSE 9090 9876

VOLUME /data
WORKDIR /app

CMD ["python3", "nvblox_rerun.py", "--help"]
