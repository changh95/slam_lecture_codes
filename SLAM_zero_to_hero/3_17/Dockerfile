FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libboost-all-dev \
    libeigen3-dev \
    libtbb-dev \
    && rm -rf /var/lib/apt/lists/*

# Install GTSAM
RUN git clone --depth 1 --branch 4.2 https://github.com/borglab/gtsam.git /tmp/gtsam && \
    cd /tmp/gtsam && \
    mkdir build && cd build && \
    cmake .. -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
             -DGTSAM_BUILD_TESTS=OFF \
             -DGTSAM_BUILD_PYTHON=OFF && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/gtsam

# Install Kimera-RPGO
RUN git clone --depth 1 https://github.com/MIT-SPARK/Kimera-RPGO.git /tmp/kimera-rpgo && \
    cd /tmp/kimera-rpgo && \
    mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/kimera-rpgo

# Update library path
RUN ldconfig

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Build examples
RUN mkdir -p build && cd build && \
    cmake .. && \
    make -j$(nproc)

CMD ["/bin/bash"]
