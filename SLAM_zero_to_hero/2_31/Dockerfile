FROM slam_zero_to_hero:base

WORKDIR /workspace

# Install dependencies
RUN apt-get update && apt-get install -y \
    libpcl-dev \
    liboctomap-dev \
    octomap-tools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Clone Bonxai (header-only library)
RUN git clone https://github.com/facontidavide/Bonxai.git /tmp/bonxai && \
    mkdir -p /workspace/2_31/3rdparty/bonxai && \
    cp -r /tmp/bonxai/bonxai_core/include/bonxai/* /workspace/2_31/3rdparty/bonxai/ && \
    rm -rf /tmp/bonxai

# Copy project files
COPY . /workspace/2_31

# Build the tutorial examples
RUN cd /workspace/2_31 && \
    mkdir -p build && cd build && \
    cmake -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        .. && \
    ninja

WORKDIR /workspace/2_31/build

# Default command: show available executables and run demos
CMD ["sh", "-c", "echo '=== Spatial Data Structures Tutorial ===' && \
    echo '' && \
    echo 'Available executables:' && \
    echo '  ./octree_demo         - PCL Octree spatial search' && \
    echo '  ./octomap_demo        - OctoMap probabilistic mapping' && \
    echo '  ./octomap_navigation  - Path planning with OctoMap' && \
    echo '  ./bonxai_demo         - Bonxai high-performance voxels' && \
    echo '  ./comparison          - Compare all three methods' && \
    echo '' && \
    echo '--- Running Octree Demo ---' && \
    ./octree_demo && \
    echo '' && \
    echo '--- Running OctoMap Demo ---' && \
    ./octomap_demo && \
    echo '' && \
    echo '--- Running Comparison ---' && \
    ./comparison"]
