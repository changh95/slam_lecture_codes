FROM slam_zero_to_hero:base

WORKDIR /workspace
COPY . /workspace/3_15

# Build
RUN cd /workspace/3_15 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

# Download sample BAL dataset
RUN cd /workspace/3_15/build && \
    wget -q https://grail.cs.washington.edu/projects/bal/data/ladybug/problem-49-7776-pre.txt.bz2 && \
    bunzip2 problem-49-7776-pre.txt.bz2 || true

WORKDIR /workspace/3_15/build

CMD ["sh", "-c", "echo '=== Ceres-Solver Tutorial ===' && \
    echo '' && \
    echo '1. Running basic examples:' && \
    ./ceres_basics && \
    echo '' && \
    echo '2. Running bundle adjustment (if BAL data available):' && \
    if [ -f problem-49-7776-pre.txt ]; then ./ceres_bundle_adjustment problem-49-7776-pre.txt; else echo 'No BAL dataset found'; fi"]
