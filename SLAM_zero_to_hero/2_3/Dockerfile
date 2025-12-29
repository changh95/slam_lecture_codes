FROM slam_zero_to_hero:base

WORKDIR /workspace
COPY . /workspace/2_3

# Build the examples
RUN cd /workspace/2_3 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

WORKDIR /workspace/2_3/build

# Default command: run feature detection demo
CMD ["./feature_detection"]
