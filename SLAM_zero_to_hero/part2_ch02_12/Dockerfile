FROM slam:base

WORKDIR /workspace
COPY . /workspace/part2_ch02_12

# Build the RANSAC/USAC examples
RUN cd /workspace/part2_ch02_12 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

WORKDIR /workspace/part2_ch02_12/build
