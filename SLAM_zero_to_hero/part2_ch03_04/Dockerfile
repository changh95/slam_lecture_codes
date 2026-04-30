FROM slam:base

WORKDIR /workspace
COPY . /workspace/part2_ch03_04

# Build
RUN cd /workspace/part2_ch03_04 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

WORKDIR /workspace/part2_ch03_04/build
