FROM slam:base

WORKDIR /workspace
COPY . /workspace/part2_ch03_04

# Build
RUN cd /workspace/part2_ch03_04 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

# The exercise root, not build/, so the demos find data/000000.bin
WORKDIR /workspace/part2_ch03_04
