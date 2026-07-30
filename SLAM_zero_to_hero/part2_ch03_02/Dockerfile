FROM slam:base

WORKDIR /workspace
COPY . /workspace/part2_ch03_02

# Build
RUN cd /workspace/part2_ch03_02 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

# The exercise root, not build/: visualization opens ./000000.bin relative to CWD.
WORKDIR /workspace/part2_ch03_02
