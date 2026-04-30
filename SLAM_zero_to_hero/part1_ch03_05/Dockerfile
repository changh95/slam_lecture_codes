FROM slam:base

WORKDIR /workspace
COPY . /workspace/part1_ch03_05

# Build
RUN cd /workspace/part1_ch03_05 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

WORKDIR /workspace/part1_ch03_05/build
