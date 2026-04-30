FROM slam:base

WORKDIR /workspace
COPY . /workspace/part2_ch02_02

# Build the examples
RUN cd /workspace/part2_ch02_02 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

WORKDIR /workspace/part2_ch02_02/build
