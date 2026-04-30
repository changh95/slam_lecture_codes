FROM slam:base

WORKDIR /workspace
COPY . /workspace/part2_ch01_10

RUN cd /workspace/part2_ch01_10 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

WORKDIR /workspace/part2_ch01_10/build
