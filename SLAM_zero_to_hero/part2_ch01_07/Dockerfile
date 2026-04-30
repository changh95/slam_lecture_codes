FROM slam:base

WORKDIR /workspace
COPY . /workspace/part2_ch01_07

RUN cd /workspace/part2_ch01_07 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

WORKDIR /workspace/part2_ch01_07/build
