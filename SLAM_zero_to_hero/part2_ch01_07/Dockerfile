FROM slam:base

WORKDIR /workspace
COPY CMakeLists.txt /workspace/part2_ch01_07/CMakeLists.txt
COPY examples /workspace/part2_ch01_07/examples

RUN cd /workspace/part2_ch01_07 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

VOLUME ["/data"]
WORKDIR /workspace/part2_ch01_07/build
