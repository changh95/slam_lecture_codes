FROM slam:base

WORKDIR /workspace
COPY . /workspace/part1_ch02_04

RUN cd /workspace/part1_ch02_04 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

# Create output directory for profile files
RUN mkdir -p /output

WORKDIR /workspace/part1_ch02_04/build
