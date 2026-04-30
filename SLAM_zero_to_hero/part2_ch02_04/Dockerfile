FROM slam:base

WORKDIR /workspace
COPY . /workspace/part2_ch02_04

# Build the project
RUN cd /workspace/part2_ch02_04 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

# Create output directory for visualization results
RUN mkdir -p /output

WORKDIR /workspace/part2_ch02_04/build
