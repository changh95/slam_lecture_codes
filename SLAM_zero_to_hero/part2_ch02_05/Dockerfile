FROM slam:base

WORKDIR /workspace
COPY . /workspace/part2_ch02_05

# Build the project
RUN cd /workspace/part2_ch02_05 && \
    mkdir -p build && cd build && \
    cmake .. && \
    make -j4

# Create output directory for results
RUN mkdir -p /output

WORKDIR /workspace/part2_ch02_05/build
