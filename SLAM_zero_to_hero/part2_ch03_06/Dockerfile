FROM slam:base

WORKDIR /workspace

# Copy project files
COPY . /workspace/part2_ch03_06

# Build the project
RUN cd /workspace/part2_ch03_06 && \
    mkdir -p build && cd build && \
    cmake .. && \
    make -j4

# Create output directory for results
RUN mkdir -p /output

WORKDIR /workspace/part2_ch03_06/build
