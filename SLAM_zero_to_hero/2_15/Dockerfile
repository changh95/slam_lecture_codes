FROM slam:base

WORKDIR /workspace
COPY . /workspace/2_15

# Build the project
RUN cd /workspace/2_15 && \
    mkdir -p build && cd build && \
    cmake .. && \
    make -j4

# Create output directory for results
RUN mkdir -p /output

WORKDIR /workspace/2_15/build
