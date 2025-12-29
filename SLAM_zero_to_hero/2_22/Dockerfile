FROM slam:base

WORKDIR /workspace
COPY . /workspace/2_22

# Build the RANSAC/USAC examples
RUN cd /workspace/2_22 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

WORKDIR /workspace/2_22/build
