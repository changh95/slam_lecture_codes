FROM slam:base

WORKDIR /workspace
COPY . /workspace/2_12

# Build the examples
RUN cd /workspace/2_12 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

WORKDIR /workspace/2_12/build
