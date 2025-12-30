FROM slam:base

WORKDIR /workspace
COPY . /workspace/1_21

# Build
RUN cd /workspace/1_21 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

WORKDIR /workspace/1_21/build
