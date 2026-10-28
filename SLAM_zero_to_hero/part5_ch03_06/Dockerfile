FROM slam:base

WORKDIR /workspace
COPY . /workspace/part5_ch03_06

# Build the examples
RUN cd /workspace/part5_ch03_06 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

WORKDIR /workspace/part5_ch03_06/build
