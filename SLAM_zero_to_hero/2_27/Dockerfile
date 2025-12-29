FROM slam:base

WORKDIR /workspace
COPY . /workspace/2_27

# Build
RUN cd /workspace/2_27 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

WORKDIR /workspace/2_27/build
