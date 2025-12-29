FROM slam:base

WORKDIR /workspace
COPY . /workspace/2_10

RUN cd /workspace/2_10 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

WORKDIR /workspace/2_10/build
