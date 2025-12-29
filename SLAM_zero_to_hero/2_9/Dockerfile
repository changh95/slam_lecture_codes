FROM slam:base

WORKDIR /workspace
COPY . /workspace/2_9

RUN cd /workspace/2_9 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

WORKDIR /workspace/2_9/build
