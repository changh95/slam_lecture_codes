FROM slam_zero_to_hero:base

WORKDIR /workspace
COPY . /workspace/2_17

RUN cd /workspace/2_17 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

WORKDIR /workspace/2_17/build

CMD ["./triangulation_demo"]
