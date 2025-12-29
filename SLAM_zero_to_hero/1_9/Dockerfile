FROM slam_zero_to_hero:base

WORKDIR /workspace
COPY . /workspace/1_9

RUN cd /workspace/1_9 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

WORKDIR /workspace/1_9/build
CMD ["./for_loop"]
