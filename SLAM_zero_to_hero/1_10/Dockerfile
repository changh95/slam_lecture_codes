FROM slam_zero_to_hero:base

WORKDIR /workspace
COPY . /workspace/1_10

RUN cd /workspace/1_10 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

WORKDIR /workspace/1_10/build
CMD ["sh", "-c", "./use_static_lib && echo '' && ./use_shared_lib && echo '' && ./use_header_only"]
