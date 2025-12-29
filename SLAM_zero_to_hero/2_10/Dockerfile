FROM slam_zero_to_hero:base

WORKDIR /workspace
COPY . /workspace/2_10

RUN cd /workspace/2_10 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

WORKDIR /workspace/2_10/build
CMD ["sh", "-c", "./icp_point_to_point && echo '' && ./icp_point_to_plane && echo '' && ./icp_convergence"]
