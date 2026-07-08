FROM slam:base

# AprilTag library (not part of slam:base)
RUN git clone https://github.com/AprilRobotics/apriltag.git -b v3.4.2 && \
    cd apriltag && \
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_WRAPPER=OFF && \
    cmake --build build -j4 --target install && \
    ldconfig && \
    cd .. && rm -rf apriltag

# Python deps for the rerun visualization script:
#  - rerun-sdk pinned to the host viewer version (live gRPC streaming requires
#    the SDK and viewer versions to match)
#  - opencv-python-headless so viz_pnp.py can read frames from the mp4
RUN pip3 install --break-system-packages \
    rerun-sdk==0.33.0 \
    opencv-python-headless

WORKDIR /workspace
COPY . /workspace/part2_ch02_09

RUN cd /workspace/part2_ch02_09 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

WORKDIR /workspace/part2_ch02_09/build
