FROM slam:base

# AprilTag library (not part of slam:base)
RUN git clone https://github.com/AprilRobotics/apriltag.git -b v3.4.2 && \
    cd apriltag && \
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_WRAPPER=OFF && \
    cmake --build build -j4 --target install && \
    ldconfig && \
    cd .. && rm -rf apriltag

# rerun C++ SDK: the demos stream live to a viewer on the host while they
# run. Pinned to the host viewer version (live gRPC streaming requires the
# SDK and viewer versions to match).
RUN wget -q https://github.com/rerun-io/rerun/releases/download/0.33.0/rerun_cpp_sdk.zip && \
    cmake -E tar xf rerun_cpp_sdk.zip && \
    cmake -S rerun_cpp_sdk -B rerun_cpp_sdk/build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build rerun_cpp_sdk/build -j4 --target install && \
    ldconfig && \
    rm -rf rerun_cpp_sdk rerun_cpp_sdk.zip

# Python deps for the offline rerun visualization script (viz_pnp.py):
#  - rerun-sdk pinned to the same version as the C++ SDK / host viewer
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

# Default: push the viewer layout, then run all three PnP demos back to back,
# each streaming live to the rerun viewer on the host (requires: rerun viewer
# started on the host and docker run --network=host). The video frames are
# identical for every method, so only the first demo streams them.
CMD ["bash", "-c", "python3 ../blueprint.py; ./pnp_opencv && ./pnp_poselib --no-stream-images && ./pnp_opengv --no-stream-images"]
