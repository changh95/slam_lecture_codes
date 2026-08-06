FROM slam:base

# rerun C++ SDK: the demos stream live to a viewer on the host while they run.
# Pinned to the host viewer version (live gRPC streaming requires the SDK and
# viewer versions to match).
RUN wget -q https://github.com/rerun-io/rerun/releases/download/0.33.0/rerun_cpp_sdk.zip && \
    cmake -E tar xf rerun_cpp_sdk.zip && \
    cmake -S rerun_cpp_sdk -B rerun_cpp_sdk/build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build rerun_cpp_sdk/build -j4 --target install && \
    ldconfig && \
    rm -rf rerun_cpp_sdk rerun_cpp_sdk.zip

# slam:base ships Python rerun-sdk 0.28.1, which cannot talk to a 0.33.0 viewer.
# The demos themselves stream through the C++ SDK above, but this pin keeps the
# image's Python SDK usable against the same viewer (ad-hoc inspection, `rerun`
# CLI helpers) instead of silently mismatched. Nothing here needs a 2D plotting
# library: all visualization goes through rerun.
RUN pip3 install --break-system-packages 'rerun-sdk==0.33.0'

WORKDIR /workspace
COPY . /workspace/part3_ch01_14

# Build
RUN cd /workspace/part3_ch01_14 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

# BAL dataset used by gtsam_bundle_adjustment (Trafalgar Square, 21 cameras,
# 11315 points, 36455 observations).
RUN cd /workspace/part3_ch01_14/build && \
    wget -q https://grail.cs.washington.edu/projects/bal/data/trafalgar/problem-21-11315-pre.txt.bz2 && \
    bunzip2 problem-21-11315-pre.txt.bz2

WORKDIR /workspace/part3_ch01_14/build
