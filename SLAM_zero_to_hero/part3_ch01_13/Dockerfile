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

WORKDIR /workspace

COPY . /workspace/part3_ch01_13

# Build the three demos. g2o and Eigen come from slam:base.
RUN cd /workspace/part3_ch01_13 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

# BAL dataset for the bundle-adjustment demo (Trafalgar Square, 21 cameras,
# 11315 points, 36455 observations).
RUN cd /workspace/part3_ch01_13/build && \
    wget -q https://grail.cs.washington.edu/projects/bal/data/trafalgar/problem-21-11315-pre.txt.bz2 && \
    bunzip2 problem-21-11315-pre.txt.bz2 && \
    test -s problem-21-11315-pre.txt

WORKDIR /workspace/part3_ch01_13/build
