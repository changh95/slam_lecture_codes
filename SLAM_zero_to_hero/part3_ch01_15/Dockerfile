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
COPY . /workspace/part3_ch01_15

# Build. Ceres, Eigen and glog all come from slam:base; nothing else is needed
# (the demos are pure C++ - there is no Python or plotting step any more).
RUN cd /workspace/part3_ch01_15 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

# Download sample BAL dataset (Trafalgar Square: 21 cameras, 11315 points).
# The final `test -s` makes a failed download fail the build: the README promises
# the image already contains the dataset, so a green build with a missing file
# would only surface as a "cannot open" at run time.
RUN cd /workspace/part3_ch01_15/build && \
    wget -q https://grail.cs.washington.edu/projects/bal/data/trafalgar/problem-21-11315-pre.txt.bz2 && \
    bunzip2 problem-21-11315-pre.txt.bz2 && \
    test -s problem-21-11315-pre.txt

WORKDIR /workspace/part3_ch01_15/build
