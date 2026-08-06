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

# Kimera-RPGO. GTSAM, Boost, Eigen and TBB all come from slam:base - only RPGO
# itself is built here.
#
# The commit pin and the base image's GTSAM version are a MATCHED PAIR: this
# commit compiles against the GTSAM 4.3a1 that slam:base ships. Older RPGO
# revisions need GTSAM 4.2, and building 4.2 from source here would cost ~40
# minutes and shadow the base image's install. Bump one, re-check the other.
# (No --depth 1: the clone has to be deep enough to check the commit out.)
RUN git clone https://github.com/MIT-SPARK/Kimera-RPGO.git /tmp/kimera-rpgo && \
    cd /tmp/kimera-rpgo && \
    git checkout d28b4df0570d642a2bb00e511344ce1110f87519 && \
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j4 --target install && \
    ldconfig && \
    cd / && rm -rf /tmp/kimera-rpgo

WORKDIR /workspace

# Copy project files
COPY . /workspace/part3_ch01_17

# Build the three examples
RUN cd /workspace/part3_ch01_17 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

WORKDIR /workspace/part3_ch01_17/build

CMD ["/bin/bash"]
