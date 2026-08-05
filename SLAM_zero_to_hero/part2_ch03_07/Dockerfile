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

# TEASER++: the real solver, so teaser_demo runs certifiably-robust global
# registration rather than a stand-in. Tests/bindings are off - only the
# registration library is needed. Pinned to a master commit for reproducibility.
RUN git clone https://github.com/MIT-SPARK/TEASER-plusplus.git && \
    cd TEASER-plusplus && \
    git checkout 52a9c52ee7d4c838c5e8a75458c33178be5bfb70 && \
    cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTS=OFF \
        -DBUILD_DOC=OFF \
        -DBUILD_TEASER_FPFH=OFF \
        -DBUILD_MATLAB_BINDINGS=OFF \
        -DBUILD_PYTHON_BINDINGS=OFF && \
    cmake --build build -j8 --target install && \
    ldconfig && \
    cd .. && rm -rf TEASER-plusplus

WORKDIR /workspace

# Copy project files
COPY . /workspace/part2_ch03_07

# Build the project. TEASER++ is installed above, so teaser_demo is built
# against the real library.
RUN cd /workspace/part2_ch03_07 && \
    mkdir build && cd build && \
    cmake .. -DUSE_TEASER=ON && \
    make -j4

# Create output directory for results
RUN mkdir -p /output

WORKDIR /workspace/part2_ch03_07/build
