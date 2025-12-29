FROM slam:base

WORKDIR /workspace
COPY . /workspace/2_7

RUN cd /workspace/2_7 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

WORKDIR /workspace/2_7/build

# Run both optical flow examples
CMD ["sh", "-c", "echo '=== Sparse Optical Flow (Lucas-Kanade) ===' && ./sparse_optical_flow && echo '' && echo '=== Dense Optical Flow (Farneback) ===' && ./dense_optical_flow"]
