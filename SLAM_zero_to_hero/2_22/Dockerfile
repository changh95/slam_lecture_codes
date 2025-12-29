FROM slam:base

WORKDIR /workspace
COPY . /workspace/2_22

# Build the RANSAC/USAC examples
RUN cd /workspace/2_22 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

WORKDIR /workspace/2_22/build

# Default command: run all examples
CMD ["sh", "-c", "\
    echo '=== RANSAC Homography Estimation ===' && \
    ./ransac_homography && \
    echo '' && \
    echo '=== RANSAC Fundamental Matrix Estimation ===' && \
    ./ransac_fundamental && \
    echo '' && \
    echo '=== Custom RANSAC Implementation ===' && \
    ./ransac_custom \
"]
