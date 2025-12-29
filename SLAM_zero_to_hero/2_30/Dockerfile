FROM slam:base

WORKDIR /workspace

# Copy project files
COPY . /workspace/2_30

# Build the project
# Note: TEASER++ and KISS-ICP are optional dependencies
# To enable them, install the libraries and rebuild with:
#   cmake -DUSE_TEASER=ON -DUSE_KISS_ICP=ON ..
RUN cd /workspace/2_30 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

# Create output directory for results
RUN mkdir -p /output

WORKDIR /workspace/2_30/build

# Default command: show usage and run demos
CMD ["sh", "-c", "\
    echo '=== Advanced ICP Methods Tutorial ===' && \
    echo '' && \
    echo 'Available executables:' && \
    echo '  ./gicp_demo [source.pcd target.pcd]' && \
    echo '  ./ndt_demo [source.pcd target.pcd] [resolution]' && \
    echo '  ./teaser_demo [source.pcd target.pcd]' && \
    echo '  ./kiss_icp_demo [sequence_dir]' && \
    echo '  ./method_comparison [source.pcd target.pcd]' && \
    echo '' && \
    echo '=== Running demos with synthetic data ===' && \
    echo '' && \
    echo '--- GICP Demo ---' && \
    ./gicp_demo && \
    echo '' && \
    echo '--- NDT Demo ---' && \
    ./ndt_demo && \
    echo '' && \
    echo '--- Method Comparison ---' && \
    ./method_comparison \
"]
