FROM slam_zero_to_hero:base

WORKDIR /workspace
COPY . /workspace/2_12

# Build the examples
RUN cd /workspace/2_12 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

WORKDIR /workspace/2_12/build

# Run all examples by default
CMD ["sh", "-c", "\
    echo '========================================' && \
    echo '   Advanced ICP Methods Tutorial' && \
    echo '========================================' && \
    echo '' && \
    echo '[1/3] Running G-ICP Example...' && \
    echo '--------------------------------------' && \
    ./gicp_example && \
    echo '' && \
    echo '[2/3] Running NDT Example...' && \
    echo '--------------------------------------' && \
    ./ndt_example && \
    echo '' && \
    echo '[3/3] Running Registration Comparison...' && \
    echo '--------------------------------------' && \
    ./registration_comparison && \
    echo '' && \
    echo '========================================' && \
    echo '   All examples completed!' && \
    echo '========================================' \
"]
