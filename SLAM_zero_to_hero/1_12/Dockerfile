FROM slam_zero_to_hero:base

# Install Valgrind and visualization tools
RUN apt-get update && apt-get install -y \
    valgrind \
    graphviz \
    && apt-get clean

WORKDIR /workspace
COPY . /workspace/1_12

# Build in Debug mode for better Valgrind output
RUN cd /workspace/1_12 && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Debug && \
    make -j4

# Create output directory for profiling results
RUN mkdir -p /output

WORKDIR /workspace/1_12/build

# Default: run memory examples and generate massif output
CMD ["sh", "-c", "echo '=== Running memory examples with Valgrind ===' && \
    echo '' && \
    echo '1. Memory leak detection:' && \
    valgrind --leak-check=full --show-leak-kinds=all ./memory_examples --with-leak 2>&1 | head -100 && \
    echo '' && \
    echo '2. Memory profiling with Massif:' && \
    valgrind --tool=massif --massif-out-file=/output/massif.out ./memory_examples && \
    ms_print /output/massif.out > /output/massif_report.txt && \
    cp /output/massif.out /output/ && \
    echo '' && \
    echo 'Massif output saved to /output/massif.out' && \
    echo 'Massif report saved to /output/massif_report.txt' && \
    echo '' && \
    echo 'View report with: cat /output/massif_report.txt'"]
