FROM slam_zero_to_hero:base

WORKDIR /workspace
COPY . /workspace/2_27

# Build
RUN cd /workspace/2_27 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

WORKDIR /workspace/2_27/build

CMD ["sh", "-c", "echo '=== Point Cloud Processing Tutorial ===' && \
    echo '' && \
    echo '1. Running Basic I/O:' && \
    ./basic_io /workspace/2_27/output && \
    echo '' && \
    echo '2. Running Filtering:' && \
    ./filtering /workspace/2_27/output && \
    echo '' && \
    echo '3. Running Normal Estimation:' && \
    ./normal_estimation /workspace/2_27/output && \
    echo '' && \
    echo '4. Running Segmentation:' && \
    ./segmentation /workspace/2_27/output && \
    echo '' && \
    echo '5. Running Visualization (headless mode):' && \
    ./visualization /workspace/2_27/output && \
    echo '' && \
    echo 'All demos completed. Output files in /workspace/2_27/output/'"]
