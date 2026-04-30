FROM slam:base

WORKDIR /workspace
COPY . /workspace/part3_ch01_13

# Fix spdlog/fmt version mismatch: system libspdlog.a was compiled against fmt v8
# but system libfmt-dev provides v10+. Rebuild spdlog from source with bundled fmt.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libopenmpi-dev && \
    rm -rf /var/lib/apt/lists/* && \
    git clone https://github.com/gabime/spdlog.git -b v1.12.0 /tmp/spdlog && \
    cd /tmp/spdlog && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DSPDLOG_FMT_EXTERNAL=OFF && \
    make -j4 && make install && ldconfig && \
    rm -rf /tmp/spdlog

# Build
RUN cd /workspace/part3_ch01_13 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

# Download sample BAL dataset
RUN cd /workspace/part3_ch01_13/build && \
    wget -q https://grail.cs.washington.edu/projects/bal/data/ladybug/problem-49-7776-pre.txt.bz2 && \
    bunzip2 problem-49-7776-pre.txt.bz2 || true

WORKDIR /workspace/part3_ch01_13/build
