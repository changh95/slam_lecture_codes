FROM slam:base

WORKDIR /workspace
COPY . /workspace/part3_ch01_14

# libopenmpi-dev: VTK runtime dep; matplotlib: 2D plots (rerun-sdk is in base).
RUN apt-get update && \
    apt-get install -y --no-install-recommends libopenmpi-dev && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --break-system-packages matplotlib 'rerun-sdk==0.33.0'

# Build
RUN cd /workspace/part3_ch01_14 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

# Download sample BAL dataset
RUN cd /workspace/part3_ch01_14/build && \
    wget -q https://grail.cs.washington.edu/projects/bal/data/trafalgar/problem-21-11315-pre.txt.bz2 && \
    bunzip2 problem-21-11315-pre.txt.bz2 || true

WORKDIR /workspace/part3_ch01_14/build
