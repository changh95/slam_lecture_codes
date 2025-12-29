FROM slam:base

WORKDIR /workspace
COPY . /workspace/3_15

# Build
RUN cd /workspace/3_15 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

# Download sample BAL dataset
RUN cd /workspace/3_15/build && \
    wget -q https://grail.cs.washington.edu/projects/bal/data/ladybug/problem-49-7776-pre.txt.bz2 && \
    bunzip2 problem-49-7776-pre.txt.bz2 || true

WORKDIR /workspace/3_15/build
