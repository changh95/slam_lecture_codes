FROM slam:base

WORKDIR /workspace
COPY . /workspace/1_11

RUN cd /workspace/1_11 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

# Create output directory for profile files
RUN mkdir -p /output

WORKDIR /workspace/1_11/build
