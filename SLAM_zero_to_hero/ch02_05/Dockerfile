FROM slam:base

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
