FROM slam:base

WORKDIR /workspace

# Copy project files
COPY . /workspace/2_29

# Build the project
RUN cd /workspace/2_29 && \
    mkdir -p build && cd build && \
    cmake .. && \
    make -j4

# Create output directory for results
RUN mkdir -p /output

WORKDIR /workspace/2_29/build
