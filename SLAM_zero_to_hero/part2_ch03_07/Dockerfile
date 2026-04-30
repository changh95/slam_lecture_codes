FROM slam:base

WORKDIR /workspace

# Copy project files
COPY . /workspace/part2_ch03_07

# Build the project
# Note: TEASER++ and KISS-ICP are optional dependencies
# To enable them, install the libraries and rebuild with:
#   cmake -DUSE_TEASER=ON -DUSE_KISS_ICP=ON ..
RUN cd /workspace/part2_ch03_07 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

# Create output directory for results
RUN mkdir -p /output

WORKDIR /workspace/part2_ch03_07/build
