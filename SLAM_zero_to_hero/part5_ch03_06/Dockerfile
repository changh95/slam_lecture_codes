FROM slam:base

WORKDIR /workspace

# komrad36's code is vendored in third_party/komrad36 (copies of the upstream
# repositories, see its README), so it is copied in with the exercise rather
# than cloned at build time.
COPY . /workspace/part5_ch03_06

# Build the examples
RUN cd /workspace/part5_ch03_06 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

WORKDIR /workspace/part5_ch03_06/build
