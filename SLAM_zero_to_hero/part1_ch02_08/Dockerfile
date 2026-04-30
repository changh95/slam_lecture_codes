FROM slam:base

# nanobind requires Python development headers (already in base)
# Additional tools for building
RUN apt-get update && apt-get install -y \
    python3-numpy \
    && apt-get clean

WORKDIR /workspace
COPY . /workspace/part1_ch02_08

# Build the nanobind module
RUN cd /workspace/part1_ch02_08 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

# Create output directory
RUN mkdir -p /output

WORKDIR /workspace/part1_ch02_08

# Set Python path for running examples
ENV PYTHONPATH=/workspace/part1_ch02_08/build:${PYTHONPATH}
