FROM slam:base

# nanobind requires Python development headers (already in base)
# Additional tools for building
RUN apt-get update && apt-get install -y \
    python3-numpy \
    && apt-get clean

WORKDIR /workspace
COPY . /workspace/1_15

# Build the nanobind module
RUN cd /workspace/1_15 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

# Create output directory
RUN mkdir -p /output

WORKDIR /workspace/1_15

# Set Python path for running examples
ENV PYTHONPATH=/workspace/1_15/build:${PYTHONPATH}
