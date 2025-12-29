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

# Set Python path and run examples
ENV PYTHONPATH=/workspace/1_15/build:${PYTHONPATH}

CMD ["sh", "-c", "echo '=== Testing nanobind SLAM bindings ===' && \
    python3 examples/test_bindings.py && \
    echo '' && \
    echo 'Build artifacts:' && \
    ls -la build/slam_bindings*.so && \
    cp build/slam_bindings*.so /output/ 2>/dev/null || true"]
