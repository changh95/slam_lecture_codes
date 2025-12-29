FROM slam:base

WORKDIR /workspace
COPY . /workspace/2_14

# Build the project
RUN cd /workspace/2_14 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

# Create output directory for visualization results
RUN mkdir -p /output

WORKDIR /workspace/2_14/build

# Run the essential/fundamental matrix estimation example
CMD ["sh", "-c", "./essential_fundamental_estimation && cp -f epipolar_lines.png /output/ 2>/dev/null; echo 'Done! Check /output for visualization results.'"]
