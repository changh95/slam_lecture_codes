FROM slam_zero_to_hero:base

# Install SymForce Python package
RUN pip3 install symforce --break-system-packages

WORKDIR /workspace
COPY . /workspace/3_16

WORKDIR /workspace/3_16

CMD ["sh", "-c", "echo '=== SymForce Tutorial ===' && \
    echo '' && \
    echo 'Running Python examples:' && \
    python3 examples/symforce_basics.py"]
