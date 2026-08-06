FROM slam:base

# slam:base ships SymForce and a rerun-sdk, but that rerun-sdk is 0.28.1. Live
# gRPC streaming requires the SDK and the host viewer to be the same version, so
# this pin to 0.33.0 (the version the host viewer runs) is load-bearing.
RUN pip3 install --break-system-packages 'rerun-sdk==0.33.0'

# Working dir for running the examples + the BAL problem for bundle adjustment.
WORKDIR /workspace
RUN mkdir -p /workspace/part3_ch01_16/run && \
    cd /workspace/part3_ch01_16/run && \
    wget -q https://grail.cs.washington.edu/projects/bal/data/trafalgar/problem-21-11315-pre.txt.bz2 && \
    bunzip2 problem-21-11315-pre.txt.bz2

COPY . /workspace/part3_ch01_16

WORKDIR /workspace/part3_ch01_16/run
