FROM slam:base

WORKDIR /workspace
COPY . /workspace/part3_ch01_16

# matplotlib for 2D plots (SymForce + rerun-sdk are already in slam:base).
RUN pip3 install --break-system-packages matplotlib 'rerun-sdk==0.33.0'

# Working dir for running examples + a sample BAL dataset for bundle adjustment.
RUN mkdir -p /workspace/part3_ch01_16/run && \
    cd /workspace/part3_ch01_16/run && \
    wget -q https://grail.cs.washington.edu/projects/bal/data/trafalgar/problem-21-11315-pre.txt.bz2 && \
    bunzip2 problem-21-11315-pre.txt.bz2 || true

WORKDIR /workspace/part3_ch01_16/run
