FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace/1_10

RUN cd /workspace/1_10 && \
    rm -rf build && mkdir build && cd build && \
    cmake .. && \
    make -j4

WORKDIR /workspace/1_10/build
