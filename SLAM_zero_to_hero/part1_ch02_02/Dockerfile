FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace/part1_ch02_02

RUN cd /workspace/part1_ch02_02 && \
    rm -rf build && mkdir build && cd build && \
    cmake .. && \
    make -j4

WORKDIR /workspace/part1_ch02_02/build
