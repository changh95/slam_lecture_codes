FROM ubuntu:jammy

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y build-essential cmake git wget \
    python3 python3-pip python3-dev \
    libeigen3-dev libgl1-mesa-dev && \
    apt-get clean

# Install KISS-SLAM (v0.0.2)
RUN pip3 install kiss-slam==0.0.2

WORKDIR /data
