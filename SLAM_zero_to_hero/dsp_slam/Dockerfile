FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04

MAINTAINER changh95
ARG DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

ARG FORCE_CUDA=1
ENV FORCE_CUDA=${FORCE_CUDA}

RUN apt-get update -y && apt-get upgrade -y

RUN apt-get update -y &&\
    apt-get install build-essential -y && \
    apt-get install cmake -y && \
    apt-get install git -y && \
    apt-get install sudo -y && \
    apt-get install wget -y && \
    apt-get install ninja-build -y && \
    apt-get install software-properties-common -y && \
    apt-get install python3 -y && \
    apt-get install python3-pip -y && \
    apt-get install -y ssh && \
    apt-get install -y gcc && \
    apt-get install -y g++ && \
    apt-get install -y gdb && \
    apt-get install -y cmake && \
    apt-get install -y rsync && \
    apt-get install -y tar && \
    apt-get install -y x11-utils && \
    apt-get install -y x11-apps && \
    apt-get install -y zip &&\
    apt-get install -y vim &&\
    apt-get install -y libglvnd-dev

# Conda (miniconda)
ENV CONDA_DIR /opt/conda
ENV CONDA_OVERRIDE_CUDA "11.3.1"
ENV CONDA_OVERRIDE_GLIBC="2.17"

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

ENV PATH=$CONDA_DIR/bin:$PATH 

RUN conda update -n base conda &&\
    conda install -n base conda-libmamba-solver &&\
    conda config --set solver libmamba

# Clone DSP-SLAM
WORKDIR /
RUN git clone --recursive https://github.com/JingwenWang95/DSP-SLAM.git
WORKDIR /DSP-SLAM

