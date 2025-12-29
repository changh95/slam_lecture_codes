FROM --platform=linux/amd64 ubuntu:jammy

MAINTAINER changh95
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get upgrade -y

RUN apt-get install build-essential -y && \
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
    apt-get clean

RUN apt-get install -y mesa-utils && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get install -y libglu1-mesa-dev && \
    apt-get install -y libglew-dev &&\
    apt-get install -y libglvnd-dev &&\
    apt-get install -y libgl1-mesa-dev &&\
    apt-get install -y libegl1-mesa-dev &&\
    apt-get install -y mesa-common-dev

RUN apt-get install -y qtbase5-dev qt5-qmake libqglviewer-dev-qt5

RUN cd / && \
    mkdir data && cd data && \
    wget http://ais.informatik.uni-freiburg.de/projects/datasets/octomap/fr_campus.graph.gz && \
    gzip -d fr_campus.graph.gz
    
RUN cd / && \
    git clone https://github.com/OctoMap/octomap.git &&\
    cd octomap && \
    cd octomap && \
    mkdir build && cd build &&\
    cmake .. && make -j4 && make install &&\
    cd ../../octovis &&\
    mkdir build && cd build &&\
    cmake .. && make -j4
