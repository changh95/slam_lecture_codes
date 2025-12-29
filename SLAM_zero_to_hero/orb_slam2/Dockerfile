FROM ubuntu:jammy 

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
    
# ORB-SLAM3
RUN apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev -y
RUN apt-get install -y libgl1-mesa-dev libwayland-dev libxkbcommon-dev wayland-protocols libegl1-mesa-dev libc++-dev libglew-dev libeigen3-dev
RUN apt-get install -y libboost-all-dev libssl-dev
RUN git clone https://github.com/changh95/Portable_ORB_SLAM2.git &&\
    cd Portable_ORB_SLAM2 &&\
    python3 ./buildDeps.py &&\
    python3 ./build.py
