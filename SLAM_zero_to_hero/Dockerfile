FROM --platform=linux/amd64 ubuntu:noble

MAINTAINER changh95
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install build-essential -y && \
    apt-get install cmake -y && \
    apt-get install git -y && \
    apt-get install sudo -y && \
    apt-get install wget -y && \
    apt-get install ninja-build -y && \
    apt-get install software-properties-common -y && \
    apt-get install python3 -y && \
    apt-get install python3-pip -y && \
    apt-get install python3-dev -y && \
    apt-get install -y ssh && \
    apt-get install -y gcc && \
    apt-get install -y g++ && \
    apt-get install -y gdb && \
    apt-get install -y cmake && \
    apt-get install -y rsync && \
    apt-get install -y tar && \
    apt-get install -y x11-utils && \
    apt-get install -y x11-apps && \
    apt-get install -y zip && \
    apt-get install -y libfmt-dev && \
    apt-get install -y libspdlog-dev && \
    apt-get clean

# Rerun (3D viewer for robotics)
RUN pip3 install rerun-sdk --break-system-packages

# OpenCV with contrib modules (for ArUco, xfeatures2d/TEBLID, etc.)
RUN apt-get install -y cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
RUN wget https://github.com/opencv/opencv/archive/refs/tags/4.12.0.zip &&\
    wget https://github.com/opencv/opencv_contrib/archive/refs/tags/4.12.0.zip -O opencv_contrib-4.12.0.zip &&\
    unzip 4.12.0.zip &&\
    unzip opencv_contrib-4.12.0.zip &&\
    cd opencv-4.12.0 &&\
    mkdir build && cd build &&\
    cmake .. \
        -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.12.0/modules \
        -DOPENCV_ENABLE_NONFREE=ON \
        -DBUILD_opencv_python3=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF &&\
    make -j16 &&\
    make install &&\
    cd ../../

# Eigen
RUN wget https://gitlab.com/libeigen/eigen/-/archive/5.0.0/eigen-5.0.0.zip &&\
    unzip eigen-5.0.0.zip &&\
    cd eigen-5.0.0 &&\
    mkdir build && cd build &&\
    cmake .. &&\
    make -j16 &&\
    make install &&\
    cd ../../

# Ceres-solver [Use the latest branch, as latest release (2.2.0) does not support Eigen 5.0.0]
RUN apt-get install -y libgoogle-glog-dev libgflags-dev libatlas-base-dev libsuitesparse-dev &&\
    git clone https://github.com/ceres-solver/ceres-solver.git --recursive &&\
    cd ceres-solver && \
    mkdir build && cd build &&\
    cmake .. &&\
    make -j16 &&\
    make install &&\
    cd ../../

# Sophus [EDIT: Remove 3.4.0 dependency to enable build with Eigen 5.0.0]
RUN wget https://github.com/strasdat/Sophus/archive/refs/tags/1.24.6.zip &&\
    unzip 1.24.6.zip &&\
    cd Sophus-1.24.6 &&\
    sed -i 's/find_package(Eigen3 3.4.0/find_package(Eigen3/g' CMakeLists.txt &&\
    mkdir build && cd build &&\
    cmake .. &&\
    make -j16 &&\
    make install && \
    cd ../../

# Pangolin
RUN apt-get update && \
    apt-get install -y mesa-utils \
    libgl1 \
    libglu1-mesa-dev \
    libglew-dev \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    mesa-common-dev \
    libepoxy-dev

RUN wget https://github.com/stevenlovegrove/Pangolin/archive/refs/tags/v0.9.4.zip &&\
    unzip v0.9.4.zip &&\
    cd Pangolin-0.9.4 &&\
    mkdir build && cd build &&\
    cmake .. &&\
    make -j16 &&\
    make install &&\
    cd ../../
    
# MPI (required by VTK for PCL visualization)
RUN apt-get install -y libopenmpi-dev

# PCL
RUN apt-get install -y libpcl-dev

# easy_profiler (CPU profiler with GUI visualization)
RUN apt-get install -y qtbase5-dev libqt5widgets5 &&\
    git clone https://github.com/yse/easy_profiler.git &&\
    cd easy_profiler &&\
    mkdir build && cd build &&\
    cmake .. -DEASY_PROFILER_NO_GUI=OFF &&\
    make -j16 &&\
    make install &&\
    ldconfig &&\
    cd ../../

# g2o [Use latest main branch for Eigen 5.0.0 support]
RUN git clone https://github.com/RainerKuemmerle/g2o.git &&\
    cd g2o &&\
    mkdir build && cd build &&\
    cmake .. &&\
    make -j16 &&\
    make install &&\
    cd ../../

# GTSAM [Use 4.3a1 for Eigen 5.0.0 support]
# Patch: Add missing #include <cassert> in SL4.cpp
# Disable warnings-as-errors for Eigen 5.0.0 compatibility (uninitialized variable warnings)
RUN apt-get install -y libtbb-dev &&\
    git clone https://github.com/borglab/gtsam.git -b 4.3a1 &&\
    cd gtsam &&\
    sed -i '1i #include <cassert>' gtsam/geometry/SL4.cpp &&\
    mkdir build && cd build &&\
    cmake .. -DGTSAM_USE_SYSTEM_EIGEN=ON -DCMAKE_CXX_FLAGS="-Wno-error=maybe-uninitialized" &&\
    make -j16 &&\
    make install &&\
    cd ../../

# PoseLib (minimal pose estimation solvers)
# Note: Explicitly set Eigen3_DIR for Eigen 5.0.0 compatibility
RUN git clone https://github.com/PoseLib/PoseLib.git &&\
    cd PoseLib &&\
    mkdir build && cd build &&\
    cmake .. -DCMAKE_BUILD_TYPE=Release -DEigen3_DIR=/usr/local/share/eigen3/cmake &&\
    make -j16 &&\
    make install &&\
    ldconfig &&\
    cd ../../

# OpenGV (geometric vision algorithms)
# Note: Explicitly set Eigen3_DIR for Eigen 5.0.0 compatibility
RUN git clone https://github.com/laurentkneip/opengv.git &&\
    cd opengv &&\
    mkdir build && cd build &&\
    cmake .. -DCMAKE_BUILD_TYPE=Release -DEigen3_DIR=/usr/local/share/eigen3/cmake &&\
    make -j16 &&\
    make install &&\
    ldconfig &&\
    cd ../../

# RansacLib (header-only RANSAC library for education)
# Note: Explicitly set Eigen3_DIR for Eigen 5.0.0 compatibility
RUN git clone https://github.com/tsattler/RansacLib.git &&\
    cd RansacLib &&\
    mkdir build && cd build &&\
    cmake .. -DCMAKE_BUILD_TYPE=Release -DEigen3_DIR=/usr/local/share/eigen3/cmake &&\
    make -j16 &&\
    make install &&\
    ldconfig &&\
    cd ../../

# MAGSAC (state-of-the-art threshold-free robust estimation)
# Note: Explicitly set Eigen3_DIR for Eigen 5.0.0 compatibility
RUN apt-get install -y libgflags-dev &&\
    git clone https://github.com/danini/magsac.git --recursive &&\
    cd magsac &&\
    mkdir build && cd build &&\
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCREATE_SAMPLE_PROJECT=OFF -DEigen3_DIR=/usr/local/share/eigen3/cmake &&\
    make -j16 &&\
    make install &&\
    ldconfig &&\
    cd ../../

# SymForce (requires libgmp-dev for SymEngine)
RUN apt-get install -y libgmp-dev &&\
    git clone https://github.com/symforce-org/symforce.git &&\
    cd symforce &&\
    pip3 install . --break-system-packages &&\
    cd ../

RUN git clone https://github.com/changh95/fastcampus_slam_codes.git
