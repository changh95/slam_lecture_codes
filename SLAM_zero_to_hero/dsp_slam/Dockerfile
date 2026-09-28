# DSP-SLAM (Wang, Ruenz, Agapito, 3DV 2021) on CUDA 12.8 / PyTorch 2.7.
#
# Upstream targets CUDA 11.3 + PyTorch 1.10 + mmdetection3d 0.17, which cannot
# run kernels on Blackwell (sm_120). This image keeps the C++ side as upstream
# built it (OpenCV 3.4, Eigen 3.4, Pangolin from Feb 2022) and runs the Python
# side (DeepSDF shape optimisation) on PyTorch 2.7 cu128. The detectors run
# offline: KITTI 07 ships pre-computed MaskRCNN / PointPillars labels, so
# mmdetection is not installed (`detect_online: false` in config/config_kitti.json).
# The base image is the plain CUDA 12.8 one: the torch cu128 wheel brings its own
# CUDA runtime and cuDNN.
FROM docker.io/nvidia/cuda:12.8.1-base-ubuntu22.04

LABEL maintainer="changh95"
ARG DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git wget ca-certificates pkg-config unzip \
        python3 python3-dev python3-pip \
        libgtk2.0-dev libjpeg-dev libpng-dev \
        libeigen3-dev libglew-dev libgl1-mesa-dev libegl1-mesa-dev \
        libwayland-dev libxkbcommon-dev wayland-protocols libepoxy-dev \
        libgl1-mesa-dri mesa-utils xvfb x11-apps x11-utils imagemagick xdotool \
    && rm -rf /var/lib/apt/lists/*

# OpenCV 3.4 (DSP-SLAM's CMake asks for OpenCV 3; upstream used 3.4.1, which
# no longer compiles with gcc 11 -- 3.4.16 is the last 3.4 release that does).
WORKDIR /opt
RUN git clone --branch 3.4.16 --depth 1 https://github.com/opencv/opencv.git && \
    cmake -S opencv -B opencv/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_LIST=core,imgproc,imgcodecs,highgui,features2d,calib3d,flann,photo,video,objdetect,ml \
        -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DWITH_GTK=ON -DWITH_QT=OFF -DWITH_FFMPEG=OFF \
        -DWITH_GSTREAMER=OFF -DWITH_TIFF=OFF -DWITH_WEBP=OFF -DWITH_OPENEXR=OFF -DWITH_IPP=OFF \
        -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF \
        -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DBUILD_JAVA=OFF && \
    cmake --build opencv/build -j"$(nproc)" && cmake --install opencv/build && \
    rm -rf opencv

# Pangolin, pinned to the master commit upstream's build script would have
# cloned when DSP-SLAM was frozen (16 Mar 2022).
RUN git clone https://github.com/stevenlovegrove/Pangolin.git && cd Pangolin && \
    git checkout 3f4a8b8d9e3cd91818128ac99e15aa6230770e44 && git submodule update --init --recursive && \
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF \
        -DBUILD_PANGOLIN_PYTHON=OFF -DBUILD_PYPANGOLIN_MODULE=OFF -DBUILD_TOOLS=OFF && \
    cmake --build build -j"$(nproc)" && cmake --install build && ldconfig && \
    cd .. && rm -rf Pangolin

# Python side: PyTorch 2.7 cu128 runs on sm_120. numpy stays on 1.x.
RUN pip3 install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128 && \
    pip3 install --no-cache-dir "numpy==1.26.4" "scikit-image==0.24.0" "addict==2.4.0" \
        "plyfile==1.1" "opencv-python-headless==4.10.0.84" "open3d==0.19.0" "scipy==1.14.1"
# The KITTI 07 labels are pickled numpy arrays; PyTorch >= 2.6 refuses them by
# default (weights_only=True).
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# DSP-SLAM, pinned to the last upstream commit.
WORKDIR /
RUN git clone --recursive https://github.com/JingwenWang95/DSP-SLAM.git && \
    cd DSP-SLAM && git checkout ed14d02a55401bfacdcce58547f3cb64a6fcbcbd && \
    git submodule update --init --recursive
WORKDIR /DSP-SLAM
COPY patches/dsp_slam.patch /tmp/dsp_slam.patch
RUN git apply /tmp/dsp_slam.patch

# Upstream compiles g2o, DBoW2 and DSP-SLAM with -march=native. Strip it from all
# three together: it ties the binary to the build host, and if only some of them
# keep it, Eigen's alignment differs between g2o and libDSP-SLAM and pose
# optimisation segfaults (EdgeSE3ProjectXYZOnlyPose::linearizeOplus).
RUN sed -i 's/-march=native//g' CMakeLists.txt Thirdparty/g2o/CMakeLists.txt Thirdparty/DBoW2/CMakeLists.txt && \
    cmake -S Thirdparty/g2o -B Thirdparty/g2o/build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build Thirdparty/g2o/build -j"$(nproc)" && \
    cmake -S Thirdparty/DBoW2 -B Thirdparty/DBoW2/build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build Thirdparty/DBoW2/build -j"$(nproc)"

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DPYTHON_EXECUTABLE=/usr/bin/python3 \
        -DPYTHON_INCLUDE_DIRS=/usr/include/python3.10 \
        -DPYTHON_LIBRARIES=/usr/lib/x86_64-linux-gnu/libpython3.10.so && \
    cmake --build build -j"$(nproc)"

COPY config/config_kitti.json /DSP-SLAM/configs/config_kitti.json
COPY scripts/ /DSP-SLAM/scripts/
RUN mkdir -p /DSP-SLAM/map

CMD ["bash"]
