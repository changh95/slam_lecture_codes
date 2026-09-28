# SuMa++: Efficient LiDAR-based Semantic SLAM (PRBonn, IROS 2019)
# Repo: https://github.com/PRBonn/semantic_suma
#
# SuMa++ = SuMa surfel SLAM + RangeNet++ (DarkNet53) semantic segmentation.
# The segmentation runs inside the visualizer through rangenet_lib + TensorRT.
#
# Upstream targets TensorRT 5 / CUDA 10. TensorRT 5 cannot run on an RTX 50xx
# (Blackwell, sm_120): that needs TensorRT >= 10.8 + CUDA >= 12.8, and the
# TensorRT 10 API dropped everything rangenet_lib used (createNetwork,
# setFp16Mode, buildCudaEngine, enqueue, IPluginFactory, ...). So this image
# uses NGC TensorRT 25.04 (TensorRT 10.9, CUDA 12.9, Ubuntu 24.04) and a
# TensorRT 10 port of rangenet_lib (patches/rangenet_lib/). catkin is not
# needed; patches/CMakeLists.ws.txt builds all three repos with plain CMake.

FROM nvcr.io/nvidia/tensorrt:25.04-py3

ARG DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      build-essential cmake git wget ca-certificates \
      libeigen3-dev libboost-all-dev libyaml-cpp-dev libopencv-core-dev libtbb-dev \
      qtbase5-dev libqt5opengl5-dev libglew-dev libx11-dev libgl-dev \
      libgl1-mesa-dri mesa-utils xvfb x11-apps imagemagick xauth && \
    rm -rf /var/lib/apt/lists/*

# GTSAM 4.2.0 (upstream used the 4.0 PPA, which has no Ubuntu 24.04 build)
RUN git clone --depth 1 --branch 4.2.0 https://github.com/borglab/gtsam.git /tmp/gtsam && \
    cmake -S /tmp/gtsam -B /tmp/gtsam/build -DCMAKE_BUILD_TYPE=Release \
      -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF -DGTSAM_BUILD_UNSTABLE=OFF \
      -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF -DGTSAM_USE_SYSTEM_EIGEN=ON -DGTSAM_WITH_TBB=OFF \
      -DGTSAM_BUILD_PYTHON=OFF && \
    cmake --build /tmp/gtsam/build -j16 && cmake --install /tmp/gtsam/build && \
    ldconfig && rm -rf /tmp/gtsam

# OpenCV's CMake config ships only in libopencv-dev on Ubuntu 24.04
RUN apt-get update -y && apt-get install -y --no-install-recommends libopencv-dev && \
    rm -rf /var/lib/apt/lists/*

# Sources, pinned: semantic_suma 531954d (2024-03-12), rangenet_lib 3fc223e
# (2022-04-24), glow e66d7f8 (the commit upstream's own Dockerfile pins).
WORKDIR /ws
RUN git clone https://github.com/PRBonn/semantic_suma.git && git -C semantic_suma checkout 531954d && \
    git clone https://github.com/PRBonn/rangenet_lib.git && git -C rangenet_lib checkout 3fc223e && \
    git clone https://github.com/jbehley/glow.git && git -C glow checkout e66d7f8

COPY patches /ws/patches
RUN python3 /ws/patches/apply_patches.py /ws && \
    cp /ws/patches/CMakeLists.ws.txt /ws/CMakeLists.txt && \
    cmake -S /ws -B /ws/build -DCMAKE_BUILD_TYPE=Release -DOPENGL_VERSION=430 && \
    cmake --build /ws/build -j16 && \
    test -x /ws/semantic_suma/bin/visualizer

# RangeNet++ DarkNet53 trained on SemanticKITTI (ONNX + cfg). The TensorRT
# engine is GPU-specific, so it is built on first run and cached in
# $RANGENET_ENGINE_DIR (mount a host dir there to keep it between runs).
RUN wget -q https://www.ipb.uni-bonn.de/html/projects/semantic_suma/darknet53.tar.gz -O /tmp/d.tgz && \
    tar -xzf /tmp/d.tgz -C /opt && rm /tmp/d.tgz && ls /opt/darknet53/model.onnx
ENV RANGENET_ENGINE_DIR=/engine_cache
RUN mkdir -p /engine_cache /results

COPY config /ws/config
COPY scripts /ws/scripts
RUN chmod +x /ws/scripts/*.sh

WORKDIR /ws/semantic_suma/bin
CMD ["bash"]
