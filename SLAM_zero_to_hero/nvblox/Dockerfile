# nvblox -- NVIDIA's GPU-accelerated TSDF / ESDF volumetric mapping library.
#
# Built standalone (no ROS). nvblox is the *mapping* half of a SLAM system: it
# takes posed depth images and fuses them into a TSDF, a mesh and an ESDF. The
# poses come from outside, which is what the scripts in this image produce from
# the Humanoid Everyday recordings.
#
# CUDA 12.8 is the floor, not a preference: nvblox' own deps image moved to
# 12.8/24.04, and sm_120 (GeForce RTX 50-series / Blackwell) only exists from
# CUDA 12.8 onwards. Ubuntu 24.04 also ships CMake 3.28, comfortably above
# nvblox' 3.22.1 minimum.
FROM docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}compute,utility,graphics

# nvblox pulls Eigen, gflags, glog, stdgpu, SQLite and GoogleTest itself through
# CMake FetchContent, so the apt list is only a toolchain plus the runtime libs
# Open3D wants (libgl1/libegl1/libgomp1, per Open3D's own docker notes).
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates curl wget unzip \
      python3 python3-pip python3-venv \
      libgl1 libegl1 libgomp1 libglib2.0-0t64 \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# Pinned to the release this demo was verified against so a rebuild months from
# now produces the same image. Override with --build-arg NVBLOX_TAG=main.
ARG NVBLOX_TAG=v0.0.10
RUN git clone --depth 1 --branch ${NVBLOX_TAG} \
      https://github.com/nvidia-isaac/nvblox.git /nvblox

# 86 = RTX 30-series, 89 = RTX 40-series, 120 = RTX 50-series (verified on a
# 5090). Left explicit because a container build cannot see the GPU, so nvblox'
# "native" default would resolve to the wrong thing. Trim this list to your own
# card to cut the compile roughly in half. sm_121 (GB10 / DGX Spark) needs a
# newer toolkit than the 12.8 base above -- bump the FROM line to build for it.
ARG CUDA_ARCHS="86;89;120"

# Only the library and the dataset fusers are wanted here. The PyTorch wrapper,
# the Vulkan renderer, the unit tests and the benchmarks together more than
# double the build and none of them are used by this demo.
RUN cmake -S /nvblox -B /nvblox/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
      -DBUILD_TESTING=OFF \
      -DBUILD_BENCHMARKS=OFF \
      -DBUILD_EXPERIMENTS=OFF \
      -DBUILD_PYTORCH_WRAPPER=OFF \
      -DBUILD_RENDERER=OFF \
    && cmake --build /nvblox/build -j"$(nproc)" \
    && cmake --install /nvblox/build \
    && ldconfig \
    && ls -la /nvblox/build/executables/

ENV PATH=/nvblox/build/executables:${PATH}

# Python side: dataset conversion (numpy/Pillow to read the lzma-compressed
# depth and the JPEGs), pose refinement and PLY reading (Open3D), and the
# viewer (rerun). Kept in a venv because Ubuntu 24.04 marks the system
# interpreter externally-managed.
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir \
         numpy pillow scipy tqdm open3d rerun-sdk trimesh
ENV PATH=/opt/venv/bin:${PATH}
ENV VIRTUAL_ENV=/opt/venv

# Scripts last: editing one of these must not invalidate the CUDA compile above.
COPY scripts/ /nvblox_demo/scripts/
COPY config/ /nvblox_demo/config/
RUN chmod +x /nvblox_demo/scripts/*.sh /nvblox_demo/scripts/*.py

ENV NVBLOX_DEMO=/nvblox_demo
WORKDIR /nvblox_demo
CMD ["bash"]
