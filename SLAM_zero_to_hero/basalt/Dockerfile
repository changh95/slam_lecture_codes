FROM docker.io/nvidia/opengl:1.0-glvnd-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

# TODO: Add --no-install-recommends to reduce image size (requires going step-by-step)
RUN apt update && \
  apt upgrade -y && \
  apt install -y \
    gcc \
    g++ \
    cmake \
    mold \
    git \
    clang-format-15 \
    cmake-format \
    unzip \
    python3-pip \
    python3-venv \
    libtbb-dev \
    libeigen3-dev \
    libglew-dev \
    ccache \
    libgl1-mesa-dev \
    libjpeg-dev \
    libpng-dev \
    liblz4-dev \
    libbz2-dev \
    libboost-regex-dev \
    libboost-filesystem-dev \
    libboost-date-time-dev \
    libboost-program-options-dev \
    libgtest-dev \
    libopencv-dev \
    libfmt-dev \
    ninja-build \
    libepoxy-dev && \
  apt clean

# Install yaml-cpp from source (Ubuntu 22.04 version lacks yaml-cpp::yaml-cpp target)
RUN git clone https://github.com/jbeder/yaml-cpp.git -b 0.8.0 && \
    cd yaml-cpp && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DYAML_BUILD_SHARED_LIBS=ON -DYAML_CPP_BUILD_TESTS=OFF && \
    make -j$(nproc) && make install && ldconfig && \
    cd / && rm -rf yaml-cpp

RUN apt-get install -y wget sudo

# Bake in the Valve Index "Beat Saber" sequence so the image ships with an
# out-of-the-box demo dataset. Larger datasets (EuRoC, TUM-VI) are mounted at
# run time instead (e.g. -v ~/data/euroc_mav:/datasets:ro).
RUN wget https://huggingface.co/datasets/collabora/monado-slam-datasets/resolve/main/M_monado_datasets/MI_valve_index/MIP_playing/MIPB_beat_saber/MIPB07_beatsaber_fitbeat_expertplus_2.zip && \
    unzip MIPB07_beatsaber_fitbeat_expertplus_2.zip && \
    rm MIPB07_beatsaber_fitbeat_expertplus_2.zip

# Build Basalt from the Monado fork (canonical fork tracked by Monado SLAM datasets),
# pinned to the commit verified in this folder (2026-06-15); submodules follow it.
ARG BASALT_COMMIT=a90a57d7cc1974dfd2ecc9f0345d4166a849548d
RUN git clone https://gitlab.freedesktop.org/mateosss/basalt.git && \
    cd basalt && git checkout ${BASALT_COMMIT} && \
    git submodule update --init --recursive
WORKDIR /basalt
# Downgrade Sophus to 1.22.10 (compatible with CMake 3.22 on this base image)
RUN cd thirdparty/basalt-headers/thirdparty/Sophus && git checkout 1.22.10
# Refresh the package lists first: the apt layers above may be cached from an
# older build, and their pinned .deb versions 404 once Ubuntu publishes updates.
RUN apt-get update && ./scripts/install_deps.sh && apt-get clean
RUN cmake --preset development
RUN cmake --build build --target install -j$(nproc) && ldconfig

WORKDIR /
ENTRYPOINT []
CMD ["bash"]

# Build and run commands: see README.md.
