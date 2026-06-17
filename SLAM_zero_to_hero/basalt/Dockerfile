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

# Build Basalt from the Monado fork (canonical fork tracked by Monado SLAM datasets).
RUN git clone --recursive https://gitlab.freedesktop.org/mateosss/basalt.git
WORKDIR /basalt
# Downgrade Sophus to 1.22.10 (compatible with CMake 3.22 on this base image)
RUN cd thirdparty/basalt-headers/thirdparty/Sophus && git checkout 1.22.10
RUN ./scripts/install_deps.sh
RUN cmake --preset development
RUN cmake --build build --target install -j$(nproc) && ldconfig

WORKDIR /
ENTRYPOINT []
CMD ["bash"]

#RUN git clone --recursive https://github.com/changh95/vslam_lecture.git
#WORKDIR vslam_lecture/basalt_monado
#RUN apt-get install -y sudo
#RUN ./scripts/install_deps.sh
#RUN apt-get install -y pluginlib-dev libgpgme-dev
#RUN cmake --preset development
#RUN sudo cmake --build build --target install

# Build Command:
# docker build . -t basalt

# Run Command:
# docker run -it --privileged --net=host --ipc=host -e "DISPLAY=$DISPLAY" -e "QT_X11_NO_MITSHM=1" -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" -e "XAUTHORITY=$XAUTH" --gpus all basalt

#docker run -it --rm \
#    --gpus all \
#    --net=host \
#    --ipc=host \
#    --env DISPLAY=$DISPLAY \
#    --env QT_X11_NO_MITSHM=1 \
#    --env XAUTHORITY=$HOME/.Xauthority \
#    -v /tmp/.X11-unix:/tmp/.X11-unix \
#   -v $HOME/.Xauthority:$HOME/.Xauthority \
#    --privileged \
#    basalt

# Inside docker, run command:

# (Samsung Odyssey Plus)
# basalt_vio --show-gui 1 --dataset-path ../MOO09_short_1_updown/ --dataset-type euroc --cam-calib ./data/msd/msdmo_calib.json --config-path ./data/msd/msdmo_config.json

