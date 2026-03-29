FROM docker.io/ros:noetic

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y \
    build-essential cmake git wget \
    python3-catkin-tools \
    libeigen3-dev libpcl-dev libopencv-dev \
    ros-noetic-pcl-ros ros-noetic-cv-bridge ros-noetic-image-transport \
    ros-noetic-eigen-conversions libusb-1.0-0-dev && \
    apt-get clean

# Install Eigen 3.4.1 (Sophus needs >= 3.4, but noetic ships 3.3.7)
RUN wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.1/eigen-3.4.1.zip && \
    unzip -q eigen-3.4.1.zip && cd eigen-3.4.1 && \
    mkdir build && cd build && cmake .. && make install && \
    cd / && rm -rf eigen-3.4.1 eigen-3.4.1.zip

# Install Sophus v0.9.5 + create .h symlinks (rpg_vikit expects .h not .hpp)
RUN git clone https://github.com/strasdat/Sophus.git -b v0.9.5 && \
    cp -r Sophus/sophus /usr/local/include/sophus && \
    cd /usr/local/include/sophus && \
    for f in *.hpp; do ln -sf "$f" "${f%.hpp}.h"; done && \
    mkdir -p /usr/local/lib/cmake/Sophus && \
    echo 'set(Sophus_INCLUDE_DIRS "/usr/local/include")' > /usr/local/lib/cmake/Sophus/SophusConfig.cmake && \
    echo 'set(Sophus_FOUND TRUE)' >> /usr/local/lib/cmake/Sophus/SophusConfig.cmake && \
    cd / && rm -rf Sophus

# Install livox_ros_driver and vikit_common
RUN mkdir -p /catkin_ws/src && cd /catkin_ws/src && \
    git clone https://github.com/Livox-SDK/livox_ros_driver.git && \
    git clone https://github.com/uzh-rpg/rpg_vikit.git

# Clone FAST-LIVO2 (pinned commit)
RUN cd /catkin_ws/src && \
    git clone https://github.com/hku-mars/FAST-LIVO2.git && \
    cd FAST-LIVO2 && git checkout 0d2c034 && \
    git submodule update --init --recursive

WORKDIR /catkin_ws
ENV CMAKE_PREFIX_PATH=/usr/local:$CMAKE_PREFIX_PATH
RUN . /opt/ros/noetic/setup.sh && \
    catkin_make -j$(nproc) -DCMAKE_PREFIX_PATH="/usr/local;/opt/ros/noetic"

RUN sed -i "6i source \"/catkin_ws/devel/setup.bash\"" /ros_entrypoint.sh

WORKDIR /data

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
