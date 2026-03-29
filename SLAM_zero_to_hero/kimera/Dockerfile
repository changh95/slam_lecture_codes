FROM docker.io/ros:noetic

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y \
    build-essential cmake git wget unzip \
    python3-catkin-tools python3-vcstool \
    libgoogle-glog-dev libgflags-dev libatlas-base-dev \
    libsuitesparse-dev libtbb-dev libboost-all-dev \
    libopencv-dev libgtest-dev && \
    apt-get clean

# Install GTSAM 4.2.0
# Use GTSAM 4.3a1 (Kimera v5.0 needs newer API than 4.2.0)
RUN git clone https://github.com/borglab/gtsam.git -b 4.3a1 && \
    cd gtsam && mkdir build && cd build && \
    cmake .. -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF -DGTSAM_WITH_TBB=OFF && \
    make -j$(nproc) && make install && ldconfig && \
    cd / && rm -rf gtsam

# Install OpenGV
RUN apt-get install -y libeigen3-dev && \
    git clone https://github.com/laurentkneip/opengv.git && \
    cd opengv && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DEIGEN_INCLUDE_DIR=/usr/include/eigen3 && \
    make -j$(nproc) && make install && ldconfig && \
    cd / && rm -rf opengv

# Install DBoW2
RUN git clone https://github.com/dorian3d/DBoW2.git && \
    cd DBoW2 && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && make install && ldconfig && \
    cd / && rm -rf DBoW2

# Build OpenCV 4.5.5 from source (Kimera needs OpenCV 4.x cmake config)
RUN wget -q https://github.com/opencv/opencv/archive/refs/tags/4.5.5.zip && \
    unzip -q 4.5.5.zip && cd opencv-4.5.5 && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF && \
    make -j$(nproc) && make install && ldconfig && \
    cd / && rm -rf opencv-4.5.5 4.5.5.zip

# Install KimeraRPGO (required by Kimera-VIO)
RUN git clone https://github.com/MIT-SPARK/Kimera-RPGO.git && \
    cd Kimera-RPGO && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && make install && ldconfig && \
    cd / && rm -rf Kimera-RPGO

# Build Kimera-VIO (v5.0)
# Patch: Accept OpenCV 4.x (Kimera requests 3.3.1 but OpenCV 4 rejects major version mismatch)
RUN mkdir -p /catkin_ws/src && cd /catkin_ws/src && \
    git clone https://github.com/MIT-SPARK/Kimera-VIO.git -b v5.0 --recursive && \
    sed -i 's/find_package(OpenCV 3.3.1/find_package(OpenCV 4.0.0/' Kimera-VIO/CMakeLists.txt && \
    sed -i 's/GIT_TAG.*master/GIT_TAG main/' Kimera-VIO/cmake/gtest.cmake

WORKDIR /catkin_ws
RUN catkin init && \
    catkin config --extend /opt/ros/noetic --cmake-args -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=/usr/local/lib/cmake/opencv4 -DKIMERA_BUILD_TESTS=OFF && \
    catkin build -j$(nproc)

RUN sed -i "6i source \"/catkin_ws/devel/setup.bash\"" /ros_entrypoint.sh

WORKDIR /data

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
