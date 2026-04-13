FROM ubuntu:jammy
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y \
    build-essential cmake git sudo wget ninja-build \
    software-properties-common python3 python3-pip \
    gcc g++ gdb rsync tar x11-utils x11-apps zip \
    libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev \
    libgl1-mesa-dev libwayland-dev libxkbcommon-dev wayland-protocols \
    libegl1-mesa-dev libc++-dev libglew-dev libeigen3-dev \
    libboost-all-dev libssl-dev && \
    apt-get clean

RUN pip3 install --no-cache-dir numpy pillow rerun-sdk

# Clone Portable_ORB_SLAM2 and build dependencies
# buildDeps.py hardcodes the easy_profiler install path to x86_64-linux-gnu;
# copy it to /usr/local/lib so the linker finds it on any architecture.
RUN git clone https://github.com/changh95/Portable_ORB_SLAM2.git && \
    cd Portable_ORB_SLAM2 && \
    python3 ./buildDeps.py && \
    cp Thirdparty/easy_profiler/install/Release/lib/libeasy_profiler.so /usr/local/lib/ && \
    ldconfig

# Inject our patched mono_kitti variant that streams SLAM state to stdout
COPY mono_kitti_rerun.cc /Portable_ORB_SLAM2/Examples/Monocular/mono_kitti_rerun.cc

# Patch System.h to expose the Map pointer so the example can access all/reference map points
RUN sed -i '/std::vector<cv::KeyPoint> GetTrackedKeyPointsUn();/a\ \ Map* GetMap() { return mpMap; }' /Portable_ORB_SLAM2/include/System.h

# Add a new CMake target for mono_kitti_rerun
RUN cd /Portable_ORB_SLAM2 && \
    printf '\nadd_executable(mono_kitti_rerun Examples/Monocular/mono_kitti_rerun.cc)\ntarget_link_libraries(mono_kitti_rerun ${PROJECT_NAME})\nset_target_properties(mono_kitti_rerun PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)\n' >> CMakeLists.txt && \
    python3 ./build.py

COPY live_rerun.py /app/live_rerun.py

ENV PYTHONUNBUFFERED=1
WORKDIR /data
EXPOSE 9090 9876
ENTRYPOINT ["python3", "-u", "/app/live_rerun.py"]
