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

# The KITTI/TUM example binaries construct the Pangolin viewer unconditionally, so
# they need a display even when you only want the trajectory. xvfb provides a
# virtual one (`xvfb-run -a ./mono_kitti ...`) for headless / CI runs; X11
# forwarding still works when you do want to watch the map being built.
#
# libgl1-mesa-dri is what makes the *headless* Xvfb path possible: libgl1-mesa-dev
# alone ships no DRI driver, so Pangolin dies inside Xvfb with
#   CommonMakeCurrent: Assertion `oldCtxInfo != NULL' failed
# Under Xvfb, Mesa then auto-selects the llvmpipe software rasterizer on its own;
# no LIBGL_ALWAYS_SOFTWARE is needed (verified both with and without it).
#
# Deliberately NOT setting ENV LIBGL_ALWAYS_SOFTWARE=1. It is a Mesa-only knob, so
# once nvidia-container-runtime injects the NVIDIA GLVND vendor library it is
# ignored outright -- but on a host where Mesa can reach /dev/dri it would force
# software GL for the GUI path for no reason. libcanberra-gtk-module only silences
# a GTK warning from the OpenCV frame window.
RUN apt-get install -y xvfb libgl1-mesa-dri mesa-utils libcanberra-gtk-module && apt-get clean

# `xvfb-run -a` intermittently hangs in this image without ever exec'ing its
# child, so ship a deterministic wrapper instead: it starts Xvfb, waits for the
# display with xdpyinfo, then runs the command.
#   podman run ... slam_zero_to_hero:orb_slam2 headless ./mono_kitti ...
COPY headless /usr/local/bin/headless
RUN chmod +x /usr/local/bin/headless

# Truly headless KITTI examples. The stock mono_kitti / stereo_kitti hard-code
# `true` for System's bUseViewer argument, so they always build a Pangolin GL
# window *and* an OpenCV highgui window. Under Xvfb that combination aborts with
#   CommonMakeCurrent: Assertion `oldCtxInfo != NULL' failed
# partway through a long run (reproduced on KITTI 00 after ~8 min), losing the
# trajectory that is only written at shutdown. These variants pass `false`
# instead: no GL, no X, no display of any kind. Same SLAM, same output files.
RUN P=/Portable_ORB_SLAM2 && set -e && \
    for pair in "Monocular/mono_kitti MONOCULAR" "Stereo/stereo_kitti STEREO"; do \
      set -- $pair; src="$P/Examples/$1.cc"; name=$(basename $1); \
      sed "s/ORB_SLAM2::System::$2,true/ORB_SLAM2::System::$2,false/" "$src" > /tmp/${name}_headless.cc; \
      grep -q "ORB_SLAM2::System::$2,false" /tmp/${name}_headless.cc || { echo "viewer-flag patch failed for $name"; exit 1; }; \
      g++ -O3 -std=c++11 -DCOMPILEDWITHC11 -w /tmp/${name}_headless.cc -o /usr/local/bin/${name}_headless \
        -I$P -I$P/include -I/usr/include/eigen3 \
        -I$P/Thirdparty/opencv/install/Release/include \
        -I$P/Thirdparty/pangolin/install/Release/include \
        -L$P/lib -lORB_SLAM2 \
        -L$P/Thirdparty/pangolin/install/Release/lib -lpangolin \
        -L$P/Thirdparty/opencv/install/Release/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc \
        -lGL -lGLU -lGLEW \
        -Wl,-rpath,$P/lib:$P/Thirdparty/opencv/install/Release/lib:$P/Thirdparty/pangolin/install/Release/lib:$P/Thirdparty/DBoW2/lib:$P/Thirdparty/g2o/lib; \
      test "$(ldd /usr/local/bin/${name}_headless | grep -c 'not found')" = "0"; \
    done && rm -f /tmp/*_headless.cc

WORKDIR /Portable_ORB_SLAM2
