# SVO Pro (rpg_svo_pro_open) -- semi-direct visual-inertial odometry from UZH RPG.
#
# Upstream tests on Ubuntu 18.04/Melodic and Ubuntu 20.04/Noetic. We take Noetic,
# and use desktop-full because the demo drives rviz for its visualisation.
FROM osrf/ros:noetic-desktop-full

ENV DEBIAN_FRONTEND=noninteractive

# NOTE: no `SHELL ["/bin/bash", "-c"]` here on purpose. Podman's default OCI
# image format ignores SHELL, which would silently run the steps below under
# dash, where `source` does not exist. Every step that needs the ROS
# environment therefore invokes bash explicitly.

# catkin_tools + vcstool drive the workspace; the rest is what upstream's
# install docs ask for (glew/opencv/yaml-cpp) plus the Ceres dependencies
# (blas/lapack/suitesparse), since ceres_catkin builds Ceres from source.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-catkin-tools python3-vcstool python3-osrf-pycommon \
      libglew-dev libopencv-dev libyaml-cpp-dev \
      libblas-dev liblapack-dev libsuitesparse-dev \
      git wget unzip ca-certificates \
      python3-pip \
      autoconf automake libtool libtool-bin pkg-config \
    && rm -rf /var/lib/apt/lists/*
# The autotools line above is not in upstream's dependency list but is required:
# glog_catkin builds glog from source through autoreconf, which fails with
# "Can't exec libtoolize" on a stock ROS image.

# dependencies.yaml lists every repo as git@github.com: (SSH). There are no SSH
# keys in a build container, so rewrite the transport to HTTPS globally.
RUN git config --global url."https://github.com/".insteadOf "git@github.com:"

WORKDIR /svo_ws
RUN catkin config --init --mkdirs --extend /opt/ros/noetic \
      --cmake-args -DCMAKE_BUILD_TYPE=Release -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3

# Pinned to the commit this demo was verified against, so a rebuild months from
# now produces the same image. This is master as of 2022-08-04 -- upstream's
# most recent code commit -- so the pin costs nothing today and protects against
# a surprise upstream change later. Override with
# --build-arg SVO_COMMIT=master to track upstream deliberately.
ARG SVO_COMMIT=ca371f304637e7fb355cf4624d0a02da4e3da220
RUN cd /svo_ws/src \
    && git clone https://github.com/uzh-rpg/rpg_svo_pro_open.git \
    && cd rpg_svo_pro_open && git checkout ${SVO_COMMIT} && cd .. \
    && vcs import < ./rpg_svo_pro_open/dependencies.yaml \
    && touch minkindr/minkindr_python/CATKIN_IGNORE

# Bag-of-words vocabulary for the loop-closing module (svo_online_loopclosing).
RUN cd /svo_ws/src/rpg_svo_pro_open/svo_online_loopclosing/vocabularies \
    && ./download_voc.sh && ls -la

# SvoSetup.cmake ships -Werror. Under GCC 9 (Noetic) upstream's own code trips
# newer -Wall diagnostics that did not exist on GCC 7 (Melodic), which turns
# warnings into a hard build failure. Drop only -Werror; warnings still print,
# and no optimisation or ABI flag is touched.
RUN sed -i 's/ -Werror//' /svo_ws/src/rpg_svo_pro_open/svo_cmake/cmake/Modules/SvoSetup.cmake \
    && grep -n "CMAKE_CXX_FLAGS " /svo_ws/src/rpg_svo_pro_open/svo_cmake/cmake/Modules/SvoSetup.cmake

# Build the front-end + sliding-window backend + loop closure. The global-map
# (iSAM2/GTSAM) variant stays off: svo_global_map keeps its upstream
# CATKIN_IGNORE, matching USE_GLOBAL_MAP FALSE in SvoSetup.cmake.
RUN bash -c 'source /opt/ros/noetic/setup.bash \
    && cd /svo_ws \
    && catkin build --no-status -j"$(nproc)" \
    && ls -la /svo_ws/devel/lib/svo_ros/'

# Headless GUI capture. Xvfb gives rviz a display with nothing else on it and
# ImageMagick's `import` writes it to PNG, so the README's output image can be
# regenerated on a machine with no X server -- and the host desktop can never
# end up in the frame. Also after the compile, to keep that layer cached.
RUN apt-get update && apt-get install -y --no-install-recommends \
      xvfb x11-utils imagemagick \
    && rm -rf /var/lib/apt/lists/*
# x11-utils, not x11-apps: xdpyinfo (the Xvfb readiness probe in
# scripts/capture_rviz.sh) ships in x11-utils.

# rviz otherwise opens a modal "ROS 1 End-of-Life" dialog over the render view.
ENV DISABLE_ROS1_EOL_WARNINGS=1

# evo, for scoring the estimated trajectory against the UZH-FPV ground truth.
# Deliberately after the compile: this layer changes far more often than the
# source does, and putting it above would rebuild all 35 catkin packages.
#
# argcomplete has to be pinned. evo pulls it in unpinned, and current releases
# subscript collections.abc.Iterable, which needs Python 3.9 while Noetic is on
# 3.8 -- without the pin every evo_* entry point dies with "'ABCMeta' object is
# not subscriptable". `packaging` is a real evo dependency that it fails to
# declare. The --help call turns a broken install into a failed build here
# rather than a confusing error at the end of a run.
RUN pip3 install --no-cache-dir evo==1.28.0 "argcomplete==3.1.6" packaging \
    && evo_ape --help >/dev/null

# UZH-FPV assets: calibration, tuned parameters, launch files, and the helper
# that converts SVO's trajectory output for evo. Copied after the build so
# editing a config does not invalidate the compile cache.
COPY config/UZH_FPV_indoor_forward_snapdragon_stereo.yaml \
     config/UZH_FPV_indoor_forward_snapdragon_mono.yaml \
     /svo_ws/src/rpg_svo_pro_open/svo_ros/param/calib/
COPY config/vio_stereo_fpv.yaml config/vio_mono_fpv.yaml \
     /svo_ws/src/rpg_svo_pro_open/svo_ros/param/
COPY launch/fpv_vio_stereo.launch launch/fpv_vio_mono.launch \
     /svo_ws/src/rpg_svo_pro_open/svo_ros/launch/
COPY config/fpv_vio.rviz /svo_ws/src/rpg_svo_pro_open/svo_ros/
COPY scripts/ /svo_ws/scripts/
RUN chmod +x /svo_ws/scripts/*.sh /svo_ws/scripts/*.py

# Source both ROS and the workspace for every interactive shell.
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc \
    && echo "source /svo_ws/devel/setup.bash" >> /root/.bashrc

ENV SVO_WS=/svo_ws
WORKDIR /svo_ws
ENTRYPOINT ["/svo_ws/scripts/entrypoint.sh"]
CMD ["bash"]
