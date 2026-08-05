# MonoLaneMapping (IROS'23) -- online monocular lane mapping with Catmull-Rom splines.
#
# ROS Noetic is the base because the algorithm's only input is a rosbag of
# openlane_bag/LaneList messages, and lane_ui.py reads it with rosbag.Bag.
# Noetic pins us to Ubuntu 20.04 / Python 3.8, which in turn pins numpy, scipy,
# open3d and gtsam to their last 3.8-compatible releases (see pip block below).
FROM docker.io/library/ros:noetic-ros-base

ARG DEBIAN_FRONTEND=noninteractive

# libgl/libx* are for Open3D's GUI (the map viewer at the end of a run).
# xvfb lets the same viewer render offscreen for --screenshot on a headless box.
# fonts-freefont-ttf is what misc/plot_utils.py:text_3d() hardcodes.
# python3-tk is not optional: lane_feature.py calls matplotlib.use('TkAgg') at
# import time, which an MPLBACKEND env var cannot override.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget ca-certificates \
        build-essential cmake python3-pip python3-dev \
        python3-tk \
        ros-noetic-eigen-conversions \
        libgl1-mesa-glx libgl1-mesa-dri libglu1-mesa libegl1 \
        libx11-6 libxext6 libxrender1 libxinerama1 libxcursor1 \
        libxrandr2 libxi6 libxxf86vm1 libsm6 libgomp1 libusb-1.0-0 \
        xvfb fonts-freefont-ttf && \
    rm -rf /var/lib/apt/lists/*

# Every version here is the newest release that still ships a cp38 wheel.
# numpy is held below 2.0 -- gtsam 4.2 and open3d 0.18 are both built against
# the 1.x C ABI.
#
# jaxlie is deliberately absent: jaxlib has no cp38 wheels left on PyPI, so on
# Noetic's Python 3.8 it is uninstallable. misc/lie_utils.py imports it at module
# load, which alone is fatal, yet the only functions built on it (se3_log,
# se3_exp) are never called -- lane_opt.py imports se3_log and never invokes it.
# scripts/jaxlie_shim supplies a numpy SE3 further down, checked against the real
# package by scripts/verify_jaxlie_shim.py.
#
# rerun-sdk is pinned hard at 0.18.2 for scripts/stream_mapping.py. It is the
# newest release that actually *works* on 3.8: 0.19+ still resolve and install,
# then die on import because rerun_bindings/types.py annotates with `list[float]`,
# which is PEP 585 syntax Python 3.8 cannot subscript. 0.18.2 also predates the
# gRPC rewrite, so it is rr.serve(web_port, ws_port) here, not serve_grpc /
# serve_web_viewer as in the newer rerun_viz/ bridges.
#
# --ignore-installed is load-bearing: ros:noetic-ros-base apt-installs numpy,
# PyYAML, six, python-dateutil and pyparsing with only .egg-info metadata, and
# pip refuses to upgrade a distutils project in place ("Cannot uninstall X").
# The flag makes pip write fresh copies into /usr/local/lib/python3.8/dist-
# packages, which precedes /usr/lib/python3/dist-packages on sys.path, so the
# apt copies are shadowed rather than removed and ROS's own tooling still runs.
RUN python3 -m pip install --no-cache-dir --upgrade "pip<24.1" setuptools wheel && \
    python3 -m pip install --no-cache-dir --ignore-installed \
        numpy==1.24.4 \
        scipy==1.10.1 \
        open3d==0.18.0 \
        gtsam==4.2 \
        opencv-python-headless==4.10.0.84 \
        matplotlib==3.7.5 \
        pillow==10.4.0 \
        pyquaternion==0.9.9 \
        easydict==1.13 \
        tqdm \
        rerun-sdk==0.18.2
# upstream requirements.txt also lists empy/catkin_pkg/rospkg/gnupg/pycryptodomex
# and pyyaml -- ros:noetic-ros-base already apt-installs all of them, and pip
# cannot upgrade the distutils-installed empy in place, so leave them alone.

# openlane_bag defines the LaneList/Lane/LanePoint messages the bags carry;
# msg_utils.py imports them by name, so the package has to be built even though
# rosbag itself could deserialise them from the definitions inside the bag.
ENV CATKIN_WS=/catkin_ws
RUN mkdir -p ${CATKIN_WS}/src && cd ${CATKIN_WS}/src && \
    git clone https://github.com/qiaozhijian/openlane_bag.git && \
    git clone https://github.com/HKUST-Aerial-Robotics/MonoLaneMapping.git && \
    cd ${CATKIN_WS} && \
    . /opt/ros/noetic/setup.sh && \
    catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_BUILD_TYPE=Release

ENV MONOLANE_DIR=${CATKIN_WS}/src/MonoLaneMapping
COPY scripts/jaxlie_shim /opt/jaxlie_shim
ENV PYTHONPATH=${MONOLANE_DIR}:${CATKIN_WS}/devel/lib/python3/dist-packages:/opt/jaxlie_shim:${PYTHONPATH}

COPY scripts/run_mapping.py ${MONOLANE_DIR}/run_mapping.py
COPY scripts/stream_mapping.py ${MONOLANE_DIR}/stream_mapping.py
COPY config/lane_mapping.yaml ${MONOLANE_DIR}/config/lane_mapping_docker.yaml

# Explicitly do NOT force software GL: with the host display passed through,
# Open3D should use the real driver, and under Xvfb Mesa falls back to llvmpipe
# on its own. Set this to 1 if a host's GL stack refuses the offscreen path.
ENV LIBGL_ALWAYS_SOFTWARE=0
ENV PYTHONUNBUFFERED=1

WORKDIR ${MONOLANE_DIR}
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Fail the build, not the first run, if any of the import chain is broken.
RUN /entrypoint.sh python3 -c "\
import rosbag, gtsam, open3d, cv2, jaxlie, numpy, scipy;\
from openlane_bag.msg import LaneList, Lane, LanePoint;\
from lane_slam.system.lane_mapping import LaneMapping;\
print('imports OK: numpy', numpy.__version__, '| open3d', open3d.__version__, '| gtsam', gtsam.__version__ if hasattr(gtsam,'__version__') else '4.2')"
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "run_mapping.py", "--help"]
