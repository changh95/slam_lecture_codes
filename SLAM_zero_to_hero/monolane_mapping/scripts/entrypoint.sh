#!/bin/bash
set -e
source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash

# lane_slam/lane_feature.py runs matplotlib.use('TkAgg') at import time, and
# matplotlib refuses that backend outright when there is no display -- so the
# pipeline cannot even be imported headless. If the caller passed a display
# through (GUI mode) we use it; otherwise we hand the process a throwaway one.
# Open3D's --screenshot renderer draws into the same display.
if [ -z "${DISPLAY}" ]; then
    # NOT `exec`: Xvfb tells xvfb-run it is ready by sending SIGUSR1 to its
    # parent, and as PID 1 xvfb-run never receives it -- the container then
    # hangs forever with Xvfb up and the real command never started. Keeping
    # bash as PID 1 makes xvfb-run an ordinary child and the signal lands.
    xvfb-run -a --server-args="-screen 0 1600x1200x24" "$@"
    exit $?
fi
exec "$@"
