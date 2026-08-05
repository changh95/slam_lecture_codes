#!/usr/bin/env bash
# Source ROS and the SVO workspace, then hand off to whatever was asked for.
#
# The trailing `--` is required, not cosmetic: ROS's setup.bash forwards the
# current positional parameters on to _setup_util.py, so without it the
# container's own command arguments are parsed as ROS options and the shell
# dies with "/tmp/setup.sh.XXXX: line 1: usage:: command not found".
# osrf/ros's own ros_entrypoint.sh does the same thing.
set -e
source /opt/ros/noetic/setup.bash --
source /svo_ws/devel/setup.bash --
exec "$@"
