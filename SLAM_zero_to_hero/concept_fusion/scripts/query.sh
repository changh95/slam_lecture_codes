#!/bin/bash
# Run text_query.py under a virtual X server (Open3D's renderer needs one), software GL.
# Not exec: xvfb-run hangs forever when it is PID 1 of the container (it never sees Xvfb's ready signal).
xvfb-run -a -s "-screen 0 1920x1080x24" \
  env __GLX_VENDOR_LIBRARY_NAME=mesa XDG_RUNTIME_DIR=/tmp \
  python /opt/cf_tools/text_query.py "$@"
