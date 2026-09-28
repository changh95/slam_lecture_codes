#!/usr/bin/env bash
# Screenshot the basalt_vio Pangolin GUI with no host X server involved:
# the GUI renders onto a private Xvfb display (Mesa software GL), so the PNG
# contains the visualisation and nothing else.
#
#   capture_gui.sh <out.png> <capture_at_seconds> <basalt_vio args...>
#
# The GUI flags (--show-gui 1) are added for you. The screenshot is taken
# capture_at_seconds after basalt_vio starts; pick a time after the sequence
# has finished so the full trajectory is drawn. Xvfb and ImageMagick are not in
# the image, so they are installed into the (throwaway) container first.
set -euo pipefail

SHOT=${1:?usage: capture_gui.sh <out.png> <capture_at_seconds> <basalt_vio args...>}
AT=${2:?capture_at_seconds}
shift 2

if ! command -v Xvfb >/dev/null || ! command -v import >/dev/null || ! command -v xdotool >/dev/null; then
  apt-get update -qq >/dev/null
  apt-get install -y -qq --no-install-recommends xvfb x11-utils imagemagick xdotool >/dev/null
fi

export DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1
Xvfb :99 -screen 0 1800x1000x24 -nolisten tcp >/dev/null 2>&1 &
XVFB_PID=$!
cleanup() {
  set +e
  [ -n "${VIO_PID:-}" ] && kill "$VIO_PID" 2>/dev/null
  sleep 1
  kill "$XVFB_PID" 2>/dev/null
}
trap cleanup EXIT

for _ in $(seq 1 40); do
  xdpyinfo -display :99 >/dev/null 2>&1 && break
  sleep 0.25
done
xdpyinfo -display :99 >/dev/null 2>&1 || { echo "Xvfb :99 never came up" >&2; exit 1; }

basalt_vio --show-gui 1 "$@" >"${SHOT%.png}.log" 2>&1 &
VIO_PID=$!

# With no window manager on the Xvfb display, Pangolin never gets a
# ConfigureNotify and lays every view out inside a tiny corner of the window.
# Resizing the window once sends one and fixes the layout.
for _ in $(seq 1 120); do
  WID=$(xdotool search --name '^Main$' 2>/dev/null | head -1) && [ -n "$WID" ] && break
  sleep 0.5
done
[ -n "${WID:-}" ] || { echo "Pangolin window 'Main' never appeared" >&2; exit 1; }
sleep 2
xdotool windowsize "$WID" 1799 999
sleep 1
xdotool windowsize "$WID" 1800 1000

echo "capturing at t+${AT}s ..."
sleep "$AT"
kill -0 "$VIO_PID" 2>/dev/null || { echo "basalt_vio exited early, see ${SHOT%.png}.log" >&2; exit 1; }
import -display :99 -window root "$SHOT"
echo "wrote $SHOT"
