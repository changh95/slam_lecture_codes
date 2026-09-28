#!/usr/bin/env python3
"""Run kiss_slam_pipeline --visualize unattended and save screenshots of the viewer.

The stock RegistrationVisualizer starts paused and waits for [SPACE]. This wrapper
auto-plays it, re-fits the camera, and writes the window to PNG every
CAPTURE_EVERY frames plus once on the last frame. Everything else is the stock
pipeline, so accuracy and output files are identical to a normal run.

    CAPTURE_EVERY=500 CAPTURE_DIR=/out/viewer \
      python3 capture_viewer.py --visualize --dataloader kitti --sequence 00 /data

With no DISPLAY set it starts its own Xvfb (Mesa llvmpipe renders, no GPU needed).
xvfb-run is not used because it hangs when it is PID 1 in a container.
"""
import atexit
import os
import subprocess
import time

if not os.environ.get("DISPLAY"):
    _xvfb = subprocess.Popen(["Xvfb", ":99", "-screen", "0", "1920x1080x24", "-nolisten", "tcp"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    atexit.register(_xvfb.terminate)
    os.environ["DISPLAY"] = ":99"
    time.sleep(2)

import kiss_slam.pipeline as pipeline
from kiss_slam.tools.visualizer import RegistrationVisualizer

EVERY = int(os.environ.get("CAPTURE_EVERY", "500"))
OUT = os.environ.get("CAPTURE_DIR", "viewer")


class CapturingVisualizer(RegistrationVisualizer):
    def __init__(self):
        super().__init__()
        self.frame = 0
        os.makedirs(OUT, exist_ok=True)

    def update(self, slam):
        self._update_geometries(slam)
        self.frame += 1
        if self.frame % EVERY == 0:
            self.capture()
        else:
            self.vis.poll_events()
            self.vis.update_renderer()

    def capture(self):
        self.vis.reset_view_point(True)  # the stock camera never follows
        self.vis.poll_events()
        self.vis.update_renderer()
        path = os.path.join(OUT, f"viewer_{self.frame:06d}.png")
        self.vis.capture_screen_image(path, do_render=True)
        print(f"captured {path}", flush=True)


_run_pipeline = pipeline.SlamPipeline._run_pipeline


def _run_and_capture_last(self):
    _run_pipeline(self)
    if isinstance(self.visualizer, CapturingVisualizer):
        self.visualizer.capture()


pipeline.RegistrationVisualizer = CapturingVisualizer
pipeline.SlamPipeline._run_pipeline = _run_and_capture_last

if __name__ == "__main__":
    from kiss_slam.tools.cli import run

    run()
