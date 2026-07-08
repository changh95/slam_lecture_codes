# AprilTag Detection + PnP Camera Localization

Code exercise for localizing a camera against fiducial markers on real data: detect **AprilTag** markers (`tagStandard52h13`, 42 mm) in a video of a mock construction site, and estimate the 6-DoF camera pose from the 2D-3D corner correspondences with three different PnP backends — **OpenCV**, **PoseLib**, and **OpenGV**. While they run, the demos **stream the visualization live to a rerun viewer** (camera frustum, trajectory, tag map, detections, error plots); `viz_pnp.py` remains available for offline replay from the JSON output.

Reference project: [changh95/monumental_assessment](https://github.com/changh95/monumental_assessment)

---

## Project Structure

```
part2_ch02_09/
├── README.md
├── CMakeLists.txt
├── Dockerfile
├── viz_pnp.py                  # Offline rerun viewer for the JSON output
├── blueprint.py                # Viewer layout (blueprint) for the live stream
├── data/
│   └── plantage_shed.mp4       # 1920x1080 video with tagStandard52h13 markers
├── include/
│   ├── apriltag_pnp.hpp        # Shared pipeline interface
│   └── rerun_stream.hpp        # Live rerun streaming (rerun C++ SDK)
├── src/
│   ├── apriltag_pnp.cpp        # AprilTag detection, tag map, JSON writer
│   └── rerun_stream.cpp        # Live rerun streaming implementation
└── examples/
    ├── pnp_opencv.cpp          # solvePnPRansac (EPnP) + solvePnPRefineLM
    ├── pnp_poselib.cpp         # PoseLib estimate_absolute_pose (P3P LO-RANSAC)
    └── pnp_opengv.cpp          # OpenGV Kneip P3P RANSAC + optimize_nonlinear
```

All three demos run the **same pipeline** and differ only in the PnP solver:

1. Detect AprilTags in each frame (AprilTag C library).
2. Bootstrap the world frame from **tag 10** (visible throughout the video): world origin = tag 10 center.
3. Estimate the camera pose from all corners of already-mapped tags — the method-specific PnP step.
4. Extend the tag map: unmapped tags with an unambiguous `IPPE_SQUARE` pose are transformed into the world frame.
5. Stream the frame to the rerun viewer (pose, trajectory, tag map, detections, plots) — live gRPC streaming via the rerun C++ SDK.
6. Write `pnp_<method>.json` (camera trajectory, tag map, detections) for `viz_pnp.py`.

Camera intrinsics of the recording (fx = fy = 1153.25, cx = 952.66, cy = 526.55, 14-coefficient rational distortion) are compiled in (`include/apriltag_pnp.hpp`). Units are millimetres.

The dataset arranges tags 2, 3, 39 in an L-shape (center distances: 2-3 = 1090 mm, 3-39 = 1940 mm); each demo prints the mapped distances as a sanity check.

---

## Build

Dependencies:
- **OpenCV 4.x** (`core`, `imgproc`, `imgcodecs`, `videoio`, `calib3d`) and **Eigen3** — required.
- **AprilTag** ([AprilRobotics/apriltag](https://github.com/AprilRobotics/apriltag)) — required (built in the Dockerfile; not part of `slam:base`).
- **PoseLib** and **OpenGV** — optional. `pnp_poselib` / `pnp_opengv` are built only when the respective library is found (both ship in `slam:base`).
- **rerun C++ SDK** ([rerun-io/rerun](https://github.com/rerun-io/rerun)) — optional. Live streaming from the demos is compiled in only when `rerun_sdk` is found (installed in the Dockerfile; without it the demos still run and `viz_pnp.py` covers visualization).
- **rerun-sdk** + **opencv-python-headless** (Python) — for `blueprint.py` and `viz_pnp.py`.

```bash
# Local
mkdir build && cd build
cmake ..
make -j4

# Docker
docker build . -t slam_zero_to_hero:part2_ch02_09
```

---

## Run

Each demo reads `../data/plantage_shed.mp4` by default, **streams the visualization live** to a rerun viewer at `rerun+http://127.0.0.1:9876/proxy` while it runs, and writes `pnp_<method>.json`:

```bash
cd build

# OpenCV PnP (solvePnPRansac + LM refinement)
./pnp_opencv

# PoseLib PnP (P3P LO-RANSAC + non-linear refinement)
./pnp_poselib

# OpenGV PnP (Kneip P3P RANSAC + non-linear refinement)
./pnp_opengv

# Options (same for all three)
./pnp_opencv --video <path> --output <json> --max-frames 300 --verbose
./pnp_opencv --no-stream                                # disable live streaming
./pnp_opencv --stream-url rerun+http://HOST:PORT/proxy  # non-default viewer
./pnp_poselib --no-stream-images                        # skip video/detections
```

All three demos log into the **same rerun recording** (`--stream-recording plantage_shed` by default), so running them one after another overlays the three trajectories in one 3D scene for comparison. Trajectories, tag maps and plot series are logged per method; the camera, video frames and tag detections are identical for every method and live on **one shared entity** (a single camera window in the viewer) — pass `--no-stream-images` to every demo after the first to avoid re-streaming the same frames. Re-running appends to the recording — close the old recording in the viewer (or pass a different `--stream-recording`) for a clean slate.

The C++ SDK cannot author viewer layouts, so `blueprint.py` sends one (run it once, before or after the demos — the Docker default command does it automatically): 3D world + camera view on top, and **one consolidated plot per metric** — reprojection error and PnP solve time, all methods in the same plot — below.

```bash
python3 ../blueprint.py       # push the layout to the viewer at 127.0.0.1:9876
```

### Visualization

Live streaming needs a viewer listening before the demos start:

```bash
rerun &     # on the host; then run the demos
```

In the viewer, select the **plantage_shed** recording and drag the `frame` timeline slider (or press play): the camera frustum with the undistorted video and detected tag outlines moves through the 3D scene, one trajectory per method in its own color (green = OpenCV, blue = PoseLib, red = OpenGV), the mapped tag squares grow in the world frame (origin = tag 10), and the consolidated reprojection-error / solve-time plots update per frame.

`viz_pnp.py` replays the same visualization offline from the JSON output (no need to re-run the pipeline):

```bash
# Spawn a viewer directly (local build)
python3 ../viz_pnp.py pnp_opencv.json pnp_poselib.json pnp_opengv.json

# Stream into an already-running viewer
python3 ../viz_pnp.py --connect pnp_opencv.json pnp_poselib.json pnp_opengv.json

# Headless: save an .rrd file and open it later with `rerun out.rrd`
python3 ../viz_pnp.py --save out.rrd pnp_opencv.json pnp_poselib.json pnp_opengv.json
```

### Docker

**Recommended — the demos stream automatically into a Rerun viewer running on the host:**

```bash
# 1. On the host, open the viewer once:
rerun &

# 2. Run the demos - the default command pushes the viewer layout, then runs
#    all three demos streaming live (video frames streamed once):
docker run -it --rm --network=host slam_zero_to_hero:part2_ch02_09
```

A single method (or any demo options) can be run the same way:

```bash
docker run -it --rm --network=host slam_zero_to_hero:part2_ch02_09 ./pnp_opencv
docker run -it --rm --network=host slam_zero_to_hero:part2_ch02_09 ./pnp_poselib --max-frames 300
```

`--network=host` lets the container reach the viewer at `127.0.0.1:9876`. Live gRPC streaming is version-sensitive: the container's rerun SDKs **must match** your host viewer's version (the Dockerfile pins `0.33.0` — set it to whatever `rerun --version` prints on the host).

**Headless alternative — write an .rrd to the host and open it there:**

```bash
docker run -it --rm -v $PWD/results:/results:z slam_zero_to_hero:part2_ch02_09 bash -c '
    ./pnp_opencv --no-stream --output /results/pnp_opencv.json &&
    ./pnp_poselib --no-stream --output /results/pnp_poselib.json &&
    ./pnp_opengv --no-stream --output /results/pnp_opengv.json &&
    python3 ../viz_pnp.py --save /results/pnp.rrd \
        /results/pnp_opencv.json /results/pnp_poselib.json /results/pnp_opengv.json \
        --video ../data/plantage_shed.mp4'

rerun results/pnp.rrd
```

---

## References

- [OpenCV `calib3d` module](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) (`solvePnPRansac`, `solvePnPRefineLM`, `SOLVEPNP_IPPE_SQUARE`)
- [AprilTag](https://github.com/AprilRobotics/apriltag) (`tagStandard52h13`)
- [PoseLib](https://github.com/PoseLib/PoseLib) (`estimate_absolute_pose`)
- [OpenGV](https://laurentkneip.github.io/opengv/) (`AbsolutePoseSacProblem`, `optimize_nonlinear`)
- [Rerun](https://rerun.io/)
- [changh95/monumental_assessment](https://github.com/changh95/monumental_assessment) — original dataset and assignment
