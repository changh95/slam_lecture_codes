# Triangulation

Code exercise for triangulating ORB correspondences on a real KITTI stereo pair —
using OpenCV / Eigen, and optionally OpenGV — and visualizing the result
(2D images + 3D point cloud + 3D camera frustums) with Rerun.

---

## Project Structure

```
part2_ch02_07/
├── README.md
├── CMakeLists.txt
├── Dockerfile
├── data/                              # KITTI stereo pair (left.png = cam0, right.png = cam1)
├── viz_triangulation.py               # Rerun viewer for the JSON output
└── examples/
    ├── triangulation_demo.cpp         # OpenCV DLT, custom DLT, mid-point, stereo disparity depth
    └── triangulation_opengv.cpp       # OpenGV linear & mid-point triangulation
```

---

## Build

Dependencies:
- **OpenCV 4.x** (`core`, `imgproc`, `imgcodecs`, `features2d`, `calib3d`) and **Eigen3** — required.
- **OpenGV** — optional. `triangulation_opengv` is built only when OpenGV is found (ships in `slam:base`).
- **rerun-sdk** (Python) — required for visualization. Pre-installed in `slam:base`.

```bash
# Local
mkdir build && cd build
cmake ..
make -j4

# Docker
docker build . -t slam_zero_to_hero:part2_ch02_07
```

---

## Run

Both demos default to the bundled KITTI stereo pair (`data/left.png`,
`data/right.png`) using the KITTI seq 00–02 calibration
(`fx=fy=718.856, cx=607.1928, cy=185.2157`, baseline `~0.5372 m` along `+X`).
Each demo writes a JSON file in the working directory with the inlier
keypoints, camera intrinsics/extrinsics, and one 3D point cloud per
triangulation method.

### Local

```bash
# OpenCV DLT, custom DLT, mid-point, stereo disparity depth
./build/triangulation_demo
# -> writes triangulation_demo.json

# OpenGV linear + mid-point (built only when OpenGV is available)
./build/triangulation_opengv
# -> writes triangulation_opengv.json

# Override the image pair
./build/triangulation_demo /path/to/left.png /path/to/right.png
```

### Visualize with Rerun

The Python viewer reads one or more JSON files and logs the images, ORB
keypoints, 3D point clouds (color-coded per method), and camera frustums.

```bash
# Spawn the native Rerun viewer (requires a display)
python3 ../viz_triangulation.py triangulation_demo.json triangulation_opengv.json

# Stream into an already-running viewer (start it on the host with: rerun &)
python3 ../viz_triangulation.py --connect triangulation_demo.json triangulation_opengv.json
#   --connect uses rerun+http://127.0.0.1:9876/proxy by default
#   override with: --connect-url rerun+http://HOST:PORT/proxy

# Headless: save an .rrd file and open it later with `rerun out.rrd`
python3 ../viz_triangulation.py --save out.rrd triangulation_demo.json triangulation_opengv.json
```

### Docker

**Recommended — stream into a Rerun viewer running on the host:**

```bash
# 1. On the host, open the viewer once:
rerun &

# 2. Run both demos and stream the result into that viewer.
#    Set the rerun-sdk version to your host viewer's (check with: rerun --version).
docker run --rm --network=host slam_zero_to_hero:part2_ch02_07 bash -c '
    pip install -q --break-system-packages rerun-sdk==0.33.0   # == your host viewer
    ./triangulation_demo
    ./triangulation_opengv
    python3 ../viz_triangulation.py --connect \
        triangulation_demo.json triangulation_opengv.json'
```

`--network=host` lets the container reach the viewer at `127.0.0.1:9876`. Live
gRPC streaming is version-sensitive: the container's `rerun-sdk` **must match**
your host viewer's version. `slam:base` ships an older SDK (`0.28.1`), so the
`pip install` above upgrades it — set the version to whatever `rerun --version`
prints on the host.

`--connect` streams to `rerun+http://127.0.0.1:9876/proxy` by default. To target
a different host/port, pass `--connect-url` (this implies `--connect`):

```bash
python3 ../viz_triangulation.py --connect-url rerun+http://HOST:PORT/proxy \
    triangulation_demo.json triangulation_opengv.json
```

**Headless — write an `.rrd` and open it afterwards:**

```bash
docker run --rm -v "$PWD/out:/out" slam_zero_to_hero:part2_ch02_07 bash -c '
    ./triangulation_demo && ./triangulation_opengv
    python3 ../viz_triangulation.py --save /out/triangulation.rrd \
        triangulation_demo.json triangulation_opengv.json'
rerun out/triangulation.rrd
```

**X11 — spawn the viewer inside the container** (needs a display):

```bash
xhost +local:docker
docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
    slam_zero_to_hero:part2_ch02_07
# inside (working dir is build/):
./triangulation_demo && ./triangulation_opengv
python3 ../viz_triangulation.py triangulation_demo.json triangulation_opengv.json
```

---

## References

- [OpenCV `calib3d` module — `triangulatePoints`](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- [OpenGV](https://laurentkneip.github.io/opengv/)
- [Eigen3](https://eigen.tuxfamily.org/dox/)
- [Rerun](https://rerun.io/)
