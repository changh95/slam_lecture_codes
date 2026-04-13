# Running cuVSLAM with Rerun Visualization

This guide shows how to run cuVSLAM on KITTI sequences inside Docker and view
the trajectory, camera feed, and landmarks live in your browser via Rerun.

## Prerequisites

- NVIDIA GPU + NVIDIA Container Toolkit installed (`docker run --gpus all` works)
- Docker
- KITTI odometry dataset extracted locally (e.g. `~/data/kitti_vo_slam/dataset/sequences/`)

## 1. Build the Docker image

From the `cuvslam/` directory:

```bash
docker build -t slam:cuvslam .
```

This installs the PyCuVSLAM wheel, `rerun-sdk`, and clones the cuVSLAM repo with
its KITTI example scripts into `/cuVSLAM` inside the image.

## 2. Run cuVSLAM with the Rerun web viewer

The cuVSLAM KITTI examples use Rerun for visualization. We'll run the container
with:
- GPU access (`--gpus all`)
- Two ports forwarded: **9090** (web viewer HTTP) and **9876** (Rerun gRPC data channel)
- The KITTI dataset mounted read-only at `/data`
- `PYTHONUNBUFFERED=1` so we see logs immediately

```bash
docker run --rm -it --gpus all \
  -p 9090:9090 -p 9876:9876 \
  -e PYTHONUNBUFFERED=1 \
  -v ~/data/kitti_vo_slam/dataset/sequences:/data \
  slam:cuvslam bash
```

Inside the container:

```bash
# Install the Rerun SDK matching version the examples expect (already done in image)
# Start a Python shell that serves the web viewer + runs the tracker

python3 -u /cuVSLAM/examples/kitti/track_kitti.py
```

The default example uses `rr.init('kitti', spawn=True)` which tries to open a
native Rerun viewer. Inside headless Docker that fails. Use the headless
variant (see next section) OR patch the example to use `serve_grpc` +
`serve_web_viewer`.

### Headless variant (recommended)

Copy the tracker script from `profiling/cuvslam/track_kitti_gui.py` into the
container (or rebuild the image including it) and run:

```bash
python3 -u track_kitti_gui.py /data/00 --web
```

Expected startup logs:

```
Sequence: /data/00
gRPC server URI: rerun+http://127.0.0.1:9876/proxy
=== Web viewer at http://localhost:9090 ===
Processing 4541 frames...
  Frame 0/4541  failed=0
```

## 3. Open the web viewer

Open this URL in your browser on the host machine:

```
http://localhost:9090/?url=rerun+http://localhost:9876/proxy
```

**Important:** The query string `?url=rerun+http://localhost:9876/proxy` is
required. Without it you'll see Rerun's default intro page with no data.
The served HTML does not auto-embed the connect URL, so you have to pass it
explicitly.

You should see:
- **3D view:** KITTI car trajectory, landmark point cloud, car body box
- **2D view:** Left camera image (`cam0`) with feature observations overlaid

## 4. Running on different sequences

```bash
# Small test (500 frames)
python3 -u track_kitti_gui.py /data/00 --web --max-frames 500

# Full sequence 00 (4541 frames, ~60s processing)
python3 -u track_kitti_gui.py /data/00 --web

# Other sequences
python3 -u track_kitti_gui.py /data/02 --web   # 4661 frames
python3 -u track_kitti_gui.py /data/07 --web   # 1101 frames
python3 -u track_kitti_gui.py /data/08 --web   # 4071 frames
```

## 5. Recording instead of live view

Save a `.rrd` file you can open later on the host with `rerun <file>.rrd`:

```bash
python3 -u track_kitti_gui.py /data/00 --rrd /output/cuvslam_kitti00.rrd
```

Mount an output volume to retrieve the file:

```bash
docker run --rm --gpus all \
  -v ~/data/kitti_vo_slam/dataset/sequences:/data \
  -v $(pwd)/results:/output \
  slam:cuvslam \
  python3 -u track_kitti_gui.py /data/00 --rrd /output/cuvslam_kitti00.rrd
```

Then on the host:

```bash
pip install rerun-sdk
rerun results/cuvslam_kitti00.rrd
```

## 6. Troubleshooting

### Browser shows Rerun intro page, no data
You forgot the `?url=...` query parameter. Use the full URL:
`http://localhost:9090/?url=rerun+http://localhost:9876/proxy`

### "Web viewer still running. Press Ctrl+C to exit." then nothing
Tracker finished processing. Data is still in the gRPC server. Hard-refresh
the browser (Ctrl+Shift+R) and re-load the URL above.

### Process stuck, no "Frame N/M" logs
Python stdout is buffered. Start the container with `-e PYTHONUNBUFFERED=1`
and run Python with `-u` flag.

### "GPU functionality will not be available"
You forgot `--gpus all`. cuVSLAM needs the GPU — without it, tracking will
fail or be dramatically slower.

### Port conflict on 9090/9876
Change the host-side port mapping, e.g. `-p 19090:9090 -p 19876:9876`, and
open `http://localhost:19090/?url=rerun+http://localhost:19876/proxy`.
Note the gRPC port in the URL must match the external (host) port.

### KITTI sequence lengths (for reference)

| Seq | Frames | Notes                                   |
|-----|-------:|-----------------------------------------|
| 00  |   4541 | Residential, large loop closures        |
| 01  |   1101 | Highway                                  |
| 02  |   4661 | Largest, residential                    |
| 05  |   2761 | Residential                              |
| 07  |   1101 | Small loop                               |
| 08  |   4071 | Residential                              |
| 09  |   1591 | City                                     |
