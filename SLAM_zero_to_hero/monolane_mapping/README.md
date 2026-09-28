# MonoLaneMapping (MonoLaM)

Online lane mapping from a monocular camera. Per-frame 3D lane detections plus odometry go in; a global lane map comes out, with each lane marking stored as a **Catmull-Rom spline** rather than a point cloud. Lanes are associated across frames with Chamfer distance + pose uncertainty + lateral order consistency, and the control points are refined incrementally in a GTSAM factor graph together with the vehicle pose.

- **Repo**: [HKUST-Aerial-Robotics/MonoLaneMapping](https://github.com/HKUST-Aerial-Robotics/MonoLaneMapping)
- **Paper**: [Online Monocular Lane Mapping Using Catmull-Rom Spline](https://arxiv.org/abs/2307.11653) — Qiao, Yu, Yin, Shen, IEEE/RSJ IROS 2023 · [video](https://www.youtube.com/watch?v=9aHNV3TQ6xw)
- Rosbag converter: [qiaozhijian/openlane_bag](https://github.com/qiaozhijian/openlane_bag) (defines the `LaneList` messages)
- Dataset (default): the authors' [OpenLane](https://github.com/OpenDriveLab/OpenLane) validation split converted to rosbags, in `~/data/openlane/` · detector: [PersFormer](https://github.com/OpenDriveLab/PersFormer_3DLane)

**What this is not:** the monocular detector is *not* run here. The rosbags carry PersFormer's 3D lane predictions, the ground-truth lanes, and the vehicle pose — no images. This repository is the mapping and optimisation back end, which is what the paper contributes.

## Output

One 20-second OpenLane segment (`segment-9041488218266405018_6454_030_6474_030`, from the `curve` split), top-down: 308 m of road sweeping through a 44 degree left-hand curve, about seven physical lane markings across. Grey = raw per-frame detections accumulated in the map frame, coloured line = the fitted Catmull-Rom spline, red spheres = the control points the factor graph actually optimises.

![lane map, top down](docs/lane_map_bev.png)

Detections are continuous on this segment, so the two central markings survive the whole drive as single landmarks — 339 m and 329 m, 121 and 117 control points. The remaining five run 131–237 m: the two on the far left only exist over the first 145 m and two others only appear later, which is the road genuinely changing lane count through the curve rather than tracking dropping out. The map also carries two spurious 4-control-point stubs that survived the NMS prune.

A 46 m close-up from three quarters of the way along — the control-point chord is 3 m, and the grey ribbon around each spline is the measurement spread it was fitted through:

![lane map, close up](docs/lane_map_detail.png)

## Build

```bash
podman build -t slam_zero_to_hero:monolane_mapping .
```

Clones MonoLaneMapping and `openlane_bag` (pinned to upstream commits `31f383d` and `e0ae18c`) into a catkin workspace inside the image and builds the `LaneList`/`Lane`/`LanePoint` messages. CPU only, no CUDA.

## Download the dataset

The authors provide the OpenLane validation split already converted to rosbags — 202 segments, 433 MB zipped. This is all the demo needs; the original OpenLane image/annotation download is not required.

```bash
python3 ../download_openlane.py           # 433 MB -> 630 MB in ~/data/openlane/
python3 ../download_openlane.py --list    # what is in the zip, and the scenario splits
```

It is one archive, so there is no per-segment download. Once it is unpacked, `--scenario curve` (or `night`, `updown`, `intersection`, …) prints the bags in OpenLane's own scenario splits, which is how to pick a `--bag`. Note that the `?download=1` share link in the upstream readme now answers 403 without the share page's cookie; the script uses the `_layouts/15/download.aspx?share=…` form of the same file, which also resumes. A [Baidu mirror](https://pan.baidu.com/s/1Hrd8ashoiB4_f0B-iz6OHQ?pwd=2023) is in the upstream readme.

The image also carries one straight segment at `examples/data/`, so the commands below still run with `--bag` dropped before you download anything.

## Run

**Build the map and render the two figures above** (headless, writes to `results/`):

```bash
podman run --rm \
  -v ~/data/openlane/OpenLane:/data/OpenLane:ro \
  -v "$(pwd)/results":/out \
  slam_zero_to_hero:monolane_mapping \
  python3 run_mapping.py --output_dir /out \
    --bag /data/OpenLane/lane3d_1000/rosbag/segment-9041488218266405018_6454_030_6474_030_with_camera_labels.bag \
    --screenshot /out/lane_map_bev.png \
    --detail_screenshot /out/lane_map_detail.png --detail_at 0.78
```

31 s wall-clock for 198 frames: 308.5 m of path, 9 lanes / 541 control points in the saved map, 104x fewer points than the 56 320 raw detections (6.3 kB vs 660 kB), 119 ms/frame (74 ms graph build + 43 ms iSAM2). The bag's poses are ground truth, so without `--odo_noise` the pose RPE is 0 by construction.

**Watch the map being built**, streamed to a [Rerun](https://rerun.io) viewer frame by frame:

```bash
podman run --rm -it -p 9090:9090 -p 9877:9877 \
  -v ~/data/openlane/OpenLane:/data/OpenLane:ro \
  -v "$(pwd)/results":/out \
  slam_zero_to_hero:monolane_mapping \
  python3 stream_mapping.py --output_dir /out/stream --rate 10 --odo_noise \
    --bag /data/OpenLane/lane3d_1000/rosbag/segment-9041488218266405018_6454_030_6474_030_with_camera_labels.bag
```

Then open the URL it prints: **`http://localhost:9090/?url=ws://localhost:9877`**. Rendering happens in your browser, so this needs no X11 and no GPU in the container. The viewer stays up after the run so you can scrub back through the timeline; Ctrl-C to stop it.

![rerun stream](docs/rerun_stream.jpg)

Left: the lane map as it is built (splines coloured by landmark id, red control points, estimated vehicle pose, and the ground-truth / raw-odometry / optimised trajectories). Right: map growth, per-frame cost, and position error against GT for raw odometry (red) and the optimised pose (green). With `--odo_noise` the stream runs at 3.5–4 fps flat out and ends at 20 landmarks / 692 control points (vs 9 / 541 for the noise-free run above). Add `--rrd /out/stream/curve.rrd` to record instead of serving; replay it with a rerun **0.18.2** viewer (`pip install rerun-sdk==0.18.2`), since newer viewers refuse the old `.rrd` format.

## Supported datasets

| Dataset | Status | Notes |
|---|---|---|
| **OpenLane rosbags** (default) — `~/data/openlane/OpenLane/lane3d_1000/rosbag/`, 202 x 20 s segments | ✅ | All 202 bags open, 181–199 frames each (39 981 total), topics `/gt_pose_wc`, `/lanes_gt`, `/lanes_predict`. Verified end-to-end on the curve segment above. |
| In-image sample (`examples/data/segment-14486517341017504003…`) | ✅ | Straight segment bundled with upstream, used when `--bag` is omitted: 199 frames, 307.3 m, 14 lanes / 494 control points, 117 ms/frame. |
| Full OpenLane (images + `lane3d_1000/validation` jsons) | ❌ | Gated (Google Form + Waymo registration). Needed only for the camera panel (`--image_dir`, `--annotation_dir`) and the lane-map F1 evaluation; not exercised. |
