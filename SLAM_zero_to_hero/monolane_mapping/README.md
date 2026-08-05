# MonoLaneMapping (MonoLaM)

Online lane mapping from a monocular camera. Per-frame 3D lane detections plus odometry go in; a global lane map comes out, with each lane marking stored as a **Catmull-Rom spline** rather than a point cloud. Lanes are associated across frames with Chamfer distance + pose uncertainty + lateral order consistency, and the control points are refined incrementally in a GTSAM factor graph together with the vehicle pose.

- **Repo**: [HKUST-Aerial-Robotics/MonoLaneMapping](https://github.com/HKUST-Aerial-Robotics/MonoLaneMapping)
- **Paper**: [Online Monocular Lane Mapping Using Catmull-Rom Spline](https://arxiv.org/abs/2307.11653) — Qiao, Yu, Yin, Shen, IEEE/RSJ IROS 2023 · [video](https://www.youtube.com/watch?v=9aHNV3TQ6xw)
- Rosbag converter: [qiaozhijian/openlane_bag](https://github.com/qiaozhijian/openlane_bag) (defines the `LaneList` messages)
- Dataset: [OpenLane](https://github.com/OpenDriveLab/OpenLane) · detector: [PersFormer](https://github.com/OpenDriveLab/PersFormer_3DLane)

**What this is not:** the monocular detector is *not* run here. The rosbags carry PersFormer's 3D lane predictions, the ground-truth lanes, and the vehicle pose — no images. This repository is the mapping and optimisation back end, which is what the paper contributes.

## Output

One 20-second OpenLane segment, top-down: 325 m of road, about six physical lane markings across. Grey = raw per-frame detections accumulated in the map frame, coloured line = the fitted Catmull-Rom spline, red spheres = the control points the factor graph actually optimises.

![lane map, top down](docs/lane_map_bev.png)

The blank vertical band is real, not a rendering artifact: detections drop out for ~17 m (78 → 95 m along the drive) and every marking is re-initialised as a fresh landmark on the far side. That is why the saved map holds 14 landmarks for ~6 markings — the association stage recovers tracks across frames but not across a gap this wide.

A 45 m close-up of the same map — the control-point chord is 3 m, and the grey ribbon around each spline is the measurement spread it was fitted through:

![lane map, close up](docs/lane_map_detail.png)

## Build

```bash
podman build -t slam_zero_to_hero:monolane_mapping .
```

Clones MonoLaneMapping and `openlane_bag` into a catkin workspace inside the image and builds the `LaneList`/`Lane`/`LanePoint` messages. ROS Noetic is the base because the algorithm's only input is a rosbag; that pins Python to 3.8, which pins every wheel version in the Dockerfile.

Two things the upstream install notes don't cover, both handled in the image:

- **No jax on Python 3.8.** `jaxlib` no longer ships cp38 wheels, so `jaxlie` is uninstallable. `misc/lie_utils.py` imports it at module load, which is enough to stop the pipeline dead, even though the only two functions that use it (`se3_log`, `se3_exp`) are never actually called on any code path. [`scripts/jaxlie_shim`](scripts/jaxlie_shim/jaxlie/__init__.py) supplies an SE(3) `log`/`exp` in numpy instead. It agrees with the real package to 2e-9 in float64 over 500 random SE(3) elements including the small-angle branch — re-check with [`scripts/verify_jaxlie_shim.py`](scripts/verify_jaxlie_shim.py) on any Python ≥ 3.9 host.
- **`matplotlib.use('TkAgg')` at import time.** `lane_slam/lane_feature.py` forces the Tk backend, which raises outright with no display — the pipeline cannot even be *imported* headless. The entrypoint gives it a throwaway Xvfb display when `DISPLAY` is unset, and uses the real one when you pass it through.
- **`rerun-sdk` is pinned to `0.18.2`.** Newer releases resolve and install on 3.8, then die on import: `rerun_bindings/types.py` annotates with `list[float]`, PEP 585 syntax that 3.8 cannot subscript. 0.18.2 also predates rerun's gRPC rewrite, so it is `rr.serve(web_port=…, ws_port=…)` here rather than the `serve_grpc` / `serve_web_viewer` used by the newer bridges in [`rerun_viz/`](../rerun_viz).

## Download the dataset

The authors provide the OpenLane validation split already converted to rosbags — 202 segments, 433 MB zipped. This is all the demo needs; the original OpenLane image/annotation download is not required.

```bash
mkdir -p ~/data/openlane && cd ~/data/openlane
curl -L -o OpenLane.zip \
  "https://hkustconnect-my.sharepoint.com/:u:/g/personal/zqiaoac_connect_ust_hk/EQxCBwl1Wc5Foq1wNOJ7ZKQBrNik0GK_qa7qEed_zrbGmQ?download=1"
unzip -q OpenLane.zip
```

Gives 630 MB laid out as:

```
~/data/openlane/OpenLane/lane3d_1000/
├── rosbag/     202 x segment-*.bag   (topics: /gt_pose_wc, /lanes_gt, /lanes_predict)
└── test/       scenario split lists (curve, night, intersection, merge/split, up-down, extreme weather)
```

A [Baidu mirror](https://pan.baidu.com/s/1Hrd8ashoiB4_f0B-iz6OHQ?pwd=2023) is in the upstream readme. The image also carries one segment at `examples/data/`, so `run_mapping.py` with no `--bag` works before you download anything.

## Run

Headless, writing the map and both renders:

```bash
podman run --rm \
  -v ~/data/openlane/OpenLane:/data/OpenLane:ro \
  -v "$(pwd)/results":/out \
  slam_zero_to_hero:monolane_mapping \
  python3 run_mapping.py --output_dir /out \
    --bag /data/OpenLane/lane3d_1000/rosbag/segment-14486517341017504003_3406_349_3426_349_with_camera_labels.bag \
    --screenshot /out/lane_map_bev.png --detail_screenshot /out/lane_map_detail.png
```

Drop `--bag` to use the segment bundled in the image. ~15 s for 199 frames.

**With the GUI**, on the host X display — no `xhost` change and no `--net=host` needed:

```bash
podman run --rm -it \
  -e DISPLAY=$DISPLAY -e XDG_RUNTIME_DIR=/tmp/runtime-root \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/data/openlane/OpenLane:/data/OpenLane:ro \
  -v "$(pwd)/results":/out \
  slam_zero_to_hero:monolane_mapping \
  python3 run_mapping.py --output_dir /out --gui
```

An Open3D window titled with the segment name opens once mapping finishes; drag to orbit, scroll to zoom, `q` or `Esc` to close. This is upstream's own `LaneUI.visualize_map()`.

**Watch the map being built**, streamed to a [Rerun](https://rerun.io) viewer frame by frame:

```bash
podman run --rm -it -p 9090:9090 -p 9877:9877 \
  -v ~/data/openlane/OpenLane:/data/OpenLane:ro \
  -v "$(pwd)/results":/out \
  slam_zero_to_hero:monolane_mapping \
  python3 stream_mapping.py --output_dir /out/stream --rate 10 --odo_noise
```

Then open the URL it prints: **`http://localhost:9090/?url=ws://localhost:9877`**. Rendering happens in your browser, so this needs no X11 and no GPU in the container. The viewer stays up after the run so you can scrub back through the timeline; Ctrl-C to stop it.

On the timeline, per frame:

| Entity | What |
|---|---|
| `world/map/lane_NNN` | each landmark's spline + control points, **coloured by landmark id** |
| `world/frame/detections` | this frame's raw detections in world coords, coloured by the landmark they were associated to — **white means not yet associated** |
| `world/vehicle`, `world/traj/{est,odom,gt}` | estimated pose and the three trajectories |
| `plots/{map,timing,pose}` | landmark and control-point counts, per-stage ms, position error vs GT |

Colouring both the map and the live detections by landmark id is what makes the algorithm legible: you can watch association latch a new detection onto an existing lane, and watch the 78→95 m drop-out force a re-initialisation as the colour changes on the far side. Run it **with `--odo_noise`** — that's the only way the three trajectories separate and the pose error plot does anything.

Other sinks: `--rrd FILE` records instead of streaming (~11 MB for one segment, replay with `rerun FILE`), and `--connect HOST:PORT` targets a viewer you're already running.

**All 202 segments**, pose benchmark pooled (upstream's `examples/mapping_bm.py`, minus the per-frame json dump that needs the original OpenLane annotations):

```bash
podman run --rm \
  -v ~/data/openlane/OpenLane:/data/OpenLane:ro -v "$(pwd)/results":/out \
  slam_zero_to_hero:monolane_mapping \
  python3 run_mapping.py --output_dir /out/bm --workers 24 --odo_noise \
    --all_segments /data/OpenLane/lane3d_1000/rosbag
```

### Flags that matter

| Flag | Why |
|---|---|
| `--odo_noise` | **Read this before trusting any pose number.** The bags ship ground-truth poses, so by default the pose estimate is exact and every relative-pose error is `0.000`. This injects 0.5° yaw + 0.5 m xy per frame (upstream's `odom_noise`) so the pose half of the optimiser has something to correct. |
| `--rate` | `stream_mapping.py` only. The bags are 10 Hz, so `--rate 10` is real time and clean odometry sustains it (12.4 fps flat out). `--odo_noise` **cannot** — that pipeline is 161 ms/frame, so it runs at ~5.5 fps and the pacer just lets it fall behind rather than accumulating debt. `--rate 0` runs flat out. |
| `--from_map <map.npy>` | Re-render an existing map without re-running the pipeline. |
| `--eval_pose` | Pose metrics only; skips the map save and json output. |
| `--limit N` | Cap `--all_segments`, for a quick check. |

Upstream's own entry points are all present under `/catkin_ws/src/MonoLaneMapping`. `examples/demo_curve_fitting.py` (the toy spline fit) runs as-is. `examples/mapping_bm.py`, `examples/lane_association.py` and `examples/openlane_eval3d.py` still read `config/lane_mapping.yaml` / `config/lane_association.yaml`, which carry the authors' own absolute paths — pass `--cfg_file config/lane_mapping_docker.yaml` or edit `dataset_dir` to `/data/OpenLane/` first.

## Verified

Everything below was measured with this image on a 32-core host. The pipeline is CPU-only — nothing here touches the GPU except the viewer's OpenGL.

**One segment, `segment-14486517341017504003…`, 199 frames, 307 m** — clean odometry:

| | |
|---|---|
| lane landmarks | 14 (≈6 physical markings, fragmented at the drop-out) |
| saved map | 494 control points |
| vs raw measurements | 42,406 points → **86× fewer** (5.8 kB vs 497 kB as float32) |
| whole pipeline | 75 ms/frame |
| ├ odometry | 0.21 ms |
| ├ lane association | 1.33 ms |
| ├ graph build | 44 ms |
| └ iSAM2 / LM solve | 29 ms |

The 86× is the paper's memory argument, concretely: what gets stored is the spline, not the point cloud it was fitted through.

`stream_mapping.py` on the same segment ends at the same 14 landmarks / 494 control points — the streaming hook wraps `lane_nms` and only reads state, so it cannot perturb the result. Live logging costs ~10 ms/frame on top of the pipeline (12.4 fps vs 13.3 flat out).

**All 202 segments, 27.4 km of driving**, clean odometry: mean **260 control points** per segment map, and relative pose error of exactly `0.000` at every baseline (29,657 pairs at 10 m down to 17,731 at 50 m) — which is the `--odo_noise` caveat above, measured. With ground-truth poses in the bag there is nothing for the pose factors to fix.

Same 202 segments with `--odo_noise`. Relative pose error, optimised vs the corrupted odometry it starts from, pooled over every evaluation pair:

| baseline | rotation (opt / raw) | translation (opt / raw) | pairs |
|---|---|---|---|
| 10 m | **1.262** / 1.608 ° | **2.434** / 2.624 m | 29,657 |
| 20 m | **1.562** / 1.868 ° | **3.164** / 3.346 m | 25,794 |
| 30 m | **1.773** / 1.936 ° | **3.629** / 3.788 m | 22,728 |
| 40 m | **1.896** / 1.912 ° | **4.087** / 4.262 m | 19,988 |
| 50 m | 2.028 / **1.930** ° | **4.541** / 4.684 m | 17,731 |

Read honestly: the lane map does correct the drifting odometry, but modestly — yaw improves 21 % at a 10 m baseline and the gain decays with baseline until at 50 m the optimised rotation is slightly *worse* than raw. Translation improves 3–7 % throughout. That shape is what the geometry predicts: lane markings run parallel to the direction of travel, so they constrain lateral offset and heading well and give almost nothing along-track. Mapping is the strong result here; pose refinement is a bonus.

Note that `--all_segments` re-seeds `numpy` per segment so runs are reproducible, which means every segment sees the same noise realisation. Upstream seeds once in the parent process instead.

### Not verified here

The paper's headline lane-map **F1 / recall / precision** numbers (`examples/openlane_eval3d.py`) need the original OpenLane `lane3d_1000/validation` json annotations for the camera extrinsics per frame. Those are a separate download from OpenDriveLab and are **not** in the rosbag zip above — the zip has `rosbag/` and `test/` only. Everything reported here is pose accuracy and map size, which the rosbags alone support.
