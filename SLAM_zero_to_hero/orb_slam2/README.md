# ORB-SLAM2

Feature-based visual SLAM: ORB features, keyframe bundle adjustment, DBoW2 place recognition and loop closing. Monocular, stereo, and RGB-D.

- **Repo**: https://github.com/changh95/Portable_ORB_SLAM2 (a fork that vendors its own OpenCV, Pangolin, DBoW2, g2o)
- **Sensors**: mono / stereo camera, or RGB-D
- **GPU**: not required — the verified runs below use no display at all

## Build

```bash
podman build -t slam_zero_to_hero:orb_slam2 .
```

`buildDeps.py` builds the vendored dependencies (including a full OpenCV) before `build.py` builds ORB-SLAM2 itself, so a cold build takes a while. The image lands everything under **`/Portable_ORB_SLAM2`** (not `/ORB_SLAM2`, as earlier revisions of this file said) and that path is the `WORKDIR`:

| What | Path |
|---|---|
| Vocabulary (already extracted, 145 MB) | `/Portable_ORB_SLAM2/Vocabulary/ORBvoc.txt` |
| KITTI examples | `Examples/Monocular/mono_kitti`, `Examples/Stereo/stereo_kitti` |
| Other examples | `Examples/Monocular/{mono_tum,mono_euroc}`, `Examples/Stereo/stereo_euroc`, `Examples/RGB-D/rgbd_tum` |
| Settings | `Examples/Monocular/KITTI{00-02,03,04-12}.yaml`, `Examples/Stereo/KITTI{00-02,03,04-12}.yaml`, `TUM{1,2,3}.yaml`, `EuRoC.yaml` |

### Headless additions in this image

The stock KITTI examples hard-code `true` for the `System` constructor's `bUseViewer` argument, so they always create a Pangolin GL window **and** an OpenCV highgui window. That is a problem for automated runs: under Xvfb the pair aborts with

```
CommonMakeCurrent: Assertion `oldCtxInfo != NULL' failed.
```

partway through a long sequence — on KITTI 00 it died after ~8 minutes, i.e. after processing thousands of frames but *before* the trajectory is written at shutdown, losing the entire run. So this image adds:

| Addition | Purpose |
|---|---|
| `mono_kitti_headless`, `stereo_kitti_headless` | Same sources with `bUseViewer = false`. No GL, no X, no display. Same SLAM, same output files. **Use these for any unattended run.** |
| `headless <cmd>` | Starts Xvfb, waits for it with `xdpyinfo`, then runs `<cmd>`. Use when you *do* want the stock viewer binaries without a real display. `xvfb-run -a` is deliberately avoided: it intermittently hangs before ever exec'ing its child (observed repeatedly here, once for 30 minutes). |
| `libgl1-mesa-dri` + `LIBGL_ALWAYS_SOFTWARE=1` | `libgl1-mesa-dev` alone ships no DRI driver, so any GL context inside the container fails. This gives llvmpipe software rendering (verified: OpenGL 4.5, direct rendering yes). |

## Verified runs — KITTI odometry sequence 00

Sequence 00 is 4541 frames with large revisits. Both runs below are headless, need no display, and were verified on this host.

### Stereo

```bash
mkdir -p results/stereo
podman run --rm \
  -v ~/data/kitti_vo_slam/extracted/dataset:/data:ro \
  -v "$(pwd)/results/stereo":/out -w /out \
  slam_zero_to_hero:orb_slam2 \
  stereo_kitti_headless \
    /Portable_ORB_SLAM2/Vocabulary/ORBvoc.txt \
    /Portable_ORB_SLAM2/Examples/Stereo/KITTI00-02.yaml \
    /data/sequences/00
```

### Monocular

```bash
mkdir -p results/mono
podman run --rm \
  -v ~/data/kitti_vo_slam/extracted/dataset:/data:ro \
  -v "$(pwd)/results/mono":/out -w /out \
  slam_zero_to_hero:orb_slam2 \
  mono_kitti_headless \
    /Portable_ORB_SLAM2/Vocabulary/ORBvoc.txt \
    /Portable_ORB_SLAM2/Examples/Monocular/KITTI00-02.yaml \
    /data/sequences/00
```

Last verified: Ryzen 9 7950X, 2026-08-05, both under concurrent load (so treat the timings as pessimistic).

| | stereo | monocular |
|---|---|---|
| Frames in | 4541 | 4541 |
| Poses out | 4541 (`CameraTrajectory.txt`, 689,212 B) | 2217 keyframes (`KeyFrameTrajectory.txt`, 186,978 B) |
| Output format | KITTI, 12 floats/row, every frame | TUM, `t tx ty tz qx qy qz qw`, keyframes only |
| Loop closures + global BA | 4 | 4 |
| Median / mean tracking time | 27.4 / 28.7 ms | 14.2 / 15.9 ms |
| **RMS ATE** | **1.297 m** (SE(3) aligned) | **5.3–6.0 m** (Sim(3) aligned) |
| ATE mean / median / max | 1.169 / 1.071 / 3.380 m | 4.552 / 3.977 / 9.883 m |
| Path length | 3705.17 m (GT 3724.19 m) | scale-free |

Both land where the literature puts ORB-SLAM2 on KITTI 00 (~1.3 m stereo).

Monocular is noticeably less repeatable than stereo: two runs produced 2217 and 2024 keyframes with RMS ATE 5.28 m and 6.03 m respectively. The keyframe count itself moves between runs, so quote a range. A useful independent check on the scale recovery: 223.02 m of keyframe path × the recovered Sim(3) scale of 16.624 = 3707.5 m, within **0.45 %** of ground truth's 3724.19 m.

**Do not read wall-clock as throughput for the monocular run.** `mono_kitti.cc` deliberately throttles itself to the dataset timestamps:

```cpp
if(ttrack<T) usleep((T-ttrack)*1e6);   // T = interval to the next frame
```

so its wall clock is pinned near the sequence's own 470 s duration no matter how fast the machine is. Mean tracking time (14–18 ms against a 100 ms frame budget) is the real measure — roughly 5× faster than real time. `stereo_kitti.cc` does the same.

### Computing the ATE — mind the alignment

The alignment must match the sensor, or the number is meaningless:

- **Stereo is metric** → SE(3) alignment, no scale correction.
- **Monocular is scale-free** → Sim(3), *with* scale correction. (For this run the recovered scale factor is ~16, i.e. the raw mono trajectory is in arbitrary units.) An SE(3)-only ATE on a mono trajectory is not a meaningful number.

```bash
D=~/data/kitti_vo_slam/extracted/dataset
python3 -m venv /tmp/evo && /tmp/evo/bin/pip install -q evo

# stereo: both files are KITTI-format in the same camera frame
/tmp/evo/bin/evo_ape kitti $D/poses/00.txt results/stereo/CameraTrajectory.txt -a

# mono: convert GT to TUM first (timestamps from times.txt), then Sim(3)
/tmp/evo/bin/evo_ape tum gt_00_tum.txt results/mono/KeyFrameTrajectory.txt -as
```

Without any alignment the stereo ATE reads 7.363 m RMSE rather than 1.297 m — almost all of that is a rigid offset accumulated in the first frames, which is exactly what the alignment removes.

## About `KITTI00_02_for_stereo.yaml`

Earlier revisions of this README told you to `vim` the in-image stereo settings and paste this file in. **That is unnecessary — and following it would break the run.** The image already ships a working `Examples/Stereo/KITTI00-02.yaml`, and the verified stereo run above uses it unmodified (it loads with `Depth Threshold (Close/Far Points): 18.8008` and tracks the full sequence).

The repo file is a legacy snippet, kept for reference only. Three reasons not to paste it:

1. **Every line is commented out** (`#Camera.fx: …`), so pasting it over the settings file leaves ORB-SLAM2 with no parameters at all.
2. It mixes in **ORB-SLAM3-only keys** (`Camera.bFishEye`, `Stereo.ThDepth`) that ORB-SLAM2 does not read.
3. Its `LEFT.height: 1241` / `LEFT.width: 376` are **swapped** (sequence 00 is 1241 × 376).

Its calibration values themselves are right, and match sequence 00's `calib.txt` exactly — `fx = fy = 718.856`, `cx = 607.1928`, `cy = 185.2157`, `Camera.bf = 386.1448` (= baseline × fx from `P1`). The stock in-image yaml carries the same numbers.

## Watching it run (GUI on your desktop)

The **stock** binaries need no flag — they hard-code `bUseViewer = true`, so they always open both windows. Use them, not the `*_headless` variants:

```bash
podman run --rm -it \
  --runtime=/usr/bin/nvidia-container-runtime \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
  -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/data/kitti_vo_slam/extracted/dataset:/data:ro \
  slam_zero_to_hero:orb_slam2 \
  ./Examples/Stereo/stereo_kitti \
    Vocabulary/ORBvoc.txt Examples/Stereo/KITTI00-02.yaml /data/sequences/00
```

Swap in `./Examples/Monocular/mono_kitti … Examples/Monocular/KITTI00-02.yaml` for monocular. Relative paths work because `WORKDIR` is `/Portable_ORB_SLAM2`.

Two windows appear:

| Window | Size | Shows |
|---|---|---|
| `ORB-SLAM2: Map Viewer` | 1024×768 | Pangolin/GL: sparse map (black = all points, red = local points being tracked), blue keyframe frusta, green current pose and covisibility graph, and a control panel (Follow Camera, Show Points, Show KeyFrames, Show Graph, Localization Mode, Reset) |
| `ORB-SLAM2: Current Frame` | 1241×396 | OpenCV: the KITTI image with green boxes on tracked ORB keypoints, plus a status line like `SLAM MODE \| KFs: 421, MPs: 47288, Matches: 360` |

Three notes on the flags, each verified on this host:

- **No `xhost +local:root`.** Earlier revisions of this file said to run it. It is unnecessary — podman here is rootless, so container root maps to host uid 1000, which the default `SI:localuser:<you>` grant already accepts — and it needlessly opens your X server to every local process.
- **No `--net=host`.** The X connection goes over the bind-mounted unix socket.
- **The `--runtime` and `NVIDIA_*` lines are what get you hardware GL** (`NVIDIA GeForce RTX 5090`, OpenGL 4.6). Drop them and it still works, but through Mesa `llvmpipe` software rendering. Note podman 3.4 predates CDI, so `--device nvidia.com/gpu=all` does *not* work here; the legacy runtime is the way.

For a headless run, use `mono_kitti_headless` / `stereo_kitti_headless` as in the verified runs above, or `headless ./Examples/Stereo/stereo_kitti …` for the stock binary under Xvfb — but see the abort caveat before trusting the latter with a full sequence.

Verifying a window really mapped: use `xwininfo -root -tree | grep ORB-SLAM2`, **not** `-root -children`. The window manager reparents both windows, so `-children` shows nothing and reads as a failed launch.

## Data availability on this host

Only **sequence 00** has camera images: `data_odometry_gray.zip` is a truncated download, and seq 00's `image_0` / `image_1` (4541 frames each) were recovered by walking the zip's local file headers. Other sequences need a fresh download — see the KITTI notes in `../LIST.md`.

## Other datasets

| Dataset | Binary | Settings |
|---|---|---|
| TUM RGB-D (`~/data/tum_rgbd/`) | `Examples/RGB-D/rgbd_tum`, `Examples/Monocular/mono_tum` | `TUM{1,2,3}.yaml` — pick by camera: fr1 → TUM1, fr2 → TUM2, fr3 → TUM3. RGB-D also needs an `associations.txt`. |
| EuRoC (`~/data/euroc_mav/MH_01_easy/`) | `Examples/Monocular/mono_euroc`, `Examples/Stereo/stereo_euroc` | `EuRoC.yaml` + the sequence's `mav0/` timestamps file |
| KITTI 03 / 04–12 | same KITTI binaries | `KITTI03.yaml` / `KITTI04-12.yaml` — the intrinsics differ per sequence group, so do not reuse `KITTI00-02.yaml` |
