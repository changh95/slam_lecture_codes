# ORB-SLAM2

Feature-based visual SLAM: ORB features, keyframe bundle adjustment, DBoW2 place recognition and loop closing. Monocular, stereo, and RGB-D. The default demo here is **RGB-D on TUM RGB-D `freiburg1_desk`**.

- **Repo**: [changh95/Portable_ORB_SLAM2](https://github.com/changh95/Portable_ORB_SLAM2) (pinned to `8131ce6f`): a fork of [raulmur/ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2) that vendors its own OpenCV, Pangolin, DBoW2 and g2o
- **Paper**: [ORB-SLAM2: an Open-Source SLAM System for Monocular, Stereo and RGB-D Cameras](https://arxiv.org/abs/1610.06475), Mur-Artal and Tardós, IEEE T-RO 2017
- **GPU**: not needed for SLAM. It is only used for hardware GL in the viewer.

![ORB-SLAM2 RGB-D on TUM fr1_desk: Pangolin map viewer and current frame](results/tum_fr1_desk/viewer.png)

## Build

```bash
podman build -t slam_zero_to_hero:orb_slam2 .
```

This takes a while because it builds the vendored dependencies (a full OpenCV among them) before ORB-SLAM2 itself. Everything lands in `/Portable_ORB_SLAM2`, the image's `WORKDIR`. The vocabulary is already extracted to `Vocabulary/ORBvoc.txt`, and the TUM association files ship in `Examples/RGB-D/associations/`.

## Data

```bash
python3 ../download_tum_3d.py      # TUM RGB-D -> ~/data/tum_rgbd/
```

This downloads **all 15** TUM RGB-D sequences it knows (16 GB once extracted). It takes no arguments and has no `--list`. The demo needs only `~/data/tum_rgbd/rgbd_dataset_freiburg1_desk/`: 613 RGB frames, 595 depth frames, and 2335 ground-truth poses.

## Run (default: TUM RGB-D fr1_desk, RGB-D)

Run it on your desktop. Two windows open, and a screenshot of both is saved every 2 s:

```bash
mkdir -p results/tum_fr1_desk
podman run --rm \
  --runtime=/usr/bin/nvidia-container-runtime \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
  -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/data/tum_rgbd/rgbd_dataset_freiburg1_desk:/data:ro \
  -v "$(pwd)/scripts":/scripts:ro \
  -v "$(pwd)/results/tum_fr1_desk":/out -w /out \
  slam_zero_to_hero:orb_slam2 \
  /scripts/run_tum_rgbd.sh /data
```

[`scripts/run_tum_rgbd.sh`](scripts/run_tum_rgbd.sh) runs the stock binary:

```
Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM1.yaml /data Examples/RGB-D/associations/fr1_desk.txt
```

It places the two windows side by side and writes `CameraTrajectory.txt`, `KeyFrameTrajectory.txt` (TUM format), `viewer.png` and `shots/`. The windows are:

- **Map Viewer** (Pangolin): black and red map points, blue keyframes, the green covisibility graph, and the current camera.
- **Current Frame** (OpenCV): the image with tracked ORB keypoints and a `KFs / MPs / Matches` status line.

To use another sequence, pass `[settings] [associations]`, e.g. `/scripts/run_tum_rgbd.sh /data /Portable_ORB_SLAM2/Examples/RGB-D/TUM2.yaml /Portable_ORB_SLAM2/Examples/RGB-D/associations/fr2_xyz.txt`, with the matching sequence mounted at `/data`. Use absolute paths, because the working directory is `/out`. Use `TUM1/2/3.yaml` according to fr1/fr2/fr3.

No `xhost` change and no `--net=host` are needed. The two NVIDIA lines give hardware GL, which is how this run was verified. Without them, Mesa software GL is used; that path was verified on KITTI earlier (see NOTES.md) but not re-tested on TUM.

**Without a display**, use `rgbd_tum_headless` (viewer disabled, same SLAM, same output files):

```bash
mkdir -p results/tum_fr1_desk_headless
podman run --rm \
  -v ~/data/tum_rgbd/rgbd_dataset_freiburg1_desk:/data:ro \
  -v "$(pwd)/results/tum_fr1_desk_headless":/out -w /out \
  slam_zero_to_hero:orb_slam2 \
  rgbd_tum_headless /Portable_ORB_SLAM2/Vocabulary/ORBvoc.txt \
    /Portable_ORB_SLAM2/Examples/RGB-D/TUM1.yaml /data \
    /Portable_ORB_SLAM2/Examples/RGB-D/associations/fr1_desk.txt
```

Do not run the stock viewer binaries under Xvfb. Pangolin together with the OpenCV window on Mesa GLX aborts with `CommonMakeCurrent: Assertion 'oldCtxInfo != NULL' failed` (GLX `BadAccess`). On fr1_desk this happened after about 30 s, before the trajectory was written. It happens with or without the screenshot script.

### Accuracy

```bash
python3 scripts/eval_ate.py ~/data/tum_rgbd/rgbd_dataset_freiburg1_desk/groundtruth.txt \
  results/tum_fr1_desk/CameraTrajectory.txt --plot results/tum_fr1_desk/trajectory_vs_gt.png
```

This needs only numpy and matplotlib. It aligns with SE(3) Umeyama, since RGB-D is metric; pass `--sim3` for monocular.

## Results: TUM RGB-D fr1_desk, RGB-D (verified 2026-09-28)

| | GUI run (`run_tum_rgbd.sh`, display `:1`, hardware GL) | headless run (`rgbd_tum_headless`) |
|---|---|---|
| Frames in | 573 associated RGB-D pairs (19.8 s) | 573 |
| Poses out (`CameraTrajectory.txt`) | 434; 64 keyframes | **573**; 121 keyframes |
| Tracking | lost once, 5.7 s in, for 4.97 s (fr1_desk's fast-motion stretch), then relocalized; lost frames are not written | never lost |
| **RMS ATE** (SE(3) aligned) | **1.52 cm** (mean 1.23, median 1.00, max 6.13) | **1.57 cm** (mean 1.25, median 1.02, max 7.27) |
| Path length, estimated vs GT over the same poses | 8.02 vs 7.31 m (includes the straight jump across the gap) | 10.30 vs 9.31 m |
| Tracking time, median / mean | 22.4 / 27.6 ms | 16.4 / 17.1 ms |
| Wall clock, container start to exit | not measured (queued behind a shared GPU lock) | 1 min 23 s |

ORB-SLAM2 is multi-threaded and not deterministic from run to run, which is why the tracking loss showed up in one run and not the other. The binaries sleep to the dataset's 30 Hz timestamps, so wall-clock time does not measure throughput. The ORB-SLAM2 paper reports 1.6 cm on fr1_desk.

The plot below is from the GUI run; the straight segment on the left is the tracking gap.

![Estimated vs ground-truth trajectory, fr1_desk](results/tum_fr1_desk/trajectory_vs_gt.png)

## Supported datasets

| Dataset | Status | Binary | Settings |
|---|---|---|---|
| **TUM RGB-D `freiburg1_desk`** (default) | ✅ RGB-D, ATE 1.52–1.57 cm, viewer verified | `Examples/RGB-D/rgbd_tum`, `rgbd_tum_headless` | `TUM1.yaml` + `associations/fr1_desk.txt` |
| TUM RGB-D, other sequences | ⬜ not run here (15 sequences on disk) | `rgbd_tum`, `Examples/Monocular/mono_tum` | `TUM1/2/3.yaml` by camera; RGB-D needs an associations file |
| **KITTI odometry** seq 00 | ✅ stereo ATE 1.30 m, mono 5.3–6.0 m (headless, 2026-08-05) | `Examples/Stereo/stereo_kitti`, `Examples/Monocular/mono_kitti`, `*_kitti_headless` | `KITTI00-02.yaml`, `KITTI03.yaml`, `KITTI04-12.yaml`, which differ per sequence group |
| **EuRoC MAV** | ⬜ not run here | `Examples/Stereo/stereo_euroc`, `Examples/Monocular/mono_euroc` | `EuRoC.yaml` + the sequence's timestamps file |

On this host only KITTI **sequence 00** has camera images, because the local grey and colour zips are truncated (see [../LIST.md](../LIST.md)). The KITTI runs, the ATE alignment rules, and why the repo's `KITTI00_02_for_stereo.yaml` must not be pasted anywhere are all in [NOTES.md](NOTES.md).
