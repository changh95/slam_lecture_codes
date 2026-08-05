# ORB-SLAM2

Feature-based visual SLAM: ORB features, keyframe bundle adjustment, DBoW2 place recognition and loop closing. Monocular, stereo, and RGB-D.

- **Repo**: [changh95/Portable_ORB_SLAM2](https://github.com/changh95/Portable_ORB_SLAM2) — a fork of [raulmur/ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2) that vendors its own OpenCV, Pangolin, DBoW2 and g2o
- **Paper**: [ORB-SLAM2: an Open-Source SLAM System for Monocular, Stereo and RGB-D Cameras](https://arxiv.org/abs/1610.06475) — Mur-Artal and Tardós, IEEE T-RO 2017

## Build

```bash
podman build -t slam_zero_to_hero:orb_slam2 .
```

Takes a while: it builds the vendored dependencies (including a full OpenCV) before ORB-SLAM2 itself. Everything lands in `/Portable_ORB_SLAM2`, which is the image `WORKDIR`, and the vocabulary is pre-extracted to `Vocabulary/ORBvoc.txt`.

## Run with the GUI

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

Two windows open: **Map Viewer** (Pangolin — sparse map, keyframe frusta, covisibility graph) and **Current Frame** (the image with tracked ORB keypoints). For monocular, swap in `./Examples/Monocular/mono_kitti` with `Examples/Monocular/KITTI00-02.yaml`.

No `xhost` change and no `--net=host` are needed. The two NVIDIA lines give hardware GL; drop them and it still runs, on software rendering.

Headless instead: use `mono_kitti_headless` / `stereo_kitti_headless`, which write the same trajectory files without opening a window.

## Supported datasets

| Dataset | Binary | Settings |
|---|---|---|
| **KITTI odometry** (verified: seq 00) | `Examples/Stereo/stereo_kitti`, `Examples/Monocular/mono_kitti` | `KITTI00-02.yaml`, `KITTI03.yaml`, `KITTI04-12.yaml` — intrinsics differ per group, don't reuse across them |
| **TUM RGB-D** | `Examples/RGB-D/rgbd_tum`, `Examples/Monocular/mono_tum` | `TUM1/2/3.yaml` by camera (fr1/fr2/fr3); RGB-D also needs an `associations.txt` |
| **EuRoC MAV** | `Examples/Stereo/stereo_euroc`, `Examples/Monocular/mono_euroc` | `EuRoC.yaml` + the sequence's timestamps file |

On this host only KITTI **sequence 00** has camera images — the local grey/colour zips are truncated downloads. See [../LIST.md](../LIST.md).

Verified accuracy on KITTI 00: stereo RMS ATE **1.30 m**, monocular **5.3–6.0 m** (Sim(3)-aligned, as monocular is scale-free). Details, plus the reason the repo's `KITTI00_02_for_stereo.yaml` should not be pasted anywhere, are in [NOTES.md](NOTES.md).
