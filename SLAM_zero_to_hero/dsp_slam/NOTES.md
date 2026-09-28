# DSP-SLAM notes

## Why the image differs from upstream

Upstream (`build_cuda113.sh`) pins CUDA 11.3, PyTorch 1.10, mmcv-full 1.4.0,
mmdet 2.14, and a fork of mmdetection3d 0.17. None of these ship sm_120
kernels, so on an RTX 5090 the Python side cannot run as upstream built it.

The image instead:

| Part | Upstream | Here | Why |
|---|---|---|---|
| Base | CUDA 11.3, Ubuntu 18.04 | `nvidia/cuda:12.8.1-base-ubuntu22.04` | PyTorch cu128 wheels need glibc >= 2.28 and CUDA >= 12.8 for sm_120 |
| PyTorch | 1.10 (conda, cu113) | 2.7.1 cu128 (pip, system Python 3.10) | Blackwell support |
| OpenCV | 3.4.1 | 3.4.16 | 3.4.1 does not compile with gcc 11; CMake asks for OpenCV 3 |
| Eigen | 3.4.0 source | 3.4.0 (Ubuntu 22.04 package) | same version |
| Pangolin | master at build time | commit `3f4a8b8` (28 Feb 2022) | the master upstream's script cloned when the repo was frozen |
| DSP-SLAM | master | commit `ed14d02` (16 Mar 2022, the last one) | pinned |
| Detectors | MaskRCNN + PointPillars via mmdet/mmdet3d | **offline labels** shipped with KITTI 07 | mmdet3d 0.17 CUDA ops do not build against PyTorch 2.x |

`patches/dsp_slam.patch` is the only change to the DSP-SLAM sources:

- `reconstruct/utils.py`: `skimage.measure.marching_cubes_lewiner` was removed
  in scikit-image 0.19; use `marching_cubes(..., method="lewiner")`.
- `reconstruct/kitti_sequence.py`: `np.bool` was removed in numpy 1.24; use `bool`.
- `dsp_slam.cc`: also write `CameraTrajectory.txt` (KITTI format, every frame)
  next to the map, and skip the final `cv::waitKey(0)` when
  `DSP_SLAM_NO_WAIT=1` is set.

`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` is set in the image: the `.lbl` label
files are pickled numpy arrays, and PyTorch >= 2.6 refuses them under the new
`weights_only=True` default.

## Two things that broke on the way

- **Segfault in `EdgeSE3ProjectXYZOnlyPose::linearizeOplus`** right after
  `New map created`. g2o and DBoW2 had been compiled with upstream's
  `-march=native` (AVX-512 on this host) while libDSP-SLAM was compiled
  without it, so Eigen's fixed-size alignment differed across the library
  boundary. The Dockerfile now strips `-march=native` from all three
  CMakeLists before building any of them.
- **Pangolin drew into a corner under Xvfb.** With no window manager the X
  server never sends the `ConfigureNotify` that Pangolin sizes its views from,
  so the map was drawn in a ~100 px corner and the menu panel was missing.
  `scripts/run_kitti07_headless.sh` resizes the window once with `xdotool`,
  which sends the event, and moves the OpenCV frame window (which opens on top
  at 0,0) below the map. A normal desktop session does not need this.

## Offline vs online detection

`config/config_kitti.json` (copied over `configs/config_kitti.json` in the
image) sets `detect_online: false`, so `reconstruct/__init__.py` never imports
mmdet / mmdet3d. 2D masks come from `labels/maskrcnn_labels/*.lbl` and 3D boxes
from `labels/pointpillars_labels/*.lbl`, one file per frame, exactly what the
authors' MaskRCNN / PointPillars produced. Running the detectors live would
need mmcv-full 1.x + mmdet3d 0.17 rebuilt for PyTorch 2.7 / sm_120, which is
not attempted here. The `maskrcnn` / `pointpillars` weights can still be
fetched with `download_dsp_slam.py --items maskrcnn pointpillars`.

Only KITTI 07 ships labels. Other KITTI sequences (and the `~/data/kitti_vo_slam`
copy, whose grey/colour zips are truncated anyway) would need the online
detectors.

## Data

`download_dsp_slam.py` opens the authors' anonymous SharePoint share link,
which sets a `FedAuth` cookie, and then pulls files through the SharePoint REST
API (`GetFileByServerRelativeUrl(...)/$value`). No login is needed. KITTI 07
unpacks to `image_0..3`, `velodyne`, `labels`, `calib.txt`, `times.txt`
(1101 frames each; `image_3` has one extra stray file).

Ground truth for scoring is the standard KITTI `poses/07.txt`, from
`~/data/kitti_vo_slam/extracted/dataset/poses/07.txt` (1101 poses).

## Viewer and screenshots

Mesa llvmpipe in Xvfb reports OpenGL 4.5 core/compat (Mesa 23.2.1), so the
GLSL 3.30 object shaders compile without `MESA_GL_VERSION_OVERRIDE`. The
viewer's camera is fixed above the first frame (`Viewer.Viewpoint*` in
`KITTI04-12.yaml`); the car leaves that view for most of the sequence and comes
back at the loop closure, so the useful screenshots are the final ones.
`output.gif` is the upstream demo recording kept from the original folder.
