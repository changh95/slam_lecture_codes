# MASt3R-SLAM notes

## Default dataset

TUM RGB-D `rgbd_dataset_freiburg1_room` (`~/data/tum_rgbd/`), config
`config/calib.yaml` (calibrated mode, every 2nd frame, multi-process). It is the
sequence upstream's README example uses, it has a loop, and it has ground truth.

## Why the image is not upstream's recipe

Upstream targets PyTorch 2.5.1 with CUDA 11.8/12.1/12.4 and compiles for sm_60–sm_86.
None of those run kernels on an RTX 5090 (sm_120), so the image uses CUDA 12.8.1 +
torch 2.7.1+cu128 and compiles every extension with
`TORCH_CUDA_ARCH_LIST="8.6;8.9;12.0"`. That needed these source patches, all applied
with `sed` in the Dockerfile (each guarded by a `grep` so a silent no-op fails the build):

| File | Problem on torch 2.7 / no-GPU build | Patch |
|---|---|---|
| `setup.py` | gencodes hard-coded up to sm_86 | add `compute_89/sm_89` and `compute_120/sm_120` |
| `setup.py` | `has_cuda = torch.cuda.is_available()` is False inside `podman build`, so `ext_modules` is never defined (`NameError`) | `has_cuda = True` |
| `mast3r_slam/backend/src/matching_kernels.cu`, `thirdparty/mast3r/dust3r/croco/models/curope/kernels.cu` | `AT_DISPATCH_…(x.type(), …)`: `DeprecatedTypeProperties` no longer converts to `ScalarType` | `x.scalar_type()` |
| `mast3r_slam/backend/src/gn_kernels.cu` | `torch::linalg::linalg_norm` is not declared in the extension headers any more | `dx.norm()` (same value: ord=None, dim=None is the flattened 2-norm) |
| `curope/setup.py` | compiles for every arch torch was built with | `all_cuda_archs = []`, so `TORCH_CUDA_ARCH_LIST` decides |
| `pyproject.toml` | `lietorch @ git+…` unpinned | pinned to `e7df865` |

Runtime fixes (no source change):

- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`: torch ≥ 2.6 loads with `weights_only=True`, and the
  MASt3R checkpoint pickles an `argparse.Namespace` (`UnpicklingError: Unsupported global`).
- `libgl-dev libegl-dev`: moderngl's loader `dlopen`s the unversioned `libGL.so`/`libEGL.so`.
- `libusb-1.0-0`: `dataloader.py` imports `pyrealsense2` unconditionally.
- Python 3.11 comes from `uv` (Ubuntu 24.04 ships 3.12; pyimgui needs `Cython<0.30`,
  which is safest on 3.11).
- `--shm-size=16g` on `podman run`: the tracker, backend and viewer processes share
  frames and keyframes through `/dev/shm`.

## Headless viewer

The viewer is a GLFW window (OpenGL 3.3, 4x MSAA). In the container it runs on Xvfb with
Mesa llvmpipe (OpenGL 4.5 core), so no host X server or NVIDIA GL is needed.
`scripts/run_headless_viz.sh` screenshots the Xvfb root window with ImageMagick
`import`. The viewer does not exit when SLAM finishes (`main.py` waits in `viz.join()`),
so the script takes the final screenshot a few seconds after `done` and then kills the
process; the trajectory and `.ply` are already written by then. It sets
`PYTHONUNBUFFERED=1`: with stdout redirected to a file, `done` otherwise stays in
Python's buffer and the script never sees it (the first run here waited 12 minutes on a
finished SLAM run for that reason).

## Evaluation

The trajectory file holds **keyframe** poses only (TUM format, timestamps from
`rgb.txt`). `evo_ape tum groundtruth.txt <seq>.txt -as` aligns with Sim(3), as upstream's
`scripts/eval_tum.sh` does — monocular, so scale is unobservable.
