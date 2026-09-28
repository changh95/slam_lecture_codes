# PIN-SLAM notes

## Image

- The old image was `nvidia/cuda:11.8` + torch cu118. Those wheels carry no sm_120 kernels, so on an RTX 5090 every CUDA op fails. Now: `nvidia/cuda:12.8.1-base-ubuntu22.04` + `torch==2.7.1+cu128` (arch list includes `sm_120`). PIN-SLAM has no custom CUDA code, so the small `base` image is enough — the torch wheel brings its own CUDA runtime and cuDNN.
- Upstream's `requirements.txt` pins `numpy==1.26.4`, `open3d==0.19.0`, `gtsam==4.2`, `evo==1.28.0`, `pypose==0.6.8`; all install cleanly on Python 3.10 next to torch 2.7.1.
- evo's default plot backend is `TkAgg`, and PIN-SLAM plots the trajectories with evo in `write_results()`, **after** SLAM and the ATE printout but **before** the mesh and map are saved. Without `tkinter` that crashes with `ModuleNotFoundError: No module named 'tkinter'`. With `python3-tk` but no X display it crashes with `ImportError: Cannot load backend 'TkAgg' which requires the 'tk' interactive framework, as 'headless' is currently running`. That is what happened to the first full headless run: poses and ATE were written, the mesh was not. The image now runs `evo_config set plot_backend Agg`.
- `error: XDG_RUNTIME_DIR not set in the environment.` at start-up is harmless.

## Data path

Mounting the KITTI root at `/PIN_SLAM/data/kitti` matches the relative paths inside upstream's `config/lidar_slam/run_kitti.yaml` (`./data/kitti/sequences/00/velodyne`, `./data/kitti/poses/00.txt`, `./data/kitti/sequences/00/calib.txt`), so no config of our own is needed. `output_root: ./experiments` is mounted to `results/`. To run another sequence, pass `-i` for the scans and edit `pose_path`/`calib_path`, or use the data loader: `run_kitti.yaml kitti 04 -d -i data/kitti`.

## Visualisation

- `-v` starts the Open3D `gui` app (Filament renderer) in a separate process. Under Xvfb it renders with Mesa llvmpipe; SLAM speed with the GUI on was ~5.4 it/s vs ~9.5 fps headless on the same frames.
- The GUI does not auto-follow by default ("Follow" unchecked), so a static screenshot shows only part of the map; the mesh layer is off by default too. `scripts/render_result.py` gives a deterministic top-down picture of the mesh + trajectories instead.
- `xvfb-run` hangs forever in this rootless-podman container (it waits for Xvfb's ready signal, which never arrives), so both scripts start `Xvfb` themselves.
- With `-v`, `pin_slam.py` never exits on its own after SLAM (it keeps feeding the GUI); `xvfb_capture.sh` kills it once `Reconstructing the global mesh done` appears.

## KITTI download caveat

`../download_kitti.py` takes no arguments (`--help`/`--list` are ignored and it runs in full), extracts into `~/data/kitti_vo_slam/dataset/`, not `extracted/`, and then **deletes every zip** — including ones it failed to extract. During this verification it was started by mistake that way. It deleted the local KITTI zips (velodyne, gray, color, calib, poses) and left a partial `~/data/kitti_vo_slam/dataset/` (velodyne complete for 00–04, 584 of 2761 scans of 05). `~/data/kitti_vo_slam/extracted/` (which this demo uses) was not touched.
