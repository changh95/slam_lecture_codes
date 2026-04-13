# NICER-SLAM (Docker)

Monocular RGB **neural implicit** SLAM from CVG ETH Zurich (3DV 2024).
Paper: https://arxiv.org/abs/2302.03594
Upstream: https://github.com/cvg/nicer-slam

This image ships the full NICER-SLAM stack (Python 3.8 + PyTorch 1.11 + CUDA
11.3 + Open3D + Trimesh + PyMCubes + Omnidata + GMFlow) with three conda
environments pre-created.

## Requirements

- NVIDIA GPU with **30+ GB VRAM** recommended (per upstream)
- Docker with `--gpus all` support (NVIDIA Container Toolkit)

## How to build

```bash
cd nicer_slam/
docker build -t slam:nicer_slam .
```

This build is slow (~30 min) because it creates three conda envs
(`nicer-slam`, `omnidata`, `gmflow`) and downloads Omnidata pretrained
weights from Zenodo.

## How to run

Download the Replica dataset with the helper script at the project root:

```bash
python3 download_replica.py
# extracts to ~/data/replica/
```

Then run the container:

```bash
docker run -it --rm --gpus all \
  -v ~/data/replica:/data/replica \
  -v $(pwd)/output:/output \
  slam:nicer_slam
```

Inside the container:

```bash
# (env 'nicer-slam' is already active)

# 1) Preprocess a Replica sequence (mono depth + normals + flow)
cd /nicer-slam
bash scripts/run_replica.sh  # or follow preprocess/* scripts

# 2) Train on Replica room0 (NICER-SLAM optimises per-sequence)
cd code
python training/exp_runner.py \
  --conf confs/replica/runconf_replica_2.conf

# 3) Evaluate trajectory
python evaluation/eval_cam.py --output /output
```

Each Replica sequence takes many hours on an A100-class GPU.

## Supported datasets

- **Replica** (use `download_replica.py`) - the default benchmark
- **7-Scenes** - supported via `preprocess/7scenes_*`
- **ScanNet** - supported (requires manual download due to license)
- **Custom RGB** - render your own monocular sequence; a preprocessing
  pipeline is included

## Notes

- CUDA 11.3 / Python 3.8 are pinned by the upstream `env_yamls/` - don't
  upgrade them without patching the repo.
- The hash encoder extension (`code/hashencoder/`) is compiled at **runtime**
  via PyTorch's JIT on the first run, so the container needs `nvcc`
  available (we use the `cuda:11.3.1-devel` base for this).
- If the Omnidata weight download fails during `docker build`, re-download
  them manually into `preprocess/omnidata/pretrained_models/` at runtime.
