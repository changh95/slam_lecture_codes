# PLAN: Open-Source SLAM Algorithm Docker Environments

## Overview

Create Docker-based environments for 11 additional open-source SLAM algorithms, following the existing pattern established by `cartographer/`, `dsp_slam/`, and `orb_slam2/` folders. Each folder contains a self-contained Dockerfile that builds and prepares the SLAM system for execution.

Note: ORB-SLAM2, Cartographer, and DSP-SLAM already exist — they are excluded from implementation.

**Estimated total disk usage**: ~60-80 GB for all Docker images (7 GPU projects share CUDA base layers partially).

---

## RALPLAN-DR Summary

### Principles
1. **Self-contained builds**: Each folder must build independently with `docker build`
2. **Official sources**: Always clone from the original/official repository, pinned to a specific tag/commit
3. **Minimal base images**: Use the smallest appropriate base (ubuntu, ros, nvidia/cuda)
4. **Dataset compatibility**: Each project should be runnable with at least one freely downloadable dataset
5. **Consistent structure**: Each folder follows the same pattern: `Dockerfile` + `README.md` + config files

### Decision Drivers
1. **Sensor modality coverage**: Cover visual, LiDAR, visual-inertial, and multi-modal SLAM
2. **Build reproducibility**: Pin versions/commits to avoid breakage from upstream changes
3. **GPU requirement segmentation**: Separate GPU-required projects from CPU-only ones

### Viable Options

**Option A: Independent Dockerfiles per project (Recommended)**
- Each project gets its own base image (ubuntu/ros/cuda as needed)
- Pros: Self-contained, no dependency conflicts, can be built independently
- Cons: Larger total disk usage (~60-80 GB), some redundancy in base packages

**Option B: Layered builds on slam:base**
- Build GPU projects on nvidia/cuda, build ROS projects from slam:base + ROS install
- Pros: Smaller images, shared layers
- Cons: Dependency conflicts between projects, slam:base doesn't have ROS or CUDA
- **Invalidated**: slam:base uses ubuntu:noble without ROS or CUDA — most SLAM projects need one or both

---

## Projects to Implement (11 new folders)

### Tier 1: CPU-only, well-documented (easiest)

#### 1. `kiss_slam/`
- **Repo**: https://github.com/PRBonn/kiss-slam
- **Base image**: `ubuntu:jammy`
- **Dependencies**: Python3, pip, KISS-ICP, MapClosures, g2o, Eigen
- **Sensors**: LiDAR
- **Datasets**: KITTI (already available)
- **Build approach**: `pip install kiss-slam` or build from source
- **Pin**: `v0.0.2`
- **Notes**: Python-first with C++ backend; simplest to set up

#### 2. `basalt_vio/`
- **Repo**: https://gitlab.com/VladyslavUsenko/basalt (GitHub mirror: https://github.com/VladyslavUsenko/basalt)
- **Base image**: `ubuntu:jammy`
- **Dependencies**: CMake ≥3.24, Eigen, fmt, TBB, OpenCV (Basalt bundles its own thirdparty deps)
- **Sensors**: Stereo + IMU, Monocular
- **Datasets**: EuRoC MAV, TUM-VI
- **Build approach**: Use Basalt's bundled thirdparty build system (do NOT use vcpkg — Basalt manages its own dependency versions)
- **Pin**: commit `0f3b2b5` (no releases; clone from GitHub mirror)
- **Notes**: Basalt uses its own optimization backend, not GTSAM. The GitLab repo is canonical; GitHub mirror may lag.

#### 3. `kimera/`
- **Repo**: https://github.com/MIT-SPARK/Kimera-VIO
- **Base image**: `ros:noetic` (ROS1 dependency; note: ROS Noetic reached EOL May 2025 but Docker image remains available)
- **Dependencies**: GTSAM, OpenCV, catkin, glog, gflags
- **Sensors**: Stereo + IMU
- **Datasets**: EuRoC MAV
- **Build approach**: catkin workspace build; has official Docker support
- **Pin**: `v5.0`
- **Notes**: Produces semantic 3D meshes; ROS2 port exists at Kimera-VIO-ROS2 but is less mature

### Tier 2: ROS-dependent, complex builds

#### 4. `fast_lio2/`
- **Repo**: https://github.com/hku-mars/FAST_LIO
- **Base image**: `ros:noetic`
- **Dependencies**: Eigen, PCL, livox_ros_driver, ikd-Tree (bundled)
- **Sensors**: LiDAR + IMU (Velodyne, Ouster, Livox)
- **Datasets**: KITTI (via rosbag conversion), Newer College dataset
- **Build approach**: catkin workspace
- **Pin**: commit `7cc4175` (no releases)
- **Notes**: Efficient tightly-coupled LiDAR-inertial odometry; 100Hz+ capable

#### 5. `fast_livo2/`
- **Repo**: https://github.com/hku-mars/FAST-LIVO2
- **Base image**: `ros:noetic`
- **Dependencies**: Eigen, PCL, OpenCV, livox_ros_driver
- **Sensors**: LiDAR + Camera + IMU
- **Datasets**: FAST-LIVO2 dataset (OneDrive links — may rotate; document manual download as fallback)
- **Build approach**: catkin workspace
- **Pin**: commit `0d2c034` (no releases)
- **Notes**: Multi-modal fusion; dataset availability is a risk — document alternatives

### Tier 3: GPU-required

#### 6. `suma_pp/` ⚠️ HIGH FRAGILITY
- **Repo**: https://github.com/PRBonn/semantic_suma
- **Base image**: `nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04`
- **Dependencies**: RangeNet++ (unmaintained), g2o, PCL, OpenGL, Boost
- **Sensors**: Spinning LiDAR
- **Datasets**: KITTI (already available)
- **Build approach**: CMake + RangeNet++ model download
- **Pin**: commit `531954d`
- **Notes**: GPU required for RangeNet++ semantic segmentation inference. **High fragility risk** — RangeNet++ dependency chain is unmaintained. KISS-SLAM + PIN-SLAM provide alternative LiDAR SLAM coverage if this build proves too brittle.

#### 7. `pin_slam/`
- **Repo**: https://github.com/PRBonn/PIN_SLAM
- **Base image**: `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04`
- **Dependencies**: PyTorch, numpy, open3d, hydra, wandb
- **Sensors**: LiDAR, RGB-D
- **Datasets**: KITTI (already available), Newer College, MulRan
- **Build approach**: conda/pip environment
- **Pin**: `v1.1.1`
- **Notes**: Neural implicit LiDAR SLAM; moderate GPU requirement

#### 8. `glim/`
- **Repo**: https://github.com/koide3/glim
- **Base image**: `nvidia/cuda:12.2.0-devel-ubuntu22.04`
- **Dependencies**: CUDA, Eigen, PCL, OpenCV, GTSAM, Ceres, iridescence (koide3, for visualization)
- **Sensors**: Multi-beam LiDAR, Solid-state LiDAR, Multi-camera, IMU
- **Datasets**: KITTI (already available)
- **Build approach**: CMake with CUDA support
- **Pin**: `v1.2.0`
- **Notes**: GPU-accelerated factor graphs; iridescence library needed for GUI visualization

#### 9. `concept_fusion/` ⚠️ MEDIUM-HIGH FRAGILITY
- **Repo**: https://github.com/concept-fusion/concept-fusion
- **Base image**: `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04`
- **Dependencies**: PyTorch, Segment Anything (SAM ~2.5 GB), OpenCLIP, GradSLAM
- **Sensors**: RGB-D
- **Datasets**: TUM RGB-D, custom RGB-D sequences (NOT ScanNet — requires institutional access + signed terms of use)
- **Build approach**: conda/pip environment + model downloads
- **Pin**: commit `4457c1f`
- **Notes**: Open-set multimodal 3D mapping. **GradSLAM dependency is not well-maintained** — may require pinning or patching. Large model downloads (~5 GB total).

#### 10. `gaussian_splatting_slam/`
- **Repo**: https://github.com/muskie82/MonoGS (CVPR 2024 Best Demo)
- **Base image**: `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04`
- **Dependencies**: PyTorch, diff-gaussian-rasterization, simple-knn, OpenCV
- **Sensors**: Monocular, Stereo, RGB-D
- **Datasets**: TUM RGB-D, Replica, EuRoC
- **Build approach**: conda/pip + custom CUDA extensions
- **Pin**: commit `6c9254c` (no releases)
- **Notes**: MonoGS chosen as the representative Gaussian Splatting SLAM (CVPR'24 Highlight & Best Demo Award)

#### 11. `mast3r_slam/`
- **Repo**: https://github.com/rmurai0610/MASt3R-SLAM
- **Base image**: `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04`
- **Dependencies**: PyTorch, MASt3R checkpoints (Naver Labs), OpenCV
- **Sensors**: Monocular
- **Datasets**: TUM RGB-D, EuRoC, in-the-wild video
- **Build approach**: pip + model checkpoint downloads
- **Pin**: commit `e6f4e3d` (no releases)
- **Notes**: CVPR 2025; real-time 15 FPS monocular dense SLAM

---

## Folder Structure (per project)

```
<project_name>/
├── Dockerfile          # Self-contained build (git clones pinned to tag/commit)
├── README.md           # Instructions: build, run, dataset setup, GPU requirements
└── <config_files>      # Launch files, configs, scripts as needed
```

---

## Dataset Recommendations

### Currently Available
- **KITTI** (LiDAR + stereo camera + GPS/IMU)
  - Works with: KISS-SLAM, GLIM, PIN-SLAM, SuMa++, FAST-LIO2 (via conversion)

### Recommended Downloads

| Dataset | Size | Sensors | Projects | Access |
|---------|------|---------|----------|--------|
| **EuRoC MAV** | ~2.3 GB/seq | Stereo + IMU | Basalt-VIO, Kimera, MonoGS | Free download |
| **TUM RGB-D** | ~0.5-2 GB/seq | RGB-D | MonoGS, MASt3R-SLAM, ConceptFusion | Free download |
| **Newer College** | ~10 GB | LiDAR + IMU | PIN-SLAM, FAST-LIO2 | Free download |
| **TUM-VI** | ~3 GB/seq | Stereo + IMU | Basalt-VIO | Free download |
| **Replica** | ~2 GB | Synthetic RGB-D | MonoGS | Free download |
| **ScanNet** | ~5-10 GB/scene | RGB-D | (ConceptFusion, MASt3R-SLAM) | ⚠️ Requires institutional access + signed terms of use |

### Dataset Download Priority
1. **EuRoC MAV** (High) — covers Basalt-VIO, Kimera, MonoGS
2. **TUM RGB-D** (High) — covers MonoGS, MASt3R-SLAM, ConceptFusion
3. **Newer College** (Medium) — covers PIN-SLAM, FAST-LIO2
4. **Replica** (Low) — synthetic, good for testing Gaussian SLAM

### Dataset Download Script
Add `download_datasets.py` script to download EuRoC and TUM RGB-D datasets alongside the existing `download_kitti.py`.

---

## Implementation Order

### Phase 1: CPU-only projects (fastest to build & verify)
1. `kiss_slam/` — Python-first, KITTI-ready, simplest build
2. `basalt_vio/` — Well-documented C++ build with bundled deps
3. `kimera/` — Has official Docker, ROS-based

### Phase 2: ROS-dependent LiDAR projects
4. `fast_lio2/` — ROS1 catkin build
5. `fast_livo2/` — ROS1 catkin build, similar to FAST-LIO2

### Phase 3: GPU-required projects
6. `pin_slam/` — Python + CUDA, KITTI-ready
7. `glim/` — C++ + CUDA, KITTI-ready
8. `gaussian_splatting_slam/` — Python + custom CUDA extensions
9. `mast3r_slam/` — Python + model checkpoints
10. `concept_fusion/` — Python + large model downloads (fragile)
11. `suma_pp/` — Most fragile; implement last

### Phase 4: Dataset scripts & documentation
12. `download_datasets.py` — EuRoC + TUM RGB-D download script
13. Update main `README.md` with hyperlinks to new project folders

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Build breakage from unpinned repos | High | Medium | Pin specific commits/tags in all Dockerfiles |
| Large CUDA base images (~60-80 GB total) | Certain | Low | Document disk requirements in each README |
| Model download failures | Medium | Medium | Add fallback URLs; document manual download steps |
| ROS1 EOL (noetic = Ubuntu 20.04) | Low | Medium | Use ros:noetic Docker image; note ROS2 alternatives |
| FAST-LIVO2 dataset unavailability | Medium | Low | Document alternative datasets; provide conversion scripts |
| SuMa++ RangeNet++ dependency chain breakage | High | Medium | Pin commit; deprioritize; KISS-SLAM + PIN-SLAM as alternatives |
| ConceptFusion GradSLAM incompatibility | Medium | Medium | Pin commit; patch if needed; document known issues |
| ScanNet access restrictions | Certain | Low | Use TUM RGB-D as primary alternative; note ScanNet requires institutional access |

---

## Acceptance Criteria

1. Each Dockerfile builds successfully with `docker build --no-cache`
2. Each README.md contains: build command, run command, dataset setup, expected output, GPU requirements (if any)
3. No hardcoded paths — all data mounted via Docker volumes
4. GPU projects clearly marked in README with nvidia-docker/CDI requirements
5. At least one project per sensor modality (visual, LiDAR, multi-modal, neural)
6. All `git clone` commands pinned to specific tag or commit SHA
7. Dataset download script works for EuRoC and TUM RGB-D
8. Main README.md updated with hyperlinks to all new project folders
