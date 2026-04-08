# SLAM Algorithm Dataset Compatibility

## Datasets Already Downloaded

- **KITTI** (LiDAR + stereo camera + GPS/IMU)
- **EuRoC MAV** (stereo + IMU)
- **TUM 3D** (RGB-D)

---

## Algorithms Compatible with Downloaded Datasets

| Algorithm | KITTI | EuRoC MAV | TUM 3D | Notes |
|-----------|:-----:|:---------:|:------:|-------|
| orb_slam2 | Yes | Yes | Yes | mono_kitti, stereo_kitti, mono_euroc, stereo_euroc, mono_tum, rgbd_tum |
| basalt | Yes (via converter) | Yes | - | `--dataset-type euroc`; KITTI requires `basalt_convert_kitti_calib.py`; also supports TUM-VI |
| kiss_slam | Yes | - | - | `--dataloader kitti`; KISS-ICP also has loaders for MulRan, nuScenes, NCLT, Apollo, etc. |
| pin_slam | Yes | - | Yes | LiDAR + RGB-D; also supports Replica, Newer College, MulRan, nuScenes, KITTI-360, Hilti, etc. |
| gaussian_splatting_slam | - | Yes | Yes | MonoGS; also supports Replica |
| mast3r_slam | - | Yes | Yes | Monocular dense SLAM; also supports 7-Scenes, ETH3D SLAM |
| cartographer | Yes | - | - | KITTI odometry via kitti2bag ROS bag conversion |
| dsp_slam | Yes | - | - | KITTI sequences with `config_kitti.json` |
| kimera | - | Yes | - | Stereo + IMU; EuRoC MAV recommended |
| octomap | Yes | - | - | `benchmark_kitti` with Velodyne point clouds |
| suma_pp | Yes | - | - | Semantic LiDAR SLAM; only works with KITTI (RangeNet++ does not generalize) |
| fast_lio2 | Yes (via rosbag) | - | - | LiDAR + IMU; also supports NCLT dataset |
| concept_fusion | - | - | - | Supports ICL and ScanNet (not TUM 3D); see below |

**Not compatible with any of the three downloaded datasets:**

| Algorithm | Required Dataset | Notes |
|-----------|-----------------|-------|
| concept_fusion | ICL, ScanNet | ScanNet requires institutional access; ICL is freely available |
| fast_livo2 | FAST-LIVO2 dataset (OneDrive) | LiDAR + Camera + IMU; custom dataset from authors |
| glim | Custom rosbag data | No standard benchmark datasets supported; uses custom Ouster/Livox/Kinect rosbags |
| cuvslam | Not specified | NVIDIA cuVSLAM; generic visual SLAM framework, no dataset configs provided |
| nvblox | Replica, Redwood | NVIDIA real-time 3D reconstruction; used with Replica in benchmarks |
| voxblox | Not specified | Volumetric mapping library; generic ROS point cloud input |

---

## Additional Datasets Required

These datasets are needed to run algorithms that do not support KITTI, EuRoC MAV, or TUM 3D.

| Dataset | Size | Sensors | Required By | Access |
|---------|------|---------|-------------|--------|
| **ICL-NUIM** | ~0.5 GB | Synthetic RGB-D | concept_fusion | [Free download](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html) |
| **Replica** | ~2 GB | Synthetic RGB-D | nvblox, gaussian_splatting_slam, pin_slam | [Free download](https://github.com/facebookresearch/Replica-Dataset) |
| **FAST-LIVO2 dataset** | unknown | LiDAR + Camera + IMU | fast_livo2 | [OneDrive links](https://github.com/hku-mars/FAST-LIVO2) (may rotate; manual download as fallback) |

### Download Priority

1. **ICL-NUIM** (Medium) -- needed to run concept_fusion (only free dataset it supports)
2. **Replica** (Low) -- synthetic; additional option for gaussian_splatting_slam, pin_slam, nvblox
3. **FAST-LIVO2 dataset** (Low) -- only needed for fast_livo2; availability uncertain
