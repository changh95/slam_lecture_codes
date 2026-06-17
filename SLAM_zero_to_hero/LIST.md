# SLAM Algorithm Dataset Compatibility

## Datasets Already Downloaded (`~/data/`)

| Dataset | Path | Sensors | Used by |
|---|---|---|---|
| **EuRoC MAV** | `~/data/euroc_mav/MH_01_easy/` | stereo + IMU | basalt, kimera, orb_slam2 (mono/stereo) |
| **TUM RGB-D** | `~/data/tum_rgbd/rgbd_dataset_freiburg{1,2,3}_*` | RGB-D | orb_slam2 (rgbd_tum), pin_slam, gaussian_splatting_slam, mast3r_slam |
| **Hilti 2022** | `~/data/hilti_2022/exp14_basement_2.bag` | Hesai PandarXT-32 + Alphasense IMU + 5x cam | fast_lio2, glim, kiss_slam, cartographer (3D) |
| **cow_and_lady** | `~/data/cow_and_lady/` | RGB-D + Vicon | voxblox |

KITTI is **not** present at `~/data/`. The `download_kitti.py` script targets `~/data/kitti_vo_slam/` (≈80 GB for the velodyne dump alone) — pull only the splits a given algorithm needs, not the whole archive.

---

## Algorithms — Dataset Compatibility & Verification Status

Status legend: ✅ verified end-to-end (Docker build → real-data run → captured artifacts), 🟡 build verified, run unconfirmed, ❌ not yet attempted.

| Algorithm | Status | Verified dataset | Other supported datasets |
|-----------|:------:|---|---|
| orb_slam2 | ✅ | TUM RGB-D `freiburg1_xyz` (perf_bench) | KITTI (mono/stereo), EuRoC (mono/stereo) |
| basalt | ✅ | EuRoC MAV `MH_01_easy` (stereo + IMU, EUCM model) | TUM-VI 512x512, Monado SLAM datasets, KITTI (via `basalt_convert_kitti_calib.py`) |
| kiss_slam | ✅ | Hilti 2022 `exp14_basement_2.bag` via rosbag loader | KITTI VO (`--dataloader kitti`), KISS-ICP supports MulRan, nuScenes, NCLT, Apollo, generic dirs of `.bin`/`.pcd`/`.ply` |
| cartographer | ✅ | Hilti 2022 `exp14_basement_2.bag` (3D mode, `hilti_3d.lua`) | KITTI 2D via `kitti2bag` (legacy `velodyne_kitti_2D.lua`), any ROS PointCloud2 + IMU stream |
| fast_lio2 | ✅ | Hilti 2022 `exp14_basement_2.bag` (perf_bench) | NCLT, any Velodyne/Ouster/Livox rosbag |
| glim | ✅ | Hilti 2022 `exp14_basement_2.bag` (perf_bench) | Custom Ouster/Livox/Kinect rosbags |
| kimera | ✅ | EuRoC MAV `MH_01_easy` (perf_bench) | Stereo + IMU only |
| voxblox | ✅ | cow_and_lady (perf_bench) | Generic ROS PointCloud2 / depth |
| pin_slam | 🟡 | — | KITTI, TUM RGB-D, Replica, Newer College, MulRan, nuScenes, KITTI-360, Hilti |
| gaussian_splatting_slam | 🟡 | — | EuRoC, TUM RGB-D, Replica |
| mast3r_slam | 🟡 | — | EuRoC, TUM RGB-D, 7-Scenes, ETH3D SLAM (perf_bench has 1 numeric run on DGX Spark) |
| dsp_slam | 🟡 | — | KITTI sequences via `config_kitti.json` |
| octomap | 🟡 | — | KITTI Velodyne (`benchmark_kitti`) |
| suma_pp | 🟡 | — | KITTI only (RangeNet++ does not generalize) |
| concept_fusion | ❌ | — | ICL, ScanNet (ScanNet requires institutional access) |
| cuvslam | 🟡 | — | NVIDIA cuVSLAM; benchmark in `perf_bench/dgx_spark/cuvslam.json` |
| nvblox | 🟡 | — | Replica, Redwood; benchmark in `perf_bench/dgx_spark/nvblox.json` |
| fast_livo2 | ✅ | Hilti 2022 `exp14_basement_2.bag` (uses bundled `mapping_hesaixt32_hilti22.launch`) | FAST-LIVO2 OneDrive bags (Retail_Street, HKUST_Red_Sculpture, MARS_LVIG, NTU VIRAL) |

---

## Additional datasets that would unlock more verified runs

| Dataset | Size | Sensors | Required by | Access |
|---------|------|---------|-------------|--------|
| **KITTI VO** (color + velodyne + calib + poses) | ~80 GB total | LiDAR + stereo + IMU | dsp_slam, octomap, suma_pp, optional alt for kiss_slam / orb_slam2 | `download_kitti.py` (S3, free) |
| **ICL-NUIM** | ~0.5 GB | Synthetic RGB-D | concept_fusion | [Free](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html) |
| **Replica** | ~2 GB | Synthetic RGB-D | nvblox, gaussian_splatting_slam, pin_slam | [Free](https://github.com/facebookresearch/Replica-Dataset) |
| **FAST-LIVO2 reference bags** | ~6 GB+ each | Livox + cam + IMU | fast_livo2 end-to-end run | [OneDrive](https://github.com/hku-mars/FAST-LIVO2#3-dataset) — use `fast_livo2/download_fast_livo2_dataset.sh` |

### Recommended download priority

1. **KITTI VO** (high) — unlocks dsp_slam, octomap, suma_pp end-to-end + alternate paths for kiss_slam / cartographer / orb_slam2.
2. **FAST-LIVO2 Retail_Street.bag** (medium) — only blocker for `fast_livo2` end-to-end verification; OneDrive links rotate so only download when ready to run.
3. **ICL-NUIM** (medium) — only free dataset concept_fusion supports.
4. **Replica** (low) — synthetic alternative for already-verified RGB-D pipelines.
