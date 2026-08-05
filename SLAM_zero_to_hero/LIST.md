# SLAM Algorithm Dataset Compatibility

## Datasets Already Downloaded (`~/data/`)

| Dataset | Path | Sensors | Used by |
|---|---|---|---|
| **EuRoC MAV** | `~/data/euroc_mav/MH_01_easy/` | stereo + IMU | basalt, kimera, orb_slam2 (mono/stereo) |
| **TUM RGB-D** | `~/data/tum_rgbd/rgbd_dataset_freiburg{1,2,3}_*` | RGB-D | orb_slam2 (rgbd_tum), pin_slam, gaussian_splatting_slam, mast3r_slam |
| **Hilti 2022** | `~/data/hilti_2022/exp14_basement_2.bag` | Hesai PandarXT-32 + Alphasense IMU + 5x cam | fast_lio2, fast_livo2, kiss_slam, cartographer (3D + IMU) |
| **KITTI odometry** | `~/data/kitti_vo_slam/extracted/dataset/` | stereo grey + Velodyne HDL-64E + GT poses | orb_slam2 (mono/stereo), glim, kiss_slam |
| **cow_and_lady** | `~/data/cow_and_lady/` | RGB-D + Vicon | voxblox |
| **Monado SLAM** | baked into the `basalt` image at `/MIPB07_beatsaber_fitbeat_expertplus_2` | Valve Index stereo + IMU | basalt |
| **FAST-LIVO2-Dataset** | `~/data/fast_livo2/Retail_Street.bag` (**the fast_livo2 demo sequence**) + `calibration.yaml`; `Red_Sculpture.bag` and `CBD_Building_01.bag` also downloaded but unused | Livox Avia + built-in IMU + RGB pinhole cam | fast_livo2 (`download_fast_livo2.py`, 17 more sequences available) |
| **OpenLane (rosbag conversion)** | `~/data/openlane/OpenLane/lane3d_1000/rosbag/` — 202 x 20 s Waymo segments, 630 MB | PersFormer 3D lane detections + GT lanes + vehicle pose (**no images**) | monolane_mapping (`download_openlane.py`, one 433 MB zip holds all 202) |
| **UZH-FPV Drone Racing** | `~/data/uzh_fpv/indoor_forward_3_snapdragon_with_gt.bag` (1.5 GiB) + `calib/` (Kalibr, per environment) | Snapdragon Flight 640x480 stereo **fisheye** @30 Hz + 500 Hz IMU + partial GT (49.5 s of 92 s) | svo_pro_open (`download_uzh_fpv.py`, 28 sequences available; each environment has its **own** calibration) |
| **Humanoid Everyday** | `~/data/humanoid_everyday/<task>/episode_N/` — 2 verified tasks of the 259 (~830 MB of ~500 GB) | Unitree G1 head-mounted RealSense D435: 640×480 RGB + **raw uint16 depth in an lzma buffer, no npy header**, both @30 Hz and **not** mutually aligned (depth ≈79° FOV, colour ≈56°); an unnamed LiDAR @~6.8 k xyz pts; joint state, IMU and legged odometry in `robot_data.jsonl` | nvblox (`download_humanoid_everyday.py`; `--list` for all 259 tasks, `--category loco_manipulation` for the ones that walk) |
| **Cerberus 2.0 Go1** | `~/data/cerberus2/{cmu_garage,mill19_trail,wightman_park,indoor}/` — 12.6 GB of the 33 GB release | Unitree Go1 quadruped: trunk IMU @400 Hz + **4 foot IMUs** (WT901 @200 Hz, gyro in deg/s) + joint encoders @400 Hz + rectified stereo IR @15 Hz. Indoor sequences add Optitrack on `/natnet_ros/Shuo_Go1/pose`; outdoor ones ship iPhone GPS as a MATLAB `timetable` **object** that only MATLAB can read | cerberus_2 (`download_cerberus2.py`, 11 sequences) |

### KITTI: what is actually extracted

`~/data/kitti_vo_slam/` holds the four odometry zips plus an `extracted/` tree. Only part of it is usable:

| Split | State |
|---|---|
| `data_odometry_velodyne.zip` (84 GB) | **intact.** Sequences 00 (4541 scans) and 04 (271) extracted to `extracted/dataset/sequences/NN/velodyne/`. |
| `data_odometry_gray.zip` (22 GB) | **truncated download** — valid PK header, no central directory, so `unzip`/`zipfile` both reject it. Sequence 00's `image_0` and `image_1` (4541 frames each) were recovered by walking local file headers; anything stored after the truncation point is unrecoverable. |
| `data_odometry_color.zip` (1.1 GB) | **truncated download**, far short of the full colour split. Nothing extracted. |
| `data_odometry_poses.zip`, `data_odometry_calib.zip` | intact; `extracted/dataset/poses/{00..10}.txt` + per-sequence `calib.txt` present. |

So stereo/mono ORB-SLAM2 and LiDAR SLAM both work on **sequence 00**, and LiDAR-only additionally on **sequence 04**. Re-download the grey and colour zips before relying on other sequences' images.

---

## Algorithms — Dataset Compatibility & Verification Status

Status legend: ✅ verified end-to-end (Docker build → real-data run → captured artifacts), 🟡 build verified, run unconfirmed, ❌ not yet attempted.

| Algorithm | Status | Verified dataset | Other supported datasets |
|-----------|:------:|---|---|
| orb_slam2 | ✅ | KITTI 00 stereo (4541 frames, RMS ATE 1.30 m SE(3)-aligned) and mono (2217 keyframes, RMS ATE 5.28 m Sim(3)-aligned), both headless; TUM RGB-D `freiburg1_xyz` (perf_bench) | EuRoC (mono/stereo), TUM RGB-D (rgbd/mono), KITTI 03 / 04-12 with their own yaml |
| basalt | ✅ | Monado SLAM Valve Index `MIPB07` (8105 frames, RMS ATE 0.062 m) **and** EuRoC `MH_01_easy` (3682 frames, RMS ATE ≈0.074 m) | TUM-VI 512x512, further Monado SLAM sequences, KITTI (via `basalt_convert_kitti_calib.py`) |
| kiss_slam | ✅ | KITTI 00 (4541 scans, ATE 5.59 m / 0.58 %, 7 loop closures) and 04 (271 scans, ATE 0.59 m); Hilti `exp14_basement_2.bag` **only with `config/hilti_indoor.yaml`** — stock defaults diverge, see its README | MulRan, nuScenes, NCLT, Apollo, TUM, mcap, generic dirs of `.bin`/`.pcd`/`.ply` (14 loaders) |
| cartographer | ✅ | Hilti 2022 `exp14_basement_2.bag` in **3D + IMU** mode (`config/hilti_3d_lio.lua`): 730 poses, 38.38 m path, 4.19 m z extent, **0.084 m RMSE against FAST-LIO2**, 8 submaps / 26 loop constraints, 21.3 M-point 3D map. The old 2D config is kept for contrast — it cannot map a 4.19 m level change. | any ROS PointCloud2 + IMU stream. Legacy KITTI 2D launch needs a `kitti2bag` bag and cannot run headless (`rviz` is `required="true"`) |
| fast_lio2 | ✅ | Hilti 2022 `exp14_basement_2.bag` end-to-end with `config/hilti_pandarxt32.yaml` (737 poses, 37.93 m path, 6.05 ms/scan) | NCLT, any Velodyne/Ouster/Livox rosbag with an IMU stream |
| glim | ✅ | KITTI 04 (ATE 2.60 m over 394 m) and KITTI 00 (4531 poses, ATE 10.2–11.5 m over 3.7 km) via the `glim_kitti` driver added in `glim/` — upstream ships **no executable** | LiDAR+IMU rosbags, but only through `glim_ros1`/`glim_ros2`, which are not in this image |
| kimera | ✅ | EuRoC MAV `MH_01_easy` (perf_bench) | Stereo + IMU only |
| svo_pro_open | ✅ | UZH-FPV `indoor_forward_3` Snapdragon stereo fisheye + IMU: **RMS ATE 0.43 ± 0.04 m over 278 m (~0.16 %)** SE(3)-aligned, mean of 7 runs (range 0.376–0.476 m — real-time replay is not deterministic; headless averages 0.427 m, with the rviz GUI open 0.476 m), 2551 poses, all 92 s tracked with zero losses; monocular 0.156 m Sim(3) (2 runs, 0.132–0.180 m) with scale solved to 0.99941 | UZH-FPV `indoor_45` / `outdoor_*` (each needs its own calibration) and mDAVIS; EuRoC via upstream's `euroc_vio_{mono,stereo}.launch`; FLA stereo+IMU (frontend only) |
| voxblox | ✅ | cow_and_lady (perf_bench) | Generic ROS PointCloud2 / depth |
| pin_slam | 🟡 | — | KITTI, TUM RGB-D, Replica, Newer College, MulRan, nuScenes, KITTI-360, Hilti |
| gaussian_splatting_slam | 🟡 | — | EuRoC, TUM RGB-D, Replica |
| mast3r_slam | 🟡 | — | EuRoC, TUM RGB-D, 7-Scenes, ETH3D SLAM (perf_bench has 1 numeric run on DGX Spark) |
| dsp_slam | 🟡 | — | KITTI sequences via `config_kitti.json` |
| octomap | 🟡 | — | KITTI Velodyne (`benchmark_kitti`) |
| suma_pp | 🟡 | — | KITTI only (RangeNet++ does not generalize) |
| concept_fusion | ❌ | — | ICL, ScanNet (ScanNet requires institutional access) |
| cuvslam | 🟡 | — | NVIDIA cuVSLAM; benchmark in `perf_bench/dgx_spark/cuvslam.json` |
| nvblox | ✅ | **Humanoid Everyday `walk_towards_chair_and_rotate_the_chair/episode_0`** — 581 frames of Unitree G1 head-camera RGB-D, 1.5 cm voxels, TSDF + mesh + ESDF in **4.22 ms/frame = 237 Hz** on an RTX 5090 (depth 0.21 ms, colour 0.98 ms, mesh 0.77 ms, esdf 2.26 ms), or 136 Hz once reading the frames off disk is counted. Metric checks: nvblox' own ground-plane RANSAC lands **2 mm from z=0, 2.1° off vertical** after 581 ICP-tracked frames; on `walk_towards_outside_chair_and_pull_it_out` the reconstructed café tabletop sits **0.735 m** above the reconstructed paving. Poses come from depth frame-to-model ICP (0.6–0.9 cm inlier RMSE) — the G1's own legged odometry smears the TSDF, see its README. | Replica, Redwood, 3DMatch, cuSFM and PLY-LiDAR fusers also build. Benchmark in `perf_bench/dgx_spark/nvblox.json`. **Most Humanoid Everyday tasks cannot be tracked**: the head camera is pitched 56° down so the average frame is one plane, and `pick_up_a_caution_sign...` / `walk_towards_elevator...` collapse — its README has the ICP-conditioning table |
| monolane_mapping | ✅ | **OpenLane rosbags, all 202 segments / 27.4 km.** One segment in detail: 199 frames, 307 m, 14 lane landmarks, 494 control points = **86× fewer points than the raw detections** (5.8 kB vs 497 kB), 75 ms/frame. Pose refinement measured with `--odo_noise`: yaw RPE 1.262° vs 1.608° raw at a 10 m baseline, but no better than raw by 50 m — see its README. Bags ship GT poses, so without `--odo_noise` every RPE is 0. | OpenLane only. Lane-map F1 (`openlane_eval3d.py`) additionally needs the original OpenLane `lane3d_1000/validation` jsons, which the rosbag zip does not include |
| fast_livo2 | ✅ | **FAST-LIVO2-Dataset `Retail_Street`** (native Livox Avia + RGB cam; 1351 poses, 67.42 m closed loop, 4 cm end-to-start = 0.06 % drift, LIO 14.1 ms + VIO 4.7 ms) and Hilti `exp14_basement_2.bag` (738 poses, 37.94 m, LIO 19.3 + VIO 4.2 ms) | 19 more FAST-LIVO2-Dataset sequences via `download_fast_livo2.py` — but groups 2-4 need their own calibration block, see its README. MARS-LVIG, NTU VIRAL launches also ship. |
| cerberus_2 | ✅ | **CMU Garage, full 644 s** (the demo): 23341 poses, 476 m path, **228.3 m xy span against the proprioception-only filter's 225.9 m on the same bag**, one 0.30 m pose step and no others — while stereo **VIO alone diverges inside 30 s**, which is the papers' whole point. **Wightman Park flying trot** (upstream's own default bag and `-u 197` window): 136.9 m closed loop, **4.54 m end-to-start = 3.3 % drift**. **Indoor 31 s square vs Optitrack: ATE 0.070 m fused / 0.045 m proprioception-only / 0.148 m VIO** (the only sequences with a pose topic). Needs `estimate_extrinsic: 0`, not upstream's `1`. | 6 more Go1 sequences via `download_cerberus2.py`. **Mill19 Trail, St Mary Cemetery and the 93 s indoor square all diverge**; the outdoor vertical channel drifts a speed-proportional 4.45° grade (root cause traced to a frame mismatch in the leg-velocity hand-off) — all investigated in its NOTES.md |

---

## Additional datasets that would unlock more verified runs

| Dataset | Size | Sensors | Required by | Access |
|---------|------|---------|-------------|--------|
| **KITTI grey + colour re-download** | 22 GB + 65 GB | stereo cameras | sequences other than 00 for orb_slam2; dsp_slam, octomap, suma_pp | `download_kitti.py` (S3, free). The local copies of both are **truncated** — see the KITTI note above. |
| **ICL-NUIM** | ~0.5 GB | Synthetic RGB-D | concept_fusion | [Free](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html) |
| **Replica** | ~2 GB | Synthetic RGB-D | nvblox, gaussian_splatting_slam, pin_slam | [Free](https://github.com/facebookresearch/Replica-Dataset) |
| **FAST-LIVO2-Dataset, remaining 17 sequences** | 0.4-22 GB each, ~150 GB total | Livox Avia + IMU + RGB cam | nothing outstanding — `Retail_Street` is downloaded and verified | `download_fast_livo2.py --list`. Groups 2-4 need their own calibration block from `calibration.yaml`. |

### Recommended download priority

1. **KITTI grey split, re-download** (high) — the local zip is truncated, so only sequence 00 has images. Unlocks orb_slam2 on other sequences plus dsp_slam, octomap, suma_pp.
2. **ICL-NUIM** (medium) — only free dataset concept_fusion supports.
3. **A LiDAR+IMU rosbag for GLIM** (medium) — GLIM's loop closure needs an IMU to keep drift inside its geometric gate; KITTI is LiDAR-only here, so its global optimizer is only partly exercised.
4. **Replica** (low) — synthetic alternative for already-verified RGB-D pipelines.
5. ~~FAST-LIVO2 reference bags~~ — done. `Retail_Street` is downloaded and verified end-to-end; it is the `fast_livo2` demo sequence.
