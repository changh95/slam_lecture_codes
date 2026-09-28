# SLAM Algorithm Dataset Compatibility

## Datasets Already Downloaded (`~/data/`)

"Used by" lists the demos that run on this dataset. **Bold** means it is that demo's default.

| Dataset | Path | Sensors | Used by |
|---|---|---|---|
| **TUM RGB-D** | `~/data/tum_rgbd/rgbd_dataset_freiburg{1,2,3}_*` — all 15 sequences, 16 GB extracted | RGB-D (Kinect v1, 640x480) + GT | **orb_slam2** (`freiburg1_desk`, RGB-D), **gaussian_splatting_slam** (`freiburg1_desk`, RGB-D), **mast3r_slam** (`freiburg1_room`, monocular RGB). `download_tum_3d.py` fetches all 15 sequences; it takes no arguments and has no `--list` |
| **EuRoC MAV** | `~/data/euroc_mav/MH_01_easy/mav0/` | stereo 752x480 @20 Hz + 200 Hz IMU + GT | **basalt**; orb_slam2 (mono/stereo, not run). `download_euroc_mav.py [SEQ ...]` (default MH_01_easy, `--list`) extracts straight to `~/data/euroc_mav/<SEQ>/mav0` |
| **Monado SLAM** | baked into the `basalt` image at `/MIPB07_beatsaber_fitbeat_expertplus_2` | Valve Index stereo + IMU | basalt (VR/AR section, Final Project 6) |
| **KITTI odometry** | `~/data/kitti_vo_slam/extracted/dataset/` — see the KITTI note below | stereo grey + Velodyne HDL-64E + GT poses | **kiss_slam** (00), **pin_slam** (00), **suma_pp** (00, also 04), glim (04 / 00, secondary), orb_slam2 (00 stereo/mono, secondary) |
| **KITTI 07, DSP-SLAM package** | `~/data/dsp_slam/kitti/07` + DeepSDF `~/data/dsp_slam/weights/deepsdf/cars_64` (5.8 GB extracted) | stereo + colour + Velodyne + pre-computed MaskRCNN / PointPillars labels, 1101 frames | **dsp_slam** (`download_dsp_slam.py`, public SharePoint, ~3.6 GB). Scored against `kitti_vo_slam/extracted/dataset/poses/07.txt` |
| **Hilti 2022** | `~/data/hilti_2022/exp21_outside_building.bag` (12,014,463,331 B) + sparse GT `exp21_outside_building_gt.txt` (5 surveyed positions) + `exp21_outside_building_carto.bag` (1.9 GB, preprocessed for cartographer); `exp14_basement_2.bag` (6,260,771,085 B) + GT `exp14_basement_2_imu.txt` (689 TUM poses, IMU frame) + `exp14_basement_2_carto.bag` (898 MB) | Hesai PandarXT-32 + Alphasense IMU + 5x cam | **cartographer** (`exp21_outside_building`, 3D + IMU; `exp14` kept as secondary), **fast_lio2** (`exp14`), fast_livo2 (`run_hilti.sh`, `exp14`), kiss_slam (`config/hilti_indoor.yaml`, `exp14`). `download_hilti_2022.py [SEQ ...]` (default `exp14_basement_2`, `--all`, `--challenge-only`, `--dest`) now downloads from the [Hugging Face mirror](https://huggingface.co/datasets/Hilti-Research/hilti-slam-challenge-2022); the old S3 bucket returns 403. GT files are fetched separately (see `cartographer/README.md`) |
| **Korea_drive** | `~/data/Korea_drive/KOREA_DRIVE/` (ROS 2 bag, 49 GB) + `config/` | Hesai LiDAR (109k pts/scan) + 100 Hz OXTS IMU + GNSS, 27 min / 11 km vehicle drive | **glim** (via `glim_rosbag`). No download script: the source is not public |
| **FAST-LIVO2-Dataset** | `~/data/fast_livo2/Retail_Street.bag` + `calibration.yaml`; `Red_Sculpture.bag` and `CBD_Building_01.bag` also downloaded but unused | Livox Avia + built-in IMU + RGB pinhole cam | **fast_livo2** (`download_fast_livo2.py`, 17 more sequences). Only `CBD_Building_01` and `Bright_Screen_Wall` share Retail_Street's calibration; `Red_Sculpture` needs its own block |
| **uHumans2 (MIT SPARK, TESSE sim)** | `~/data/kimera_semantics/uHumans2_office_s1_00h_lz4.bag` (24.2 GiB LZ4 copy of the 15.7 GiB bz2 download, 506 s; default) + `uHumans2_apartment_s1_00h.bag` (2.5 GiB, 138 s; quick option) | depth 32FC1 + 2D semantic segmentation rgb8 + stereo RGB + IMU + GT odom/TF, 720x480 | **kimera** (Kimera-Semantics; `download_kimera_semantics.py`, 12 bags via `--list`). The original `kimera_semantics_demo.bag` is deleted from Google Drive (HTTP 404 for both published ids) |
| **ICL-NUIM** | `~/data/icl_nuim/living_room_traj2_frei_png/` (881 rgb + 881 depth, `associations.txt`, `livingRoom2n.gt.sim`) | synthetic RGB-D + GT | **concept_fusion** (living room 2). `download_icl_nuim.py [SEQ ...]` (default living room 2, `--list`) now extracts each sequence into its own folder. The flat `~/data/icl_nuim/rgb|depth` left by the old version is a **corrupted mix of all 16 sequences**; do not use it. ConceptFusion weights (SAM, OpenCLIP) are in `~/data/concept_fusion` via `concept_fusion/scripts/download_weights.sh` |
| **OpenLane (rosbag conversion)** | `~/data/openlane/OpenLane/lane3d_1000/rosbag/` — 202 x 20 s Waymo segments, 630 MB | PersFormer 3D lane detections + GT lanes + vehicle pose (**no images**) | **monolane_mapping** (`download_openlane.py`, one 433 MB zip holds all 202) |
| **UZH-FPV Drone Racing** | `~/data/uzh_fpv/indoor_forward_3_snapdragon_with_gt.bag` (1.5 GiB) + `calib/` (Kalibr, per environment) | Snapdragon Flight 640x480 stereo **fisheye** @30 Hz + 500 Hz IMU + partial GT (49.5 s of 92 s) | **svo_pro_open** (`download_uzh_fpv.py`, 28 sequences; each environment has its **own** calibration) |
| **UAMC (Gwanghwamun / COEX)** | `~/data/gwanghwamun_coex/extracted/lvi_set_2_restamped` (ROS 2 bag, 23.2 GB .db3, 334.7 s) + archive `lvi_set_2_restamped.zst`; `lvi_ghm_set` on disk but untested and not type-patched | Livox Avia + Mid-360 + IMU + Oak-D RGB | **uamc** (`download_gwanghwamun_coex.py --extract --patch-type`; 10.4 GB archive, 21.6 GB extracted) |
| **Cerberus 2.0 Go1** | `~/data/cerberus2/{cmu_garage,mill19_trail,wightman_park,st_mary_cemetery,indoor}/` | Unitree Go1 quadruped: trunk IMU @400 Hz + **4 foot IMUs** (WT901 @200 Hz, gyro in deg/s) + joint encoders @400 Hz + rectified stereo IR @15 Hz. Indoor sequences add Optitrack on `/natnet_ros/Shuo_Go1/pose`; outdoor ones ship iPhone GPS as a MATLAB `timetable` **object** that only MATLAB can read | **cerberus_2** (`cmu_garage`; `download_cerberus2.py`, 11 sequences — with no argument it also fetches `mill19_trail`, so pass `cmu_garage` explicitly) |
| **Humanoid Everyday** | `~/data/humanoid_everyday/<task>/episode_N/` — 8 tasks on disk, zips kept in `zips/` | Unitree G1 head-mounted RealSense D435: 640×480 RGB + **raw uint16 depth in an lzma buffer, no npy header**, both @30 Hz and **not** mutually aligned (depth ≈79° FOV, colour ≈56°); an unnamed LiDAR @~6.8 k xyz pts; joint state, IMU and legged odometry in `robot_data.jsonl` | **nvblox** (`walk_towards_chair_and_rotate_the_chair/episode_0`; `download_humanoid_everyday.py`, `--list` for all 259 tasks, `--category loco_manipulation` for the ones that walk) |
| **cow_and_lady** | `~/data/cow_and_lady/` | RGB-D + Vicon | voxblox |
| **Replica** | `~/data/replica/Replica/` (12 GB) | synthetic RGB-D | on disk, not used by any verified run |

### KITTI: what is actually on disk

The KITTI source zips in `~/data/kitti_vo_slam/` (velodyne, grey, colour, calib, poses) **are gone**: an accidental `download_kitti.py` run deleted them. What remains:

| Tree | State |
|---|---|
| `extracted/dataset/` | seq **00**: `image_0` + `image_1` (4541 frames each, recovered from a truncated grey zip). `velodyne` complete for **00 (4541), 01 (1101), 02 (4661), 03 (801), 04 (271)**; 05 partial (584 of 2761) as `05/velodyne_partial`. `calib.txt` + `times.txt` for every sequence, `poses/{00..10}.txt`. |
| `dataset/sequences/` | redundant byte-duplicates of 00/04 velodyne plus calib/poses, left over from the accidental extraction (safe to delete). |

So camera demos on KITTI work only on **sequence 00** (orb_slam2 stereo/mono); LiDAR demos on 00-04. Other sequences need the 79 GB velodyne zip (and the grey zip for images) re-downloaded. dsp_slam does **not** depend on this: it uses the authors' own KITTI 07 package.

---

## Algorithms — Dataset Compatibility & Verification Status

Status legend: ✅ verified end-to-end (Docker build → real-data run → visualization captured), 🟡 build verified, run unconfirmed, ❌ not yet attempted / does not work. Numbers are from the last verification (2026-09-27/28 unless stated).

### Classical SLAM

| Algorithm | Status | Verified dataset (default first) | Other supported datasets |
|-----------|:------:|---|---|
| orb_slam2 | ✅ | **TUM RGB-D `freiburg1_desk`, RGB-D** (`rgbd_tum` + `TUM1.yaml` + in-image `associations/fr1_desk.txt`): RMS ATE 1.52-1.57 cm SE(3) over 434-573 poses; Pangolin viewer verified on the host display with NVIDIA GL (the stock viewer aborts under Xvfb, use `rgbd_tum_headless` for unattended runs). Secondary, earlier and headless: KITTI 00 stereo 1.30 m, mono 5.3-6.0 m | EuRoC (mono/stereo) and other TUM sequences: supported, not run |
| basalt | ✅ | **EuRoC `MH_01_easy` stereo + IMU**: 3682/3682 frames, RMS ATE 0.076 m SE(3), 80.1 m path; Pangolin GUI on the host display and headless via `scripts/capture_gui.sh`. Monado SLAM Valve Index `MIPB07` (Final Project 6): 8105 frames, RMS ATE 0.062 m. Image pinned to Monado fork commit a90a57d7 | TUM-VI 512x512, other EuRoC / Monado sequences (not run) |
| cartographer | ✅ | **Hilti 2022 `exp21_outside_building.bag`, 3D + IMU** (`config/hilti_outdoor_3d.lua`): 1,518 poses, 130.3 m outdoor walk, ~87 × 84 m map, **0.093 m RMSE at the 5 survey points** (online 1x run: 1,527 poses, 0.099 m), 192 loop constraints, 49.9 M-point 3D map (verified 2026-09-28). Secondary: `exp14_basement_2.bag` with `config/hilti_3d_lio.lua`: 730 poses, 38.38 m, **0.084 m RMSE against FAST-LIO2**. Each bag needs a one-time `scripts/hesai_add_time_field.py` pass that writes `<seq>_carto.bag`. rviz uses the course-wide mouse scheme (`rviz_unified_controls`) | any ROS PointCloud2 + IMU stream. The old 2D config is kept for contrast |
| kiss_slam | ✅ | **KITTI 00**: 4541 poses, 3726.18 m path (GT 3724.19 m), unaligned ATE 5.59 m mean / 6.12 m RMSE, 0.575 % translation error, 7 loop closures, 27-54 Hz CPU only; Open3D viewer on the desktop and headless via `capture_viewer.py` (a full-sequence software-GL capture takes > 90 min, use `-n`). KITTI 04: 271 scans, ATE 0.59 m. Hilti `exp14` only with `config/hilti_indoor.yaml` | MulRan, nuScenes, NCLT, Apollo, TUM, mcap, generic `.bin`/`.pcd`/`.ply` dirs |
| glim | ✅ | **Korea_drive** ROS 2 bag via `glim_rosbag` (GPU odometry + vgicp_gpu factors, standard_viewer): full bag 16,367 poses, 10.99 km, ATE 4.58 m 2D / 12.9 m 3D vs GNSS (`scripts/eval_korea_gnss.py`). Real-time viewer config re-verified over 600 s: 1.000x pacing, empty odometry queue, 5,940 poses, ATE 2.38 m 2D. The loop does not close vertically (43 m). Secondary: KITTI 04 via `glim_kitti`, ATE 2.60 m over 376.6 m | any LiDAR+IMU ROS 2 bag (ROS 2 Jazzy + glim_ros2 v1.0.0) |
| fast_lio2 | ✅ | **Hilti 2022 `exp14_basement_2.bag`** with `config/hilti_pandarxt32.yaml`: 737 poses, 37.93 m path, **ATE 0.050 m RMSE** against Hilti's `exp14_basement_2_imu.txt` GT (`scripts/ate.py`); headless rviz screenshot via `SCREENSHOT=1`. `RELAY=1` real per-point stamps variant verified 2026-08-05 | upstream Livox / Ouster / Velodyne configs (not verified) |
| fast_livo2 | ✅ | **FAST-LIVO2-Dataset `Retail_Street`**: 1351 poses, 67.43 m out-and-back walk, 4 cm end-to-start (0.06 % drift, no GT); coloured RViz map + path via Xvfb and on the host display. Hilti `exp14` verified earlier via `run_hilti.sh` | CBD_Building_01 / Bright_Screen_Wall (same calibration); other groups need their own calibration block. MARS-LVIG, NTU VIRAL launches also ship |

### AI + SLAM

| Algorithm | Status | Verified dataset (default first) | Other supported datasets |
|-----------|:------:|---|---|
| dsp_slam | ✅ | **KITTI 07, DSP-SLAM authors' package** + DeepSDF `cars_64` (`download_dsp_slam.py`), offline detectors, CUDA 12.8 / PyTorch 2.7.1 cu128 on the RTX 5090: 1101 frames, **RMS ATE 0.52 m over 695 m** (SE(3) vs `poses/07.txt`), loop closed, 99 cars reconstructed; Pangolin viewer renders the car meshes | other KITTI sequences need the online mmdet3d detectors (not built for sm_120). Freiburg Cars / Redwood Chairs: downloadable, not run |
| kimera | ✅ | **uHumans2 `office_s1_00h`** (Kimera-Semantics with GT pose): semantic mesh 2.70 M vertices over 54.6 x 52.8 m, 17 classes, real time (508 s for the 506 s bag) from an LZ4 re-compressed copy with NVIDIA-GL RViz; the stock bz2 bag plays at only 0.33× (apartment: 755k vertices). The README's original `kimera_semantics_demo.bag` is deleted from Google Drive (HTTP 404). Kimera-VIO is no longer built in this folder | other uHumans2 bags (`download_kimera_semantics.py --list`) |
| concept_fusion | ✅ | **ICL-NUIM `living_room_traj2_frei_png`**, 44 frames (stride 20 over 0-879): SAM ViT-H + OpenCLIP ViT-H-14 pixel-aligned features fused by gradslam PointFusion with GT poses, 258,122-point map with 1024-D embeddings in 196 s on the RTX 5090; the text queries "sofa" and "table" highlight both sofas and the coffee table (Open3D renders + rerun recording) | ScanNet (institutional access), other ICL-NUIM sequences |
| gaussian_splatting_slam | ✅ | **TUM RGB-D `freiburg1_desk`, RGB-D** via MonoGS (CUDA 12.8 / torch cu128, sm_120): 592 frames, ~138 keyframes, keyframe ATE 1.44-1.56 cm (paper 1.50), PSNR 23.5-23.7 dB after refinement, 1.8-1.9 fps; MonoGS GUI captured headless under Xvfb | TUM fr2_xyz / fr3_office, TUM mono, Replica, EuRoC stereo configs ship but were not run |
| mast3r_slam | ✅ | **TUM RGB-D `freiburg1_room`**, monocular RGB (`config/calib.yaml`): 681 frames, 51 keyframes, 21 loop/retrieval edges, **ATE RMSE 0.061 m** (Sim(3), `evo -as`), 16.5 FPS on the RTX 5090; headless viewer screenshots. CUDA 12.8.1 + torch 2.7.1 cu128 with sm_120 source patches (NOTES.md); checkpoints baked into the image | other TUM sequences, EuRoC, 7-Scenes, ETH3D, MP4 / image folders (not run) |
| pin_slam | ✅ | **KITTI 00 Velodyne** (stock `run_kitti.yaml`): all 4541 scans, **SLAM ATE 0.847 m over 3724 m** (odometry-only 5.58 m), 32 loop corrections, 13.2 fps on the RTX 5090, 24 cm mesh; Open3D GUI verified under Xvfb on frames 0-1000 (ATE 0.356 m) | TUM RGB-D, Replica, MulRan, NCD, etc. (upstream configs, not verified) |
| suma_pp | ✅ | **KITTI 00 Velodyne**: 4541 scans, ATE 1.06 m mean rigid-aligned / 7.30 m unaligned, 12.6 Hz end-to-end; KITTI 04: 271 scans, 0.29 m aligned. RangeNet++ runs live through TensorRT 10.9 (TensorRT 10 port of rangenet_lib, original darknet53 ONNX, FP32 engine built in ~30 s and cached). SuMa's OpenGL needs NVIDIA GL on a real X display; Mesa llvmpipe in Xvfb does not work | KITTI only (RangeNet++ does not generalize); only 00 and 04 have velodyne extracted here |

### Final projects

| Algorithm | Status | Verified dataset (default first) | Other supported datasets |
|-----------|:------:|---|---|
| monolane_mapping (P1) | ✅ | **OpenLane rosbags**, all 202 bags open (39,981 frames). Default curve segment `9041488218266405018`: 198 frames, 308.5 m, 9 lanes / 541 control points (104x fewer than 56,320 raw points), 119 ms/frame, 31 s wall-clock. Streaming rerun web viewer works (port 9090, websocket 9877); `.rrd` files need a rerun 0.18.2 viewer. Bag poses are GT, so pose error is 0 without `--odo_noise` | Camera panel and F1 evaluation need the gated full OpenLane download: not verified |
| svo_pro_open (P2) | ✅ | **UZH-FPV `indoor_forward_3` Snapdragon stereo + IMU**: RMS ATE 0.453 m headless / 0.588 m with rviz, 2317-2526 poses over 92 s (real-time replay is not deterministic); rviz capture with Xvfb in the container. Mono verified earlier at 0.156 m Sim(3) | other UZH-FPV sequences (each needs its own calibration), EuRoC via upstream launches |
| uamc (P3) | ✅ (partial) | **UAMC COEX `lvi_set_2_restamped`**, U-AMC FAST-LIVO2-ROS2 (ROS 2 Humble, CPU). ✅ Avia LiDAR-inertial over the whole bag: 3320 poses, 336.3 m, start-end 6.6 m, real time, needs `imu_time_offset -0.0697`. ✅ Avia LiDAR-visual-inertial with a camera-coloured RViz map for the first 150 s only (drifts or diverges later). ❌ Mid-360: per-point timestamps 1200 s off. No GT. The image needs the vikit patch in `uamc/patches` or it dies with SIGFPE at start-up | `lvi_ghm_set` on disk, untested |
| cerberus_2 (P4) | ✅ | **Cerberus 2.0 Go1 CMU Garage**, full 644 s: ~478 m path, **4.6–4.9 m RMSE vs the bag's iPhone GPS** in every run. Estimator patch `0002` (don't preintegrate the first frames from a single IMU sample) removed the random early divergence: 5/58 → 0/69 short runs, and St Mary Cemetery 2/2 → 0/3 diverged. One reproducible 0.37 m single-frame spike at 376.7 s (upstream VINS outlier handling). Wightman Park and `indoor_square_31s` also work | **Mill19 Trail** still diverges in 2 of 5 runs (a second, unrelated problem); the indoor 93 s square diverges; indoor two-loops (27 Hz foot IMUs) stops emitting |
| nvblox (P5) | ✅ | **Humanoid Everyday `walk_towards_chair_and_rotate_the_chair/episode_0`**: 581 frames, depth frame-to-model ICP path 2.97 m, ground plane 1.9 mm from z=0 and 2.1° tilt, **4.17 ms/frame** mapping on the RTX 5090. `walk_towards_outside_chair_and_pull_it_out` is the second verified task. The rerun SDK in the image is pinned to the host viewer 0.33.0 | Replica, Redwood, 3DMatch fusers also build. **Most Humanoid Everyday tasks cannot be tracked** (head camera pitched 56° down); `pick_up_a_caution_sign...` / `walk_towards_elevator...` collapse |
| basalt (P6) | ✅ | Monado SLAM Valve Index `MIPB07`: 8105 frames, RMS ATE 0.062 m (see basalt above) | |

### Other folders (not re-verified in this round)

| Algorithm | Status | Verified dataset | Other supported datasets |
|-----------|:------:|---|---|
| voxblox | ✅ | cow_and_lady (perf_bench) | generic ROS PointCloud2 / depth |
| octomap | 🟡 | — | KITTI Velodyne (`benchmark_kitti`) |
| cuvslam | 🟡 | — | NVIDIA cuVSLAM; benchmark in `perf_bench/dgx_spark/cuvslam.json` |

---

## Additional datasets that would unlock more verified runs

| Dataset | Size | Sensors | Required by | Access |
|---------|------|---------|-------------|--------|
| **KITTI velodyne + grey re-download** | 79 GB + 22 GB | Velodyne + stereo cameras | orb_slam2 on sequences other than 00; kiss_slam / pin_slam / suma_pp / glim / octomap on 05-21 | `download_kitti.py` (S3, free). The local zips were deleted; see the KITTI note above |
| **Full OpenLane** (images + jsons) | large, gated | camera + 3D lanes | monolane_mapping camera panel and F1 evaluation | [OpenLane](https://github.com/OpenDriveLab/OpenLane) (registration) |
| **FAST-LIVO2-Dataset, remaining 17 sequences** | 0.4-22 GB each | Livox Avia + IMU + RGB cam | nothing outstanding | `download_fast_livo2.py --list` |

### Recommended download priority

1. **KITTI velodyne zip, re-download** (medium): only 00-04 (and part of 05) have scans. Every KITTI-based demo already has a verified default.
2. ~~**ICL-NUIM, re-extract per sequence**~~ fixed 2026-09-28: `download_icl_nuim.py` extracts into `<sequence>/`. The old flat `~/data/icl_nuim/rgb|depth` can be deleted.
3. **Full OpenLane** (low): unlocks the monolane_mapping camera panel and F1 score.
