# SVO Pro

Semi-direct visual odometry: sparse direct image alignment to track the frame, then
feature-based reprojection and optimisation to refine it. This demo runs the full
**visual-inertial** configuration — SVO frontend plus an OKVIS-style ceres
sliding-window backend — on drone-racing footage recorded by the same lab that wrote it.

- **Repo**: [uzh-rpg/rpg_svo_pro_open](https://github.com/uzh-rpg/rpg_svo_pro_open)
- **Paper**: [SVO: Semi-Direct Visual Odometry for Monocular and Multi-Camera Systems](http://rpg.ifi.uzh.ch/docs/TRO17_Forster-SVO.pdf) — Forster, Zhang, Gassner, Werlberger and Scaramuzza, IEEE T-RO 2017
- Also relevant: [SVO: Fast Semi-Direct Monocular Visual Odometry](http://rpg.ifi.uzh.ch/docs/ICRA14_Forster.pdf) (ICRA 2014, the original); [Benefit of Large Field-of-View Cameras for Visual Odometry](http://rpg.ifi.uzh.ch/docs/ICRA16_Zhang.pdf) (ICRA 2016, the fisheye camera model this demo relies on); [Keyframe-based visual–inertial odometry using nonlinear optimization](https://doi.org/10.1177/0278364914554813) (Leutenegger et al., IJRR 2015 — the OKVIS backend this repo's sliding window is modified from; paywalled, and its freely readable RSS 2013 predecessor is [Keyframe-Based Visual-Inertial SLAM Using Nonlinear Optimization](https://www.roboticsproceedings.org/rss09/p37.pdf))
- **Dataset**: [UZH-FPV Drone Racing Dataset](https://fpv.ifi.uzh.ch/) — [Are We Ready for Autonomous Drone Racing?](https://rpg.ifi.uzh.ch/docs/ICRA19_Delmerico.pdf), Delmerico, Cieslewski, Rebecq, Faessler and Scaramuzza, ICRA 2019

![SVO Pro stereo VIO on UZH-FPV indoor_forward_3](docs/rviz_stereo_vio.png)

Left: the fisheye image with tracked features (green). Right: keyframe frusta and
landmarks accumulated along the flown path, with the backend's sliding-window
trajectory in blue.

## Build

```bash
podman build -t slam_zero_to_hero:svo_pro_open .
```

## Download the dataset

```bash
python3 ../download_uzh_fpv.py            # indoor_forward_3 + calibration, ~1.6 GB
python3 ../download_uzh_fpv.py --list     # all 28 sequences, and which have ground truth
```

Everything lands in `~/data/uzh_fpv/`. The default is `indoor_forward_3`, the sequence
this demo is verified on.

The drone carries two camera systems and the dataset ships both. This demo uses the
**Snapdragon Flight**: a 640×480 stereo *fisheye* pair at 30 Hz plus a 500 Hz IMU.
`--sensor davis` gets the 346×260 mDAVIS event camera instead; SVO consumes ordinary
frames, so the higher-resolution stereo pair is the better fit.

Only sequences whose filename ends in `_with_gt` carry ground truth, and that ground
truth covers **part** of each flight — on `indoor_forward_3`, 49.5 s of the 92 s.
The bag holds exactly five topics:

| Topic | Type |
|---|---|
| `/snappy_cam/stereo_l`, `/snappy_cam/stereo_r` | `sensor_msgs/Image`, 640×480 raw |
| `/snappy_imu` | `sensor_msgs/Imu`, 500 Hz |
| `/groundtruth/pose`, `/groundtruth/odometry` | `geometry_msgs/PoseStamped`, `nav_msgs/Odometry` |

## Run the algorithm

One command runs the pipeline, records the estimated trajectory, replays the bag and
scores the result against ground truth:

```bash
podman run --rm \
  --runtime=/usr/bin/nvidia-container-runtime \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
  -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/data/uzh_fpv:/data:ro \
  -v $PWD/results:/results \
  slam_zero_to_hero:svo_pro_open \
  /svo_ws/scripts/run_fpv.sh /data/indoor_forward_3_snapdragon_with_gt.bag stereo
```

An rviz window opens showing the tracked fisheye image, the landmark cloud, keyframe
frusta and the sliding-window trajectory. No `xhost` change and no `--net=host` are
needed. The two NVIDIA lines give hardware GL; without them it falls back to software
rendering.

Swap `stereo` for `mono` to run the monocular pipeline. Useful variations:

```bash
# No display needed; writes the same trajectories and metrics.
... slam_zero_to_hero:svo_pro_open /svo_ws/scripts/run_fpv.sh /data/<bag>.bag stereo --headless

# Replay slower, or skip into the sequence (mono initialisation sometimes wants this).
... -e RATE=0.5 -e START=5 ... /svo_ws/scripts/run_fpv.sh /data/<bag>.bag mono

# Regenerate the screenshot above on a private Xvfb display, no host X server involved.
... /svo_ws/scripts/capture_rviz.sh /data/<bag>.bag stereo 85
```

Each run writes to `results/<bag>_<mode>/`: `svo_tum.txt` and `gt_tum.txt` (TUM-format
trajectories), `ape.zip` and `ape_plot_*.png` from evo, `rviz.png` if captured, and
`svo.log`.

To drive the pipeline yourself instead of using the wrapper, the launch files take the
usual roslaunch arguments:

```bash
roslaunch svo_ros fpv_vio_stereo.launch          # or fpv_vio_mono.launch
rosbag play /data/indoor_forward_3_snapdragon_with_gt.bag
```

`rviz:=false` for no GUI, `runlc:=true` to enable loop closing, and `calib_file:=` /
`param_file:=` / `cam0_topic:=` to point it at a different sensor set.

## Output

Estimated trajectory (coloured by error) against ground truth (dashed), SE(3)-aligned —
several laps of the indoor racing track, with error staying low except at one turn:

![Stereo APE against ground truth](docs/ape_stereo_trajectory.png)

`run_fpv.sh` writes the same plot for every run.

## Supported datasets

| Dataset | Launch file | Calibration | Notes |
|---|---|---|---|
| **UZH-FPV** Snapdragon, `indoor_forward` (verified: `indoor_forward_3`) | `fpv_vio_stereo.launch`, `fpv_vio_mono.launch` | `UZH_FPV_indoor_forward_snapdragon_{stereo,mono}.yaml` | The verified configuration. |
| **UZH-FPV** Snapdragon, `indoor_45` / `outdoor_*` | same | needs its own calibration — download with `--sequences indoor_45_2` etc. and convert | Each environment has a **different** calibration; do not reuse the `indoor_forward` one. `outdoor_45` is the hardest split in the dataset. |
| **UZH-FPV** mDAVIS | same, with `cam0_topic:=` overridden | not provided here | 346×260 frames; expect to lower `grid_size` and `img_align_max_level` for the smaller image. |
| **EuRoC MAV** | upstream `euroc_vio_stereo.launch`, `euroc_vio_mono.launch` | upstream `euroc_{stereo,mono}.yaml` | Ships with the repo. Mono needs a start offset (`-s 10` for `V2_02_medium`). |
| **FLA** stereo+IMU | upstream `launch/frontend/fla_stereo_imu.launch` | upstream `fla_stereo_imu.yaml` | Frontend-with-IMU only, no ceres backend. |
