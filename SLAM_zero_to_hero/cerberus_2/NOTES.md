# Cerberus 2.0 — notes

Things that are not obvious from upstream's README, in the order you hit them.

## The code publishes almost no visualization

`src/utils/visualization.cpp` registers twelve publishers (`/vilo/path`,
`/vilo/odometry`, `/vilo/point_cloud`, `/vilo/camera_pose_visual`, …) and then has **every
one of them commented out** except `pubTrackImage`. 317 of its 410 lines are comments. So a
stock run emits exactly:

| Topic | Type | From |
|---|---|---|
| `/vilo/image_track` | `sensor_msgs/Image` | `pubTrackImage` — the only live display |
| `/vilo/estimate_pose`, `/vilo/estimate_twist` | `PoseWithCovarianceStamped`, `TwistWithCovarianceStamped` | `VILOFusion::publishVILOEstimationResult` |
| `/mipo/estimate_pose`, `/mipo/estimate_twist`, `/mipo/contact` | same + `PoseStamped` | `publishPOEstimationResult` |
| `/vis_joint_state` | `sensor_msgs/JointState` | reordered to URDF joint order |
| `/tf` | `world → robot → camera` | `publishVILOTF` |

No `nav_msgs/Path`, no `nav_msgs/Odometry`, and **no landmarks** anywhere, so out of the box
rviz can show you the feature tracks as a 2D image and nothing else. Two additions here fix
that:

* `scripts/pose_to_path.py` accumulates the pose streams into `/vilo/path_viz`,
  `/mipo/path_viz` and `/gt/path_viz`, decimated at 2 cm so a 10 min run does not grow a
  Path of 250 000 poses and stall rviz. Viewer aid only.
* `patches/0001-publish-local-map-and-factor-graph.patch` reinstates the 3D structure.
  Upstream's `pubPointCloud()` computed exactly this, but everything it needs
  (`feature_manager_`, `Ps`, `Rs`, `ric`, `tic`) is private to `VILOEstimator`, so the patch
  adds one const accessor, `VILOEstimator::getLocalMap()`, and publishes
  `/vilo/point_cloud` (every landmark with `solve_flag == 1`, lifted to the world frame
  through the pose of the frame it was first seen in) and `/vilo/key_poses`. No estimator
  maths is touched. The patch is generated against the pinned commit and `git apply`d in the
  Dockerfile, so it fails loudly if upstream moves.

The same patch also publishes the window **as a factor graph**, which is what the estimator
is really doing and what no upstream topic exposes:

| Topic | Marker | What it is |
|---|---|---|
| `/vilo/key_poses` | SPHERE_LIST | the WINDOW_SIZE+1 keyframe pose nodes |
| `/vilo/factor_graph_pose` | LINE_LIST, thick | consecutive keyframe pairs: one IMU preintegration factor each, plus a leg factor when `vilo_fusion_type != 0`. Both factors share the same two nodes, so they draw as one edge. |
| `/vilo/factor_graph_obs` | LINE_LIST, thin | landmark → every keyframe that observed it, i.e. one reprojection factor per line. `FeaturePerId::feature_per_frame` holds one entry per *consecutive* frame from `start_frame`, so the observing keyframes are exactly that index range. |

A full window is ~120 landmarks × up to 11 observations, so the reprojection edges are drawn
at 4 mm width and alpha 0.25 or they swamp everything else.

Two things about the landmark cloud that surprise people. It is the **sliding window**, about
ten keyframes' worth -- a local map, not an accumulated one, so it never grows and it sits
wherever the robot currently is. And that is why the rviz view uses `Target Frame: robot`:
anchored to `world` on a 200 m sequence, the cloud and the robot leave the frame within
seconds and you are left looking at bare trajectory lines.

`publishVILOTF` broadcasts `world → robot` from the estimator state *before* the estimator
initialises, when the quaternion is still `(0,0,0,0)`. That produces a ~230-message burst of
"Ignoring transform … invalid quaternion (-nan -nan -nan -nan)" in the first 0.6 s of every
run. Harmless; tf2 drops them.

## Drawing the legs

`cerberus2_main` publishes `/vis_joint_state`: the 12 joint positions, reordered and renamed
to `FL_hip_joint, FL_thigh_joint, FL_calf_joint, FR_...` -- exactly the names in the URDF
that ships in the repo at `urdf/a1_description/`. Nothing upstream consumes it. Feed it to
`robot_state_publisher` and the whole chain
`base -> trunk -> {FL,FR,RL,RR}_{hip,thigh_shoulder,thigh,calf,foot}` appears, moving with
the gait, anchored at the fused pose through `publishVILOTF`'s `world -> robot`. That is the
kinematic chain the leg-odometry velocity is computed from, so it is worth seeing.
`launch/cerberus2_bag.launch` does this behind `leg_viz` (default true), with a static
identity `robot -> base`.

The description is **a1_description -- an A1**, while every released bag is a Go1
(thigh/calf 0.20 m vs 0.213 m). Cosmetic only: the estimator's leg kinematics come from
`include/utils/casadi_kino.hpp`, never from this URDF.

The rviz config deliberately has **no `rviz/TF` display**. `robot_state_publisher` expands
that URDF into ~25 frames inside a 0.6 x 0.3 m body, and the axis triads plus their name
labels completely bury the robot. The RobotModel display shows the same kinematics as
geometry instead.

## Topics you can only set through rosparam

`Utils::readParametersROS` reads seven topic names from the **param server only** — they are
not in the yaml — and then `readParametersFile` overwrites two of them from the yaml. The
five that stay rosparam-only:

```
FL_IMU_TOPIC  default /WT901_49_Data
FR_IMU_TOPIC  default /WT901_48_Data
RL_IMU_TOPIC  default /WT901_50_Data
RR_IMU_TOPIC  default /WT901_47_Data
GT_TOPIC      default /mocap_node/Go1_body/pose
```

Upstream's own launch files set none of them, so a stock run silently takes the defaults —
and `GT_TOPIC`'s default is **wrong for the released bags**, which publish mocap on
`/natnet_ros/Shuo_Go1/pose`. That is why a stock indoor run produces an empty
`gt-<dataset>.csv`. `launch/cerberus2_bag.launch` sets all five explicitly.

The four foot-IMU defaults *are* right for these bags. Checked by correlating each WT901's
gyro magnitude against each leg's summed joint rate over a trotting segment:

```
                  FL       FR       RL       RR
/WT901_47_Data   0.505   -0.072   -0.005    0.468
/WT901_48_Data   0.084    0.401    0.504   -0.009
/WT901_49_Data   0.433   -0.005    0.068    0.406
/WT901_50_Data   0.014    0.464    0.555   -0.073
```

A trot moves the diagonal pairs together, so magnitude alone cannot separate FL from RR —
but it does show `{47,49} = {FL,RR}` and `{48,50} = {FR,RL}`, which is exactly how the
defaults pair them. A crossed mapping would have shown up here.

## Foot gyros are in deg/s

The WT901 units report angular velocity in **degrees per second** — `|ω|` on these bags
averages 117 and peaks at 717, which is impossible in rad/s for a Go1 calf (717 rad/s is
6850 rpm) and entirely normal in deg/s. Upstream converts at
`VILOFusion.cpp:838-841` (`/ 180.0 * M_PI`). Their accelerometers *are* in m/s²: mean 28 m/s²
with a 243 m/s² peak, and a minimum near 0 during the swing phase, which is what a foot in
free flight should read.

## `rosbag play` needs `--hz=2000`

Both estimator loops derive their integration timestep from `ros::Time::now()` differences:

```cpp
double dt_ros = curr_loop_time - prev_loop_time;
if (dt_ros == 0) continue;
...
mipo_estimator->ekfUpdate(mipo_x, mipo_P, *prev_data, *curr_data, dt_ros, ...);
```

Under `use_sim_time` that resolution is the `/clock` publish rate. `rosbag play`'s default
100 Hz quantises `dt_ros` to 10 ms, so the 400 Hz PO loop sees `dt_ros == 0` on most
iterations and `continue`s out. Upstream's own launch files pass `--hz=2000` for this
reason, and `run_demo.sh` does the same (plus `--queue=1000`, since the estimator subscribes
with queue 1000 and rosbag's default publisher queue of 100 drops messages on the 200-400 Hz
topics).

Note the coupling this creates: `interpolateMIPOData` advances its data pointer by the same
`dt_ros` but then **clamps** it to what the measurement queues actually hold
(`getMIPOMinLatestTime()`). When the loop runs ahead of the data the EKF integrates over
`dt_ros` while the data only advanced by less, so the filter's accuracy is tied to the loop
keeping pace with sim time. Playing at `RATE=0.5` produced a bit-for-bit similar trajectory
(440.6 m vs 445.4 m on the same window), so on a 32-core host this is not the limiting
factor — but it is the mechanism to suspect on a slower machine.

## estimate_extrinsic must be 0, not upstream's 1

The one substantive config change here. Upstream's `hardware_go1_vilo_config.yaml` sets
`estimate_extrinsic: 1`, i.e. optimise the camera-IMU transform online around the initial
guess. That is the right choice on live hardware. On a recorded sequence whose rig transform
is already in the config it slowly corrupts the heading, and the longer the run the worse it
gets. CMU Garage, full 644 s, everything else identical:

| | xy span | end-to-start | Wightman 197 s loop closure |
|---|---|---|---|
| `estimate_extrinsic: 1` | 430.0 m | 431.7 m | 6.14 m |
| `estimate_extrinsic: 0` | **228.3 m** | **227.0 m** | **4.54 m** |

The span is the tell. `MIPO`, which never touches the camera and so cannot be affected by
the extrinsics, independently reports 225.9 m on the same bag: with `0` the fused estimate
agrees with it, with `1` it does not. `estimate_td` was tried the same way and helps less
(247.1 m span).

## Indoor sequences: which ones work

Only the indoor bags carry a pose topic, so they are the only source of a real ATE. Two
traps:

* The **20230517 series records the foot IMUs at 27 Hz**, not the 200 Hz of every other
  series (2229 messages over 81 s). On those the estimator emits 428 poses covering the
  first ~11 s of the bag and then stops writing, without crashing -- roslaunch reports a
  clean shutdown. `MIN_PO_QUEUE_SIZE` is 25, so at 27 Hz the foot queues hold barely a
  second of margin against `getMIPOMinLatestTime()`'s clamp. Use the 20230615 / 20230620 /
  20230625 series instead.
* Length still matters. On `230620-risqh-standtrot-05-06-33square1` (31 s) all three
  variants track well; on `20230615-risqh-standtrot-06-06-square` (93 s) all three diverge.

ATE against Optitrack on the 31 s square, rigidly aligned over 11.7 m of mocap path:

| variant | ATE RMSE | ATE max | RMSE / path |
|---|---|---|---|
| `mipo` (5 IMUs + joints, no camera) | **0.044 m** | 0.162 m | 0.38 % |
| `vilo-m` (fused) | **0.070 m** | 0.176 m | 0.60 % |
| `vio` (stereo + trunk IMU, no legs) | **0.148 m** | 0.411 m | 1.26 % |

Ordered exactly as the papers argue: dropping the legs roughly doubles the error. That MIPO
alone edges out the fused estimate here is not a contradiction -- at 0.4 m/s in a small
volume with continuous contact, proprioception is the stronger signal, and the camera's
contribution is what stops it drifting over hundreds of metres outdoors.

## Mill19 Trail diverges

The sequence upstream's README showcases as a video does not work with the released code.
`MIPO` — the camera-free filter — fails first: it tracks correctly for 20 s at 0.5 m/s with
the base height pinned at 0.24 m, then velocity ramps **linearly** to 24 m/s, which is the
signature of leg-velocity corrections dropping out and leaving raw accelerometer
integration (≈0.6 m/s² of uncorrected bias). Vertical stays correct the whole time; only the
horizontal channel runs away.

Ruled out, each by experiment:

| Hypothesis | Test | Result |
|---|---|---|
| Estimator can't keep up in real time | `RATE=0.5` | 440.6 m vs 445.4 m — identical failure |
| Robot sits at bag start, so `init_base_height: 0.3` is wrong | `OVERRIDES=init_base_height=0.05` | still diverges (2300 m) |
| Bad first seconds (foot force ≈ 0.2 for 5 s, robot not loaded) | `START=12` | worse (99 km) |
| Both together | `START=12` + `0.05` | still diverges (566 m) |
| Dropped or gappy messages | per-topic interval histogram | clean: 400/400/200/200/200/200/15 Hz, max gap 44 ms |
| Wrong foot-IMU→leg mapping | diagonal-pair correlation (above) | mapping is right |
| Foot gyro unit confusion | upstream converts deg→rad | handled |
| It's the harness, not the sequence | same image, Wightman Park bag, upstream's own `-u 197` | ✅ 137 m loop closes to 6.14 m |

Variant sweep on the same 120 s window: `vilo-m` 2300 m, `mipo` 4565 m, `vio` 3303 m,
`vilo-tm-n` (tightly-coupled leg factor) collapses to 3.3 m of motion, `vilo-s` (SIPO) is the
only one that stays roughly sane — 58 m of path, 34 m span, but 20 m of vertical drift.
Over the full 419 s `vilo-s` gives 219 m of path with 73 m of vertical drift.

Mill19 is the only *unstructured natural terrain* sequence in the release (a wooded trail),
so foot slip on loose ground breaking the contact assumption is the obvious suspect — but
`vio`, which never touches the legs, also diverges on it, so that alone does not explain it.

## The outdoor vertical channel drifts, and I could not fix it

Worth reading before you trust any z number from an outdoor run.

**What it looks like.** `vilo-m` on the full 644 s CMU Garage bag ends at z = −36.4 m and the
height plot looks like a clean, plausible ramp descent. It is not one.

**What it actually is.** Bucket the run into 10 s windows and regress vertical rate on
horizontal speed:

```
vertical rate vs horizontal speed:  slope = -0.0779   correlation -0.861   (63 buckets)
```

The robot "descends" a constant **4.45° grade for exactly as long as it is moving**, and stops
descending when it stops. That is not terrain and not a random walk — it is a fixed rotation
applied to a velocity. Body pitch tells the same story: it ramps monotonically from −1° to
−27° across the run.

**Where it comes from.** `LOFactor` (include/factor/lo_factor.hpp) is a *displacement*
constraint in VILO's world frame:

```cpp
residual = lo_pre_integration->evaluate(Pi, Pj);   // (Pj - Pi) - integral(v dt)
jacobian_pose_i.block<3,3>(0,0) = -Eigen::Matrix3d::Identity();
```

and the velocity being integrated is handed over in `VILOFusion::POLoop` as

```cpp
vilo_estimator->inputLOVel(curr_esti_time, mipo_x.segment<3>(3), mipo_P.block<3,3>(3,3));
```

`mipo_x.segment<3>(3)` is MIPO's velocity **in MIPO's own world frame**. MIPO and VILO each
gravity-align their own world frame independently and nothing in the pipeline relates them, so
a constant attitude offset between the two turns forward motion into a steady fake grade. A
4.45° offset accounts for the measured slope exactly.

**Two fixes tried, both worse.** Full 644 s run each time:

| | final z | grade |
|---|---|---|
| upstream as-is (`vilo_fusion_type: 1`) | −36.4 m | +4.45° |
| round-trip the velocity through the body frame, `R_rel = R_vilo · R_mipo^T` | +60.5 m | −7.11° |
| `vilo_fusion_type: 2`, tightly-coupled leg factor, which never crosses the frame boundary | −205.7 m | +27.81° |

The round-trip fails because `R_rel` is the difference of *two drifting attitude estimates*,
not the constant offset — rotating by MIPO's own attitude substitutes MIPO's attitude error
for the offset. Type 2 looked promising over a 140 s window (−1.01° vs +4.36°) and is
dramatically worse over the full bag; do not generalise from short windows here.

**What the actual fix would be.** Make the leg-odometry preintegration accumulate
`R_vilo(t) · v_body · dt` internally, the way `IntegrationBase` already does for the IMU,
instead of accumulating a world-frame `v · dt` handed in from another filter. That means
changing `include/factor/lo_intergration_base.hpp`, its covariance propagation and the factor
Jacobians — real estimator surgery, and beyond a demo whose job is to run upstream's algorithm
rather than rewrite it. Upstream's default is kept.

**So:** trust the horizontal circuit, which is independently corroborated (VILO 228.3 m xy span
against MIPO's 225.9 m on the same bag), and treat outdoor z as unvalidated. On the indoor
sequence, which has mocap, full 3D ATE is 0.070 m over 11.7 m — the vertical channel is fine
at that scale over 31 s. It is a long-run effect.

## Dropped from upstream's devcontainer

`.devcontainer/Dockerfile` installs several things this image deliberately does not:

- **gram_savitzky_golay, OSQP, osqp-eigen** — not one header from any of them is included
  anywhere in `include/` or `src/`. Leftovers from Cerberus 1.
- **The elevation-mapping workspace** (`grid_map`, `kindr`, `elevation_mapping`,
  `plane_segmentation`) — used only by `launch/elev_map/*`, not by the odometry.
- **VINS-Fusion beyond `camera_models`** — `vins_estimator` is an alternative estimator this
  demo does not run, and `global_fusion` would pull in GeographicLib. Same fork, same commit.
- **oh-my-zsh** — `zsh` itself is kept, because upstream's launch files use
  `launch-prefix="zsh -c ..."`.

Kept, though nothing links against it: **libtorch**. `CMakeLists.txt` has
`find_package(Torch REQUIRED)`, but `torch/torch.h` is included only by
`MIPOEstimatorTensor.{hpp,cpp}` and `torch_kino.{hpp,cpp}`, and neither file appears in
`fusion_SRC` or `vilo_fusion_SRC`. All the dependency contributes is `TORCH_CXX_FLAGS`
(`-D_GLIBCXX_USE_CXX11_ABI=1`, already gcc-9's default on focal). It is pinned to 1.13.1+cpu
rather than upstream's `...-latest.zip` nightly because that URL is a moving target and
libtorch ≥ 2.0's `TorchConfig.cmake` forces `CMAKE_CXX_STANDARD 17`, colliding with this
project's C++14.

## A unit test runs during the build

`CMakeLists.txt` adds `run_test_LOTightIntegration` to the `ALL` target, i.e. `catkin build`
**executes** `test_LOTightIntegration` as part of compiling. Two consequences:

- `/home/EstimationUser/estimation_ws/devel/lib/cerberus2` must exist beforehand (it is the
  target's `WORKING_DIRECTORY`) and so must `bags/output`, because the test calls
  `readParametersFile()`, which truncates the result CSVs under `output_path`. Upstream
  creates both in a devcontainer `postStartCommand`; the Dockerfile `mkdir`s them.
- The build prints a wall of `mv: cannot stat 'dvdwf_fun0.h': No such file or directory`.
  That is the test's casadi code generation, and it is cosmetic — the build succeeds.

## rviz will not start as a roslaunch node

Started as `<node pkg="rviz" type="rviz" .../>` inside the launch file, rviz stays alive but
never maps a window: nothing in its log, no "process has died" from roslaunch, and no rviz
window anywhere in the X tree 13 minutes into a run. The identical
`rviz -d <same config>` run as a plain child process on the same display comes up every
time, in under 25 s, with or without `use_sim_time` set and with or without a `/clock`
publisher. So `run_demo.sh` owns rviz itself, waits for the window by title
(`… - RViz`, since the dozen bare `rviz`-titled windows are child widgets), and keeps its
stderr in `/out/rviz.log`.
