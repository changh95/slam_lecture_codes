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
  Both publishers are called from `POLoop()`, the 400 Hz proprioceptive loop. Since
  2026-09-28 they are throttled to 10 Hz of wall time and skipped when nothing
  subscribes. Before that they ran on every iteration and try-locked the solver mutex
  400 times a second. The throttle was not the cure for the random divergence (see "Random
  divergence on CMU Garage" below), but it keeps the readout off the
  estimator's hot path.

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
`dt_ros` while the data only advanced by less. It looks like the obvious suspect for runs
that differ from each other, and it is **not** what made them diverge. Logged per iteration
at a host load of 40, the sensor time advanced by exactly `dt_ros` in 99.7 % of the loop's
iterations: after the first sample the estimator sits a constant ~85 ms behind `/clock`, so
the clamp only bites at start-up. The largest step was 17 ms.

## Random divergence on CMU Garage: the first frames had one IMU sample

Until 2026-09-28 identical runs of CMU Garage gave different answers, and some blew up: the
dog "flipped" in rviz, z jumped to 20 m and roll to ±π within the first 10 s, or later at
55 s or 447 s. It looked like thread timing under host load, `-r 0.5`, or rviz under the
NVIDIA runtime. It was none of those. It was an initialisation bug, fixed by
`patches/0002-wait-for-imu-before-first-frame.patch`.

**Mechanism.** The body IMU does not reach `VILOEstimator` from its own subscriber. It comes
through `POLoop()` (`VILOFusion.cpp`), which only starts once every proprioceptive queue
holds `MIN_PO_QUEUE_SIZE` = 25 samples. The slowest of those queues are the 200 Hz foot
IMUs, so on CMU Garage the first IMU sample arrives ~0.15 s after the first camera frame.
`processMeasurements()` waits until *some* IMU sample is newer than the frame, and
`getBodyIMUInterval()` never checks that the IMU also covers the start of the interval. So:

```
frame  interval  IMU samples  preintegrated over
  1    66.7 ms        1          0.155 s      <- the one sample, from beyond the frame
  2    66.7 ms        1          0.088 s      <- the same sample again
  3    66.7 ms       31          0.067 s
```

A preintegration of a single step has a singular covariance. Its square-root information,
measured, was **8.4e20 and 8.8e26** for those two factors (a normal one is ~1e5). Every one of
the ~106 "numerical unstable in preintegration" warnings a run used to print came from
these two factors, checked one by one. When the frames are marginalised that information
goes into the prior and stays. The gyro bias initialisation, which solves over the first
ten frames, absorbed the one raw gyro sample: across identical runs the initial bias came out
anywhere from x = -0.0098 to +0.0017 and z = -0.0076 to +0.0024 rad/s, while the robot's real
standing bias, averaged straight from the bag, is about (0.0001, 0.0000, -0.0002). The bias
random walk here (`gyr_w`) lets the estimate move only ~2e-4 rad/s over the whole run, so the
initial value decides the heading drift for all 644 s.

Which sample the loop happened to grab depends on thread timing, which is why runs differed.
A prior that ill-conditioned makes every later solve sensitive to tiny differences, which is
why the failures looked random: at 5 s, at 10 s when the robot starts to walk and vision
thins out, or minutes later. Rate, load and the GPU runtime only reshuffled the timing:

| before the fix (30-45 s runs) | diverged |
|---|---|
| `-r 1`, load 6-46 | 0/12 |
| `-r 0.5`, load 4-30 | 1/8 |
| pinned to 2 cores, load 13-66 | 4/35 |
| rviz under the NVIDIA runtime | 0/3 (3/3 on an earlier day) |
| St Mary Cemetery, full bag | 2/2 (340 km, 896 km) |

**The fix.** 14 lines in `processMeasurements()`: until the first frame has been
initialised, drop camera frames older than the first IMU sample. Nothing is clamped or
filtered on the output; the estimator simply starts two frames later, with IMU under every
frame. After it: 0 warnings, and the initial bias comes out (0.0002..0.0004,
-0.0002..0.0000, 0.0000..0.0003) in every run. Same N-run protocol:

| after the fix | diverged |
|---|---|
| 30-45 s runs, `-r 1` / `-r 0.5` / 2 cores / NVIDIA rviz, load 4-66 | 0/47 |
| final image: 12 × `-r 1`, 6 × `-r 0.5`, 3 × rviz on NVIDIA, 1 × rviz on Mesa, load 9-36 | 0/22 |
| St Mary Cemetery, full bag | 0/3 |

On bags where the IMU already starts before the camera (indoor, Wightman Park) the patch
never triggers and the results are unchanged.

**The old "good" CMU Garage numbers were wrong.** The run that produced "475.9 m path,
228.3 m span" had started from an initial gyro bias z of -0.0031, i.e. one noise sample. The
iPhone GPS that ships next to the bag (`.mat`, 276 fixes with better than 10 m accuracy, at
the start and end of the circuit outside the garage) settles it. Rigid 2D fit over the whole
run:

| | path | xy span | end-to-start | z range | RMSE vs GPS |
|---|---|---|---|---|---|
| before, six runs | 474.5-474.9 m | 202-250 m | 101-234 m | 37 m | **47-104 m** |
| after, eight runs | 475.0-478.7 m | 267-271 m | 354-357 m | 7-10 m | **4.6-4.9 m** |

4.6-4.9 m is the GPS's own accuracy (median 4.8 m). The GPS is a MATLAB Mobile `timetable`,
which `scipy.io.loadmat` cannot decode; the lat/lon/time arrays are plain float64 inside the
`__function_workspace__` blob and were read from there (phone clock = bag clock - 14397 s).

One thing the fix does not change: at 376.7 s a burst of new visual outliers raises the
visual cost from ~600 to 2.4e4 for one frame. The Huber loss lets them pull the newest pose
up by ~0.15 m (0.37 m in the logged output) before `outliersRejection()` removes them, and
the next frame is back on the track. It happens in every run at the same frame, so the drift
table reports 2 steps over 25 cm. It is upstream VINS behaviour, not a divergence.

## estimate_extrinsic must be 0, not upstream's 1

The one substantive config change here. Upstream's `hardware_go1_vilo_config.yaml` sets
`estimate_extrinsic: 1`, i.e. optimise the camera-IMU transform online around the initial
guess. That is the right choice on live hardware. On a recorded sequence whose rig transform
is already in the config, `0` is better. CMU Garage, full 644 s, with patch 0002, against the
iPhone GPS (rigid 2D fit):

| | path | xy span | RMSE vs GPS |
|---|---|---|---|
| `estimate_extrinsic: 1` | 476.7 m | 258.4 m | 6.4 m |
| `estimate_extrinsic: 0` | 478.2-478.7 m | 268-270 m | **4.6-4.7 m** |

An earlier version of this section argued from the xy span, taking MIPO's 225.9 m as an
independent reference. It is not one: MIPO takes its yaw from VILO (`getYawObservation()`),
so it inherits VILO's heading. Those numbers also predate patch 0002 and came from a bad
initial gyro bias. On Wightman Park, where the patch never triggers, `0` closes the 197 s loop
to 4.54 m against 6.14 m for `1`.

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

**Update 2026-09-28:** Mill19 also had the one-IMU-sample start (29-65 "numerical unstable"
warnings per run). With patch 0002, 3 of 5 120 s runs are clean (75.4 m, 0 jumps) where every
run used to diverge ~22 s in; the other two and a full 419 s run still diverge ~35 s in, with
0 warnings. So there is a second problem on this sequence. The experiments below were all
made before the patch.

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

**Update 2026-09-28: most of this was the start-up bug.** Everything below was measured
before patch 0002, from runs whose initial gyro bias was one noise sample (see "Random
divergence on CMU Garage"). With the patch the full CMU Garage run ends at z = −9.5 m instead
of −36.4 m, and the grade regression below gives a slope of −0.0164 (0.9°), correlation
−0.19, instead of −0.0779 and −0.861. The frame-mismatch argument about `LOFactor` still
holds as code reading, but its measured effect was mostly the tilted initialisation. The
section is kept as the record of what was tried.

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
