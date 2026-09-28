-- Cartographer 3D (LiDAR + IMU) for the Hilti SLAM Challenge 2022 sequence
-- exp21_outside_building.bag -- the default, OUTDOOR demo.
--
-- Same handheld rig, topics, frames and LiDAR<-IMU extrinsic as exp14_basement_2
-- (see hilti_3d_lio.lua for the frame and IMU-sign notes, which all still apply):
--   /hesai/pandar    PointCloud2, frame "PandarXT-32", 10 Hz, with the per-point
--                    `time` field added by scripts/hesai_add_time_field.py
--   /alphasense/imu  Imu, frame "imu_sensor_frame", ~400 Hz
--
-- What changes outdoors: returns reach 20-60 m (p95) and up to ~150 m, instead of
-- 2-15 m in the basement. The basement config (hilti_3d_lio.lua) clips at 40 m and
-- works at 5 cm; run unchanged on this bag it scores 0.517 m RMSE at the 5 survey
-- points, almost all of it height drift. This file scores 0.093 m. Every number
-- quoted below is that survey RMSE (scripts/eval_survey.py), changing one value.

include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
  map_frame = "map",
  tracking_frame = "imu_sensor_frame",     -- 3D integrates the IMU in this frame
  published_frame = "imu_sensor_frame",
  odom_frame = "odom",
  provide_odom_frame = true,
  publish_frame_projected_to_2d = false,
  use_pose_extrapolator = true,
  use_odometry = false,
  use_nav_sat = false,
  use_landmarks = false,
  num_laser_scans = 0,
  num_multi_echo_laser_scans = 0,
  num_subdivisions_per_laser_scan = 1,
  num_point_clouds = 1,
  lookup_transform_timeout_sec = 0.2,
  submap_publish_period_sec = 0.3,
  pose_publish_period_sec = 5e-3,
  trajectory_publish_period_sec = 30e-3,
  rangefinder_sampling_ratio = 1.,
  odometry_sampling_ratio = 1.,
  fixed_frame_pose_sampling_ratio = 1.,
  imu_sampling_ratio = 1.,
  landmarks_sampling_ratio = 1.,
}

MAP_BUILDER.use_trajectory_builder_3d = true
MAP_BUILDER.num_background_threads = 16

-- One Hesai message is one full sweep.
TRAJECTORY_BUILDER_3D.num_accumulated_range_data = 1

TRAJECTORY_BUILDER_3D.min_range = 0.8    -- the operator's own body
TRAJECTORY_BUILDER_3D.max_range = 120.   -- keep the far facades (max return 151 m);
                                         -- 80 m costs ~1 cm of RMSE

-- Resolution. Points are voxel-filtered at 10 cm, but the high-resolution submap
-- the fine matcher scores against is 15 cm. At 5-10 cm a 32-beam sensor leaves
-- most cells of a far wall empty, and the trajectory drifts in height. Before the weight below: 0.27 m RMSE
-- with a 10 cm submap, 0.12 m at 15 cm, 0.16 m at 20 cm; at 5 cm not one loop
-- closure was accepted.
TRAJECTORY_BUILDER_3D.voxel_filter_size = 0.10
TRAJECTORY_BUILDER_3D.submaps.high_resolution = 0.15
TRAJECTORY_BUILDER_3D.submaps.low_resolution = 0.45

-- Fine matcher: near geometry (ground, nearby walls) up to 25 m. Coarse matcher
-- and low-resolution submap: everything, which keeps yaw locked along open facades.
TRAJECTORY_BUILDER_3D.high_resolution_adaptive_voxel_filter.max_length = 1.
TRAJECTORY_BUILDER_3D.high_resolution_adaptive_voxel_filter.min_num_points = 300
TRAJECTORY_BUILDER_3D.high_resolution_adaptive_voxel_filter.max_range = 25.
TRAJECTORY_BUILDER_3D.low_resolution_adaptive_voxel_filter.max_length = 4.
TRAJECTORY_BUILDER_3D.low_resolution_adaptive_voxel_filter.min_num_points = 200
TRAJECTORY_BUILDER_3D.low_resolution_adaptive_voxel_filter.max_range = 120.
TRAJECTORY_BUILDER_3D.submaps.high_resolution_max_range = 25.

-- Weight the fine (high-resolution) grid 3x instead of 1x, against the coarse
-- grid's stock 6x: 0.12 -> 0.093 m RMSE.
TRAJECTORY_BUILDER_3D.ceres_scan_matcher.occupied_space_weight_0 = 3.

-- 160 sweeps = 16 s per submap (the stock backpack value). Outdoors a submap must
-- span enough of the scene for loop closure to recognise it; 100 gave 0.24 m.
TRAJECTORY_BUILDER_3D.submaps.num_range_data = 160

-- One trajectory node per sweep, as in the basement config.
TRAJECTORY_BUILDER_3D.motion_filter.max_time_seconds = 0.05
TRAJECTORY_BUILDER_3D.motion_filter.max_distance_meters = 0.02
TRAJECTORY_BUILDER_3D.motion_filter.max_angle_radians = 0.002

POSE_GRAPH.optimize_every_n_nodes = 160
POSE_GRAPH.optimization_problem.ceres_solver_options.max_num_iterations = 20
POSE_GRAPH.optimization_problem.ceres_solver_options.num_threads = 16
-- huber_scale stays at Cartographer's default (1e1), unlike the basement config.

-- Loop closure. The walk circles a ~90 m building, so revisits come back with
-- metres of accumulated error: search out to 30 m (stock 15 m) and +-3 m in z
-- (stock 1 m, smaller than the height drift before closure). min_score is
-- Cartographer's own default 0.55; the basement's stricter 0.62 drops about a
-- fifth of the loop constraints and gave 0.35 m. Sampling 30 % of nodes, not 10 %.
POSE_GRAPH.constraint_builder.max_constraint_distance = 30.
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher_3d.linear_z_search_window = 3.
POSE_GRAPH.constraint_builder.sampling_ratio = 0.3
POSE_GRAPH.constraint_builder.min_score = 0.55
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.66
POSE_GRAPH.max_num_final_iterations = 200

return options
