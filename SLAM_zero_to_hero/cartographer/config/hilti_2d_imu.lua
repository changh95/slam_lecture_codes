-- Cartographer 2D + IMU for Hilti 2022 exp14_basement_2.bag -- the CHEAP comparison
-- point against hilti_3d.lua. Same TF/URDF, same tracking frame, same topics; the
-- only difference is which trajectory builder runs.
--
-- With use_imu_data = true and tracking_frame = the IMU frame, Cartographer
-- gravity-aligns every sweep before cropping it with min_z/max_z, so the z slab is
-- a true horizontal slab around the current sensor height instead of a slab in the
-- tilted sensor frame. That is the tilt-compensation fix. It cannot fix the ~4 m
-- level change in this sequence, because a single 2D grid has one floor.

include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
  map_frame = "map",
  tracking_frame = "imu_sensor_frame",   -- must be the IMU frame for gravity alignment
  published_frame = "imu_sensor_frame",
  odom_frame = "odom",
  provide_odom_frame = true,
  publish_frame_projected_to_2d = true,
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

MAP_BUILDER.use_trajectory_builder_2d = true
MAP_BUILDER.num_background_threads = 16

TRAJECTORY_BUILDER_2D.num_accumulated_range_data = 1
TRAJECTORY_BUILDER_2D.use_imu_data = true        -- <-- the change vs. the old config
TRAJECTORY_BUILDER_2D.min_range = 0.8
TRAJECTORY_BUILDER_2D.max_range = 40.
-- Gravity-aligned and relative to the CURRENT sensor height, so a thin level slice.
TRAJECTORY_BUILDER_2D.min_z = -0.4
TRAJECTORY_BUILDER_2D.max_z = 0.4
TRAJECTORY_BUILDER_2D.voxel_filter_size = 0.025
TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window = 0.1
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.translation_delta_cost_weight = 10.
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.rotation_delta_cost_weight = 1e-1
TRAJECTORY_BUILDER_2D.submaps.num_range_data = 100
TRAJECTORY_BUILDER_2D.submaps.grid_options_2d.resolution = 0.05
TRAJECTORY_BUILDER_2D.motion_filter.max_time_seconds = 0.05
TRAJECTORY_BUILDER_2D.motion_filter.max_distance_meters = 0.02
TRAJECTORY_BUILDER_2D.motion_filter.max_angle_radians = 0.002

POSE_GRAPH.optimize_every_n_nodes = 100
POSE_GRAPH.optimization_problem.huber_scale = 5e2
POSE_GRAPH.optimization_problem.ceres_solver_options.max_num_iterations = 20
POSE_GRAPH.optimization_problem.ceres_solver_options.num_threads = 16
POSE_GRAPH.constraint_builder.sampling_ratio = 0.10
POSE_GRAPH.constraint_builder.min_score = 0.62
POSE_GRAPH.max_num_final_iterations = 200

return options
