-- Cartographer 3D (LiDAR + IMU) for the Hilti SLAM Challenge 2022 sequence
-- exp14_basement_2.bag.
--
-- Sensors in the bag:
--   /hesai/pandar    sensor_msgs/PointCloud2, frame_id "PandarXT-32", 10 Hz,
--                    ~52k points/sweep, 32 beams
--   /alphasense/imu  sensor_msgs/Imu, frame_id "imu_sensor_frame", 399 Hz, SI units.
--                    At rest acc = (0.17, 0.16, -9.67): this IMU's +z points DOWN.
--                    Cartographer needs no help with that -- ImuTracker seeds
--                    gravity_vector_ from the FIRST measurement with alpha = 1 and
--                    then defines the map frame so that measured specific force maps
--                    to +z, i.e. the map frame comes out z-UP and the initial
--                    tracking-frame orientation is a ~180 deg flip. (Verified: the
--                    first exported quaternion is (x,y,z,w) ~= (0.685,-0.728,0,0.013).)
--
-- The bag publishes NO /tf, so the LiDAR<-IMU extrinsic must be supplied out of band:
--   cartographer_offline_node  -urdf_filenames hilti_alphasense_pandar.urdf
--   cartographer_node          a tf2_ros static_transform_publisher works too:
--     rosrun tf2_ros static_transform_publisher \
--       -0.001 -0.00855 0.055 0.7071068 -0.7071068 0 0 imu_sensor_frame PandarXT-32
--
-- Topic remaps: points2:=/hesai/pandar  imu:=/alphasense/imu

include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
  map_frame = "map",
  -- Cartographer 3D integrates the IMU directly in the tracking frame with no
  -- IMU-to-tracking extrinsic, so tracking_frame MUST be the IMU frame. Using the
  -- LiDAR frame here is the single most common way to break a 3D run.
  tracking_frame = "imu_sensor_frame",
  published_frame = "imu_sensor_frame",
  odom_frame = "odom",
  provide_odom_frame = true,
  publish_frame_projected_to_2d = false,   -- this traverse changes level by ~4.2 m
  use_pose_extrapolator = true,
  use_odometry = false,
  use_nav_sat = false,
  use_landmarks = false,
  num_laser_scans = 0,
  num_multi_echo_laser_scans = 0,
  num_subdivisions_per_laser_scan = 1,
  num_point_clouds = 1,                    -- one PointCloud2 topic, "points2"
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

-- One Hesai sweep is already ~52k points, i.e. a complete frame. backpack_3d.lua
-- uses 160 only because that bag splits each VLP-16 rotation into many small
-- PointCloud2 messages. Accumulating here would throw away scan-matching rate and
-- make the trajectory too coarse (10 Hz -> 0.06 Hz).
TRAJECTORY_BUILDER_3D.num_accumulated_range_data = 1

TRAJECTORY_BUILDER_3D.min_range = 0.8   -- handheld: reject returns off the operator
                                        -- (0.5, matching FAST-LIO2's blind, changes
                                        --  nothing measurable: arbiter 0.595/0.520/0.432)
TRAJECTORY_BUILDER_3D.max_range = 40.   -- nothing useful beyond 40 m in a basement

-- 5 cm working resolution throughout. The stock 15 cm / 10 cm / 45 cm triple is
-- sized for an outdoor backpack run; in 2-4 m corridors it quantises away exactly
-- the wall detail the ceres matcher needs.
TRAJECTORY_BUILDER_3D.voxel_filter_size = 0.05
TRAJECTORY_BUILDER_3D.submaps.high_resolution = 0.05
TRAJECTORY_BUILDER_3D.submaps.low_resolution = 0.30

-- Shrink the "high resolution" horizon from 15/20 m to 12 m so the fine matcher
-- spends its point budget on the near walls, floor and ceiling instead of spreading
-- it down an empty corridor.
TRAJECTORY_BUILDER_3D.high_resolution_adaptive_voxel_filter.max_range = 12.
TRAJECTORY_BUILDER_3D.high_resolution_adaptive_voxel_filter.min_num_points = 200
TRAJECTORY_BUILDER_3D.low_resolution_adaptive_voxel_filter.max_range = 40.
TRAJECTORY_BUILDER_3D.submaps.high_resolution_max_range = 12.

-- 10 Hz * 1 accumulation -> 100 range data = 10 s = ~5 m walked per submap, giving
-- 8 submaps over this 74 s bag. The backpack value of 160 would give 5, too few for
-- the pose graph to find loop constraints on a sequence this short.
TRAJECTORY_BUILDER_3D.submaps.num_range_data = 100

-- Keep one trajectory node per sweep. The stock motion filter (0.1 m / 0.004 rad /
-- 0.5 s) drops roughly half of a 0.54 m/s handheld walk, which thins the pose graph
-- and leaves the exported trajectory too sparse to evaluate per scan.
TRAJECTORY_BUILDER_3D.motion_filter.max_time_seconds = 0.05
TRAJECTORY_BUILDER_3D.motion_filter.max_distance_meters = 0.02
TRAJECTORY_BUILDER_3D.motion_filter.max_angle_radians = 0.002

POSE_GRAPH.optimize_every_n_nodes = 100
POSE_GRAPH.optimization_problem.huber_scale = 5e2
POSE_GRAPH.optimization_problem.ceres_solver_options.max_num_iterations = 20
POSE_GRAPH.optimization_problem.ceres_solver_options.num_threads = 16
POSE_GRAPH.constraint_builder.sampling_ratio = 0.10
POSE_GRAPH.constraint_builder.min_score = 0.62
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.66
POSE_GRAPH.max_num_final_iterations = 200

return options
