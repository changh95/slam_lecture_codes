-- Copyright 2016 The Cartographer Authors
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--      http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.
-- modified by Jason Kim 23-12-27


  include "map_builder.lua"
  include "trajectory_builder.lua"
  

  options = {
    map_builder = MAP_BUILDER,
    trajectory_builder = TRAJECTORY_BUILDER,
    map_frame = "map",
    tracking_frame = "velodyne", -- this should be the link of IMU 
    published_frame = "imu", -- use base_link while playing rosbag
    odom_frame = "odom",
    provide_odom_frame = true,
    publish_frame_projected_to_2d = true, 
    use_odometry = false,
    use_nav_sat = false,
    use_landmarks = false,

    num_laser_scans = 0,  -- to use topics like /scan   /laserscan
    num_multi_echo_laser_scans = 0,
    num_subdivisions_per_laser_scan = 1, 
    num_point_clouds = 1,   -- to use topics like /velodyne_points 

    lookup_transform_timeout_sec = 0.2,
    submap_publish_period_sec = 0.3,    
    pose_publish_period_sec = 5e-3,     
    trajectory_publish_period_sec = 30e-3, 
    rangefinder_sampling_ratio = 1.,
    odometry_sampling_ratio = 0.1, -- pay attention to this if above parameters are set properly but still have error related to absence of fixed frame map
    fixed_frame_pose_sampling_ratio = 0.1, 
    imu_sampling_ratio = 0.1,
    landmarks_sampling_ratio = 1.,
  }
  
  MAP_BUILDER.use_trajectory_builder_2d = true
  MAP_BUILDER.num_background_threads = 8 -- Increase up to number of cores
  --
  TRAJECTORY_BUILDER_2D.num_accumulated_range_data = 1 --default value is 10; If map is being built but acts weird, then change this value to 1
  --
  --******************BY DEFAULT( those set in map_builder.lua and trajectory_builder.lua )***********************************
  
  --TRAJECTORY_BUILDER.pure_localization=false -- not pure_localiztion
  
  TRAJECTORY_BUILDER_2D.min_range = 2.8 --0.3
  -- TRAJECTORY_BUILDER_2D.missing_data_ray_length = 1.
  TRAJECTORY_BUILDER_2D.use_imu_data = false --important, if this one is true ,then you need to set the track frame as imu. Besides, imu is not neccessary for 2d slam but neccessary for 3d cases
  TRAJECTORY_BUILDER_2D.min_z = -1.
  TRAJECTORY_BUILDER_2D.max_z = 1.
  TRAJECTORY_BUILDER_2D.submaps.grid_options_2d.resolution = 0.1 -- 0.05
   
  ------------------SCAN MATCHER (FRONT END----LOCAL MAP----Lidar Odometry)
  --There are two kind of scan matcher solution. One is CSM , another one is the optimizer Ceres given by Google. Ceres is set as default
  ------CSM parameter
  TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = false --true
  --TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window = 0.1
  --TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.translation_delta_cost_weight = 10.
  --TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.rotation_delta_cost_weight = 10.
  
  ------Ceres parameter
  -- TRAJECTORY_BUILDER_2D.voxel_filter_size = 0.05 --0.15
  TRAJECTORY_BUILDER_2D.ceres_scan_matcher.translation_weight = 1e-2 --10
  TRAJECTORY_BUILDER_2D.ceres_scan_matcher.rotation_weight = 1e-2 -- 40 ---4e2 originally , increase it for avoiding new submaps being added at an angle
  TRAJECTORY_BUILDER_2D.ceres_scan_matcher.occupied_space_weight = 1e1
  
  --POSE GRAPH problem, refer to Cartographer tuning file 
  POSE_GRAPH.optimization_problem.huber_scale = 5e2 --10
  POSE_GRAPH.optimization_problem.odometry_rotation_weight= 0
  
  -----------------TUNE THESE PARAMETERS FOR LOW LATENCY-------------------------------
  
  ------------Global SLAM (Graph optimization by Ceres, BACK END)------------
  POSE_GRAPH.optimize_every_n_nodes = 1 -- Decrease --original 1 | 90 (it should not be too large or there will be a lot of problem)
  POSE_GRAPH.global_sampling_ratio = 0.0001 -- Decrease 0.00001 originally (X)
  POSE_GRAPH.constraint_builder.sampling_ratio = 0.03 -- Decrease 0.0001 (X)
  POSE_GRAPH.constraint_builder.min_score = 0.62 -- Increase 0.8 
  --POSE_GRAPH.global_constraint_search_after_n_seconds = 20 -- Increase ,orginally 20 (X)
  POSE_GRAPH.optimization_problem.ceres_solver_options.max_num_iterations = 10
  POSE_GRAPH.constraint_builder.global_localization_min_score = 0.66
  --TRAJECTORY_BUILDER_2D.ceres_scan_matcher.ceres_solver_options.max_num_iterations = 5 -- Decrease
  
  ---------Global/Local SLAM(FIND LOOP CLOSURE)---------
  --TRAJECTORY_BUILDER_2D.adaptive_voxel_filter.min_num_points = 100 -- Decrease
  --TRAJECTORY_BUILDER_2D.adaptive_voxel_filter.max_range = 10. -- Decrease
  --TRAJECTORY_BUILDER_2D.adaptive_voxel_filter.max_length = 1.0 -- Increase
  --TRAJECTORY_BUILDER_2D.loop_closure_adaptive_voxel_filter.min_num_points = 40 -- Decrease  originally 50
  --TRAJECTORY_BUILDER_2D.loop_closure_adaptive_voxel_filter.max_range = 10. -- Decrease
  --TRAJECTORY_BUILDER_2D.loop_closure_adaptive_voxel_filter.max_length = 2.0 -- Increase 1.8
  --TRAJECTORY_BUILDER_2D.voxel_filter_size = 0.05 -- Increase
  --TRAJECTORY_BUILDER_2D.submaps.resolution = 0.05 -- Increase
  TRAJECTORY_BUILDER_2D.submaps.num_range_data = 30 -- Decrease   (important,control submap number)
  --TRAJECTORY_BUILDER_2D.max_range = 15. -- Decrease
  
  -------------------------------------------------------------------------------------
  
  return options

