-- cartographer_assets_writer pipeline for the outdoor Hilti run (exp21): dense 3D
-- cloud. Same as assets_writer_hilti.lua but keeps returns out to 80 m (the SLAM
-- config's max_range) and de-duplicates into 10 cm voxels, the SLAM resolution.
-- tracking_frame MUST match the run that produced the pbstream.
options = {
  tracking_frame = "imu_sensor_frame",
  pipeline = {
    { action = "min_max_range_filter", min_range = 0.8, max_range = 80. },
    { action = "dump_num_points" },
    -- drop voxels traversed far more often than hit (pedestrians, the operator)
    { action = "voxel_filter_and_remove_moving_objects", voxel_size = 0.10, miss_per_hit_limit = 3. },
    { action = "dump_num_points" },
    { action = "write_ply", filename = "map3d.ply" },
    { action = "write_pcd", filename = "map3d.pcd" },
  }
}
return options
