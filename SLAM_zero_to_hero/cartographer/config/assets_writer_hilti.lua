-- cartographer_assets_writer pipeline for the Hilti 3D run.
-- tracking_frame MUST match the tracking_frame of the run that produced the
-- pbstream, otherwise every scan is placed with the wrong body-to-sensor offset.
options = {
  tracking_frame = "imu_sensor_frame",
  pipeline = {
    {
      action = "min_max_range_filter",
      min_range = 0.8,
      max_range = 40.,
    },
    {
      action = "dump_num_points",
    },
    -- de-duplicate into 5 cm voxels and drop voxels that are traversed far more
    -- often than they are hit (moving people, the operator's own body).
    {
      action = "voxel_filter_and_remove_moving_objects",
      voxel_size = 0.05,
      miss_per_hit_limit = 3.,
    },
    {
      action = "dump_num_points",
    },
    {
      action = "write_ply",
      filename = "map3d.ply",
    },
    {
      action = "write_pcd",
      filename = "map3d.pcd",
    },
  }
}
return options
