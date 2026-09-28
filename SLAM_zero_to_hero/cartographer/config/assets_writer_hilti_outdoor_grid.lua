-- 2D ROS occupancy grid (slab.pgm + slab.yaml) for the outdoor Hilti run (exp21),
-- re-inserting real rays (with free space) through the optimised 3D pose graph.
-- See assets_writer_hilti_grid.lua for why this beats cartographer_pbstream_to_ros_map
-- and why vertical_range_filter's z is RELATIVE TO THE SENSOR: this keeps a 1 m slab
-- around the handheld sensor, i.e. building walls, and leaves out the ground.
options = {
  tracking_frame = "imu_sensor_frame",
  pipeline = {
    { action = "min_max_range_filter", min_range = 0.8, max_range = 60. },
    { action = "vertical_range_filter", min_z = -0.40, max_z = 0.60 },
    { action = "dump_num_points" },
    {
      action = "write_ros_map",
      range_data_inserter = {
        insert_free_space = true,
        hit_probability = 0.55,
        miss_probability = 0.49,
      },
      filestem = "slab",
      resolution = 0.10,   -- the SLAM resolution; a 150 x 150 m area stays ~1500 px
    },
  }
}
return options
