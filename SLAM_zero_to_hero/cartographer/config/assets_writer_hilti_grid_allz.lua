-- 2D ROS occupancy grid carved out of the 3D Cartographer map: every z at once - deliberately shows why one 2D grid cannot hold this sequence
options = {
  tracking_frame = "imu_sensor_frame",
  pipeline = {
    { action = "min_max_range_filter", min_range = 0.8, max_range = 40. },

    { action = "dump_num_points" },
    {
      action = "write_ros_map",
      range_data_inserter = {
        insert_free_space = true,
        hit_probability = 0.55,
        miss_probability = 0.49,
      },
      filestem = "allz",
      resolution = 0.05,
    },
  }
}
return options
