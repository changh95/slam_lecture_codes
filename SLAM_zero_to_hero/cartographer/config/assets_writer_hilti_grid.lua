-- Turn the 3D Cartographer map into a real 2D ROS occupancy grid (.pgm + .yaml).
--
-- Do NOT use cartographer_pbstream_to_ros_map on a 3D pbstream: it renders the 3D
-- submaps' 2D *projection texture*, which carries almost no free-space information
-- (measured on this bag: free/occupied = 0.041). cartographer_assets_writer replays
-- the bag through the optimised pose graph and re-inserts real rays into a fresh 2D
-- probability grid with insert_free_space = true, which is what gives free space.
--
-- IMPORTANT: cartographer's `vertical_range_filter` min_z/max_z are measured
-- RELATIVE TO THE SENSOR ORIGIN of each batch, not in the map frame
-- (vertical_range_filtering_points_processor.cc:  position.z() - origin.z()).
-- So this is a 1 m slab that FOLLOWS the sensor down the 4.2 m level change --
-- the gravity-aligned equivalent of TRAJECTORY_BUILDER_2D's min_z/max_z, except
-- applied to an already globally consistent 3D map. Dropping the filter (see
-- assets_writer_hilti_grid_allz.lua) leaves floor and ceiling rings inside every
-- room: occupied cells go 4.96 % -> 6.28 % and free/occupied 5.15 -> 3.22.
options = {
  -- MUST match the tracking_frame of the run that produced the pbstream.
  tracking_frame = "imu_sensor_frame",
  pipeline = {
    {
      action = "min_max_range_filter",
      min_range = 0.8,
      max_range = 40.,
    },
    {
      action = "vertical_range_filter",
      min_z = -0.40,   -- relative to the sensor: 0.4 m below it
      max_z =  0.60,   -- to 0.6 m above it
    },
    {
      action = "dump_num_points",
    },
    {
      action = "write_ros_map",
      range_data_inserter = {
        insert_free_space = true,
        hit_probability = 0.55,
        miss_probability = 0.49,
      },
      filestem = "slab",
      resolution = 0.05,   -- same 0.05 m/px as the old baseline .pgm
    },
  }
}
return options
