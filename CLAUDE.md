This is a repository for code exercises for changh95's SLAM courses.

There are two lectures:
1. Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving (in ./Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving folder)
2. SLAM_zero_to_hero (in ./SLAM_zero_to_hero folder)

Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving course is already complete. Claude may refer to this source code to get idea of folder hierarchy, source code, and code formats.

Main focus now is the SLAM_zero_to_hero course.

The code base comprises of C++ and Python codes. The base environment for running the code is Docker (or Podman). ROS1 and ROS2 will be utilized for open-source SLAM projects (e.g. ORB-SLAM, Basalt-VIO, Cartographer, DSP-SLAM, Kimera etc).

When Claude makes any changes to the code, the following sub-agents will operate the following:

1. Markdown Inspector: Any updates to the code, the Markdown Inspector will see if any README.md or markdown files should be updated.
2. Source Code Inspector: Any updates to the code, the Source Code Inspector will see if any related source code needs to be updated (e.g. header file, nanobind, CMakeLists.txt).
3. Dockerfile Inspector: Any updates to the code, the Dockerfile Inspector will see if the Dockerfile needs to be updated.
4. Testor: Any updates to the code, the Testor will build the Dockerfile and see if the code actually runs without issues.
5. Cross-Reference Checker: Ensure folder names (e.g. 2_3, 3_14) match README table of contents entries, and hyperlinks point to existing folders.
6. Exercise Template Generator: For hands-on exercises, create consistent structure (README with instructions, starter code, solution folder, sample data paths).
