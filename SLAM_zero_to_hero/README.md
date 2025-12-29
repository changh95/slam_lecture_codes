# SLAM_zero_to_hero

This repository contains code exercises for the lecture series - ['SLAM for Robotics & 3D Computer Vision' at FastCampus](https://fastcampus.co.kr/data_online_slam). This lecture series is delivered in Korean language.

![](title.png)

## How to use

Most of the code exercises are based on the base docker image. The base docker image contains numerous C++ libraries for SLAM, such as OpenCV, Eigen, Sophus, PCL, and ceres-solver.

You can build the base docker image using the following command.

```shell
docker build . --tag slam_zero_to_hero:base --progress=plain
echo "xhost +local:docker" >> ~/.profile
```

## Table of contents

- Chapter 1: Introduction to SLAM
  - 1.1 Lecture introduction
  - 1.2 Mobile robotics
  - 1.3 What is SLAM?
  - 1.4 Hardware used in SLAM
  - 1.5 Types of SLAM
  - 1.6 Applications of SLAM
  - 1.7 Tips for studying SLAM
  - 1.8 C++ and SLAM
  - 1.9 [Basic C++ programming](1_9)
  - 1.10 [Building C++ libraries](1_10)
  - 1.11 [C++ CPU profiler](1_11)
  - 1.12 [C++ memory profiler](1_12)
  - 1.13 Python basics
  - 1.14 [Basic Python programming](1_14)
  - 1.15 [PyBind](1_15)
  - 1.16 [ROS fundamentals](1_16)
  - 1.17 Rotation and translation in 3D space
  - 1.18 Homogeneous coordinates
  - 1.19 Lie Group
  - 1.20 Basics of Lie algebra
  - 1.21 [Eigen + Sophus library hands-on](1_21)
  - 1.22 Continuous-time representation
  - 1.23 Camera basics for robotics
  - 1.24 Camera models
  - 1.25 LiDAR basics
  - 1.26 IMU basics
  - 1.27 Radar basics
  - 1.28 Forward/Inverse kinematics
  - 1.29 Sensor calibration
  - 1.30 [Kalibr package hands-on](1_30)
- Chapter 2: Dive into SLAM (Front-end)
  - 2.1 Part 2 introduction
  - 2.2 Local feature detection
  - 2.3 [Classical local feature detection hands-on](2_3)
  - 2.4 [Deep local feature detection hands-on](2_4)
  - 2.5 Feature tracking basics
  - 2.6 Advanced feature tracking in practice
  - 2.7 [Feature tracking hands-on](2_7)
  - 2.8 Global feature detection
  - 2.9 [Global feature detection hands-on](2_9)
  - 2.10 [Deep global feature detection hands-on](2_10)
  - 2.11 Epipolar geometry
  - 2.12 [Epipolar geometry hands-on](2_12)
  - 2.13 Homography
  - 2.14 [Homography hands-on](2_14)
  - 2.15 [MonoVO hands-on](2_15)
  - 2.16 Triangulation
  - 2.17 [Triangulation hands-on](2_17)
  - 2.18 Perspective-n-points
  - 2.19 [Perspective-n-points hands-on](2_19)
  - 2.20 RANSAC
  - 2.21 Advanced RANSAC
  - 2.22 [RANSAC hands-on](2_22)
  - 2.23 M-estimator & MAXCON
  - 2.24 What is point cloud?
  - 2.25 Introduction to PCL library
  - 2.26 Point cloud preprocessing
  - 2.27 [Point cloud preprocessing hands-on](2_27)
  - 2.28 ICP
  - 2.29 [ICP hands-on](2_29)
  - 2.30 [Advanced ICP hands-on](2_30)
  - 2.31 [Octree, Octomap, Bonxai hands-on](2_31)
- Chapter 3: Dive into SLAM (Back-end)
  - 3.1 Part 3 introduction
  - 3.2 Factor graph
  - 3.3 Nonlinear least squares
  - 3.4 Nonlinear optimization
  - 3.5 Optimization on manifolds
  - 3.6 Graph-based SLAM
  - 3.7 Schur complement
  - 3.8 Auto-diff
  - 3.9 Continuous-time optimization
  - 3.10 Sparsity in SLAM
  - 3.11 Bundle adjustment
  - 3.12 Nonlinear solvers
  - 3.13 [g2o hands-on](3_13)
  - 3.14 [GTSAM hands-on](3_14)
  - 3.15 [Ceres-solver hands-on](3_15)
  - 3.16 [SymForce hands-on](3_16)
  - 3.17 SLAM systems
  - 3.18 Various map representations
  - 3.19 VSLAM system architecture
  - 3.20 LiDAR SLAM system architecture
  - 3.21 RADAR SLAM system architecture
  - 3.22 Event SLAM system architecture
  - 3.23 Inertial odometry basics
  - 3.24 Leg odometry basics
  - 3.25 Sensor fusion
- Chapter 4: Classical SLAM
  - 4.1 Part 4 introduction
  - 4.2 Feature-based VSLAM
  - 4.3 Direct VSLAM
  - 4.4 Visual-inertial odometry
  - 4.5 2D LiDAR SLAM
  - 4.6 3D LiDAR SLAM
  - 4.7 Sensor fusion SLAM
  - 4.8 ORB-SLAM 2
  - 4.9 Basalt-VIO
  - 4.10 Cartographer
  - 4.11 KISS-SLAM
  - 4.12 GLIM
  - 4.13 FAST-LIO2
  - 4.14 FAST-LIVO2
- Chapter 5: Advanced SLAM - AI Integration and Hardware Optimization
  - 5.1 Part 5 introduction
  - 5.2 SLAM + Object detection + Segmentation
  - 5.3 SLAM + Depth estimation
  - 5.4 SLAM + Camera pose regression
  - 5.5 SLAM + Deep feature matching
  - 5.6 SLAM + Deep optical flow / scene flow
  - 5.7 SLAM + Differentiable bundle adjustment
  - 5.8 SLAM + Feed-forward 3D transformer
  - 5.9 SLAM + NeRF / Implicit neural field
  - 5.10 SLAM + Gaussian Splatting
  - 5.11 SLAM + Video generation
  - 5.12 SLAM + VLM/VLA
  - 5.13 SLAM + 3D Scene graph
  - 5.14 SLAM + Certifiably optimal algorithm
  - 5.15 SLAM + Auto-encoder / diffusion
  - 5.16 SLAM + Graph processor
  - 5.17 DSP-SLAM
  - 5.18 Kimera
  - 5.19 ConceptFusion
  - 5.20 Gaussian Splatting SLAM
  - 5.21 MASt3r-SLAM
  - 5.22 PIN-SLAM
  - 5.23 Suma++
  - 5.24 Differences between desktop, server, and embedded boards
  - 5.25 Characteristics of real-time SLAM
  - 5.26 Characteristics of auto-labeling / data-crunching SLAM
  - 5.27 C++ build configuration optimization
  - 5.28 SIMD acceleration and CPU optimization techniques
  - 5.29 [SIMD acceleration hands-on](5_29)
  - 5.30 Introduction to NVIDIA Jetson
  - 5.31 [CUDA acceleration hands-on](5_31)
- Final projects
  - Project 1: SLAM for autonomous driving
  - Project 2: SLAM for drones
  - Project 3: SLAM for mobile scanner systems
  - Project 4: SLAM for quadruped robots
  - Project 5: SLAM for humanoid robots
  - Project 6: SLAM for VR/AR headsets

## Acknowledgements

ORB-SLAM 2/3 authors, Basalt-VIO authors, Cartographer authors, KISS-SLAM authors, GLIM authors, FAST-LIO2 authors, FAST-LIVO2 authors, DSP-SLAM authors, Kimera authors, ConceptFusion authors, MASt3r-SLAM authors, PIN-SLAM authors, Suma++ authors, and all the authors of the libraries used in this repository.

## Contributors

Thanks goes to these wonderful people:

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
